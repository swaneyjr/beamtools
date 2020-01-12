import numpy as np
import matplotlib.pyplot as plt


class Intersection():

    def __init__(self, alignment_set):
        
        res_x = alignment_set.res_x
        res_y = alignment_set.res_y
        self.aligns = alignment_set.to_dict()
        self.phones = set(self.aligns.keys())
        aligns = set(self.aligns.values()) 

        # first find grid range
        x_offsets = [align.x for align in aligns]
        y_offsets = [align.y for align in aligns]
        max_phi, max_ux, max_uxy, max_uy = np.amax(np.abs([align.to_array()[4:] for align in aligns]), axis=0)

        dx = np.diff(x_offsets)[0]
        dy = np.diff(y_offsets)[0]

        self.x_off = max(x_offsets)-(res_x*(max_ux+np.cos(max_phi)) + res_y*(max_uxy+np.sin(max_phi)))/2
        self.y_off = max(y_offsets)-(res_y*(max_uy+np.cos(max_phi)) + res_x*(max_uxy+np.sin(max_phi)))/2

        self.x_tot = np.ceil(res_x*(max_ux+np.cos(max_phi))+res_y*(max_uxy+np.sin(max_phi)) - np.abs(dx)).astype(int)
        self.y_tot = np.ceil(res_y*(max_uy+np.cos(max_phi))+res_x*(max_uxy+np.sin(max_phi)) - np.abs(dy)).astype(int)

        # calculate intersection
        bx = np.arange(0.5, res_x, 1/np.sqrt(2))
        by = np.arange(0.5, res_y, 1/np.sqrt(2))

        bxi, byi = np.meshgrid(bx, by)

        bxi = (bxi - res_x / 2).flatten()
        byi = (byi - res_y / 2).flatten()

        interior_coords = []
        for iphone in phones:
            ali = alignment_set[iphone]
            interior_x, interior_y = ali.sensor_map(bxi, byi)
            for jphone in phones:
                if iphone == jphone: continue
                alj = alignment_set[jphone]

                bxj, byj = alj.inverse_map(interior_x, interior_y)
                    
                intersect = (np.abs(bxj) < res_x/2) & (np.abs(byj) < res_y/2)
                interior_x = interior_x[intersect]
                interior_y = interior_y[intersect]
                
            interior_x = (interior_x - self.x_off).astype(int)
            interior_y = (interior_y - self.y_off).astype(int)
            interior_coords.append(interior_x + self.x_tot * interior_y)
                
        interior_all = np.unique(np.hstack(interior_coords))
        self.binary = np.zeros((self.x_tot, self.y_tot), dtype=bool)
        self.binary[interior_all % self.x_tot, interior_all // self.x_tot] = True


    def cut_edge(self, n):
        binary_padded = np.pad(self.binary, ((n, n),(n,n)), 'constant')
        interior = binary_padded.copy()
        for roll in range(-n, n+1):
            for ax in (0,1):
                interior &= np.roll(binary_padded, roll, axis=ax)

        return interior[n:-n, n:-n]
             

    def add_spills(self, spills, filetype='align'):
        profiles = {}
        
        for p in self.phones:
            profile = 0

            for i,spl in enumerate(spills):

                for t in spl[p]:
                    f = spl.get_file(phone, t, filetype)
                    if not f['x'].size: continue

                    intersect = np.ones(f['x'].size, dtype=bool)
                    for iphone in phones:
                        if iphone == phone: continue
                        intersect &= f[iphone]

                    x = f['x'][intersect] - self.x_off
                    y = f['y'][intersect] - self.y_off

                    # dither
                    x += np.random.random(x.size) - 0.5
                    y += np.random.random(x.size) - 0.5

                    profile_temp += csr_matrix((np.ones(x.size), (x, y)), shape=(self.x_tot, self.y_tot))

                    f.close()

            if not (i+1)%10 or i==len(spills)-1:
                profile += profile_temp.toarray()
                profile_temp = 0

            profiles[p] = profile

        return Profile(self, profiles)
   


class Profile(Intersection):

    def __init__(self, intersect, hists, noise):
        self = intersect
        self._hists = hists
        self._noise = noise
        self._coeffs = {p: [1] for p in self.phones}
        self._coeffs_order = 0

    def frame_probability(self, x, y, p=None):
        raise NotImplementedError

    def hit_probability(self, x, y, p=None):
        if p:
            return self.hists[p][x,y] / self._hists[p].sum()
        else:
            return np.mean([hist[x,y] / hist.sum() for hist in self._hists])


    def coeff(self, order, bin_sz=1, cache=True, p=None):

        if not cache or self._coeffs_order < order:


            xpad = (bin_sz - self.x_tot % bin_sz) % bin_sz
            ypad = (bin_sz - self.y_tot % bin_sz) % bin_sz
            
            for p in phones:
                profile_pad = np.pad(self._hists[p], ((0, xpad), (0, ypad)), 'constant')
                profile_ds = profile_pad.reshape(profile_pad.shape[0]//bin_sz, 
                        bin_sz, 
                        profile_pad.shape[1]//bin_sz, 
                        bin_sz).sum((1,3))

                self._coeffs[p] = [1]
                profile_o = 1
                for o in range(order):
                    profile_o *= profile_ds - o
                    self._coeffs[p].append(profile_o.sum() / (profile_ds.sum()**(o+1) * bin_sz**(2*o)))

            self._coeffs_order = order

        if p:
            return self._coeffs[p][order]
        else:
            return np.mean([coeff[order] for coeff in self._coeffs])


    def add_noise(self, hists, visualize=False):
        
        templates = {}

        bx = np.arange(self.x_tot)
        by = np.arange(self.y_tot)
        bxx, byy = np.meshgrid(bx, by)


        for p, hist in hists.items():
    
            bxx_sensor, byy_sensor = self.aligns[p].inverse_map((bxx + self.x_off).flatten(), \
                                                     (byy + self.y_off).flatten())

            # now map to coordinates of the noise histograms
            cut = (np.abs(bxx_sensor) < hist.shape[0] / 2) \
                     & (np.abs(byy_sensor) < hist.shape[1] / 2)

            bxx_sensor = (bxx_sensor + hist.shape[0] / 2).astype(int)[cut]
            byy_sensor = (byy_sensor + hist.shape[1] / 2).astype(int)[cut]

            values_cut = hist[bxx_sensor, byy_sensor]

            template = np.zeros((self.x_tot, self.y_tot))
            template[bxx.flatten()[cut], byy.flatten()[cut]] = values_cut
            template *= self.binary

            templates[p] = template

            if visualize:
                plt.figure()
                plt.imshow(template, cmap='viridis', extent=[0, self.x_tot, 0, self.y_tot])
                plt.colorbar()
                plt.title('Noise: {}'.format(p[:6]))
                plt.show()

        return Profile(self, templates)


    def get_noise(self, p, edge=0):
        mask = self.cut_edge(edge) if edge else 1
        return self._noise[p] * mask

