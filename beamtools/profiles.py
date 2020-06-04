import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import factorial
import matplotlib.pyplot as plt

from .alignment import AlignmentSet

class Intersection():

    def __init__(self, alignment_set):
        
        res_x = alignment_set.res_x
        res_y = alignment_set.res_y
        self.aligns = alignment_set.to_dict()
        self.phones = set(self.aligns.keys())
        aligns = set(self.aligns.values()) 
        root = alignment_set.root

        # first find grid range
        x_offsets = [align.x for align in aligns] + [0]
        y_offsets = [align.y for align in aligns] + [0]
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
        for iphone in self.phones:

            # root is zero align, so identity transformation
            interior_x, interior_y = (bxi, byi) if iphone == root \
                    else alignment_set[iphone].sensor_map(bxi, byi)
            for jphone in self.phones:
                if iphone == jphone: continue

                bxj, byj = (interior_x, interior_y) if jphone == root \
                        else alignment_set[jphone].inverse_map(interior_x, interior_y)
                    
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
             

    def to_profile(self, spills=None, noise=None, filetype='align'):

        spill_pf = {p: np.zeros((self.x_tot, self.y_tot)) for p in self.phones}
        noise_pf = {p: np.zeros((self.x_tot, self.y_tot)) for p in self.phones}

        # first, find profile with beam on
        if spills:
            for p in spills.phones:

                # faster alternative to adding all sparse matrices together
                profile_temp = 0

                for i,spl in enumerate(spills):

                    if not p in spl.phones: continue

                    for t in spl[p]:
                        f = spl.get_file(p, t, filetype)
                        if not f['x'].size: continue

                        intersect = np.ones(f['x'].size, dtype=bool)
                        for iphone in self.phones:
                            if iphone == p: continue
                            intersect &= f[iphone]

                        x = f['x'][intersect] - self.x_off
                        y = f['y'][intersect] - self.y_off

                        # dither
                        x += np.random.random(x.size) - 0.5
                        y += np.random.random(x.size) - 0.5

                        profile_temp += csr_matrix((np.ones(x.size), (x, y)), shape=(self.x_tot, self.y_tot))

                        f.close()

                if not (i+1)%10 or i==len(spills)-1:
                    spill_pf[p] += profile_temp.toarray()
                    profile_temp = 0
 
        else:
            spill_pf = None


        # now calculate noise profile

        if noise:

            bx = np.arange(self.x_tot)
            by = np.arange(self.y_tot)
            bxx, byy = np.meshgrid(bx, by)

            for p in noise.phones:

                hist = noise[p]
                
                bxx_sensor, byy_sensor = self.aligns[p].inverse_map(
                        (bxx + self.x_off).flatten(), 
                        (byy + self.y_off).flatten())

                # now map to coordinates of the noise histograms
                cut = (np.abs(bxx_sensor) < hist.shape[0] / 2) \
                    & (np.abs(byy_sensor) < hist.shape[1] / 2)

                bxx_sensor = (bxx_sensor + hist.shape[0] / 2).astype(int)[cut]
                byy_sensor = (byy_sensor + hist.shape[1] / 2).astype(int)[cut]

                hist_cut = hist[bxx_sensor, byy_sensor]
                noise_pf[p][bxx.flatten()[cut], byy.flatten()[cut]] = hist_cut
                noise_pf[p] *= self.binary


        return Profile(self, spill_pf, noise_pf)
   


class Profile(Intersection):

    def __init__(self, intersect, spill_pf, noise_pf=None, noise_frames=None):
        super().__init__(AlignmentSet(intersect.aligns))

        self.spill_profile = spill_pf
        self.noise_profile = noise_pf
        self.noise_frames = noise_frames
        self._coeffs = {p: [1] for p in self.phones}
        self._coeffs_order = 0
        self._bin_sz = 0

    def frame_probability(self, phone, x, y):
        raise NotImplementedError

    def hit_probability(self, phone, x, y):
        return self.spill_profile[p][x,y] / self.spill_profile[p].sum()

    def coeff(self, order, bin_sz=1, phone=None, cache=True):

        if not cache or self._coeffs_order < order or self._bin_sz != bin_sz:

            if not self.spill_profile:
                self._coeffs = {p: [1] + [1/np.sum(self.binary)**o for o in range(order)] for p in self.phones}

            else:
                padding = [(bin_sz-1, bin_sz-1), (bin_sz-1, bin_sz-1)]
                for p in self.phones:
                    
                    profile_pad = np.pad(self.spill_profile[p], padding, mode='constant')
                    intersect_pad = np.pad(self.binary, padding, mode='constant')
        
                    profile_sum = 0
                    profile_counts = 0
                    profile_max = 0
        
                    for ix, iy in np.ndindex((2*bin_sz-1, 2*bin_sz-1)):
                
                        slice_x = slice(ix, ix-2*(bin_sz-1) or None)
                        slice_y = slice(iy, iy-2*(bin_sz-1) or None)

                        profile_slice = profile_pad[slice_x, slice_y]
                        profile_sum += profile_slice
                        profile_counts += intersect_pad[slice_x, slice_y]
                        profile_max = np.maximum(profile_max, profile_slice)

                    profile_counts *= self.binary
        
                    profile_sum = profile_sum[profile_counts > 0]
                    profile_max =  profile_max[profile_counts > 0]
                    profile_counts = profile_counts[profile_counts > 0] 
        
                    profile_mean = profile_sum / profile_counts
                    poisson_max = profile_counts * profile_mean**profile_max * np.exp(-profile_mean) / factorial(profile_max)
                    outliers = poisson_max < 1e-5
        
                    profile_counts[outliers] -= 1
                    profile_sum[outliers] -= profile_max[outliers]

                    self._coeffs[p] = [1]
                    profile_o = 1

                    for o in range(order):
                        profile_o *= profile_sum - o
                        self._coeffs[p].append(np.sum(profile_o / profile_counts**(2*o)) / profile_pad.sum()**(o+1))

            self._bin_sz = bin_sz
            self._coeffs_order = order

        if phone:
            return self._coeffs[phone][order]
        return np.median([self._coeffs[p] for p in self.phones], axis=0)[order]


    def get_noise(self, phone, edge=0):
        mask = self.cut_edge(edge) if edge else 1
        return self.noise_profile[phone] * mask


    def visualize(self, ds=97):

        for p in self.phones:
            plt.figure(figsize=(6, 8))

            spl = self.spill_profile[p]
            ds_x = spl.shape[0] // ds
            ds_y = spl.shape[1] // ds
            spl = spl[:ds_x*ds, :ds_y*ds]
            spl = spl.reshape(ds_x, ds, ds_y, ds)
            noise = self.noise_profile[p]
            noise = noise[:ds_x*ds, :ds_y*ds]
            noise = noise.reshape(ds_x, ds, ds_y, ds)

            plt.subplot(211)
            plt.imshow(np.sum(spl, axis=(1,3)).transpose(), 
                    cmap='plasma', 
                    extent=[0, self.x_tot, 0, self.y_tot])
            plt.colorbar()
            plt.title('Beam: {}'.format(p[:6]))

            plt.subplot(212)
            plt.imshow(np.sum(noise, axis=(1,3)).transpose(), 
                    cmap='viridis', 
                    extent=[0, self.x_tot, 0, self.y_tot])
            plt.colorbar()
            plt.title('Noise: {}'.format(p[:6]))
            
            plt.show()

            

