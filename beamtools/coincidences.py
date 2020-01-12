import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import itertools as it
import pickle

from .spills import SpillSet


class Coincidence():
    def __init__(self, spills, profile, hist=None, cshapes={}):

        self.spills = spills
        self.profile = profile
        self.phones = sorted(self.profile.phones())

        self._hist = hist
        self._cshapes = cshapes
        self._csizes = {}
        self._optimal = {} 

    @staticmethod
    def from_pkl(fname):
        return pickle.load(fname)

    def to_pkl(self, fname):
        pickle.dump(self, fname)

    
    def compute(self, n_spills=50, s=15, filetype='align'):
        r = s // 2
 
        interior = self.profile.cut_edge(2*r)
        dim = (s,) * (2*len(self.phones)-2)
        self._hist = np.zeros(dim)
        mat_shape = (self.profile.x_tot, self.profile.y_tot)

        for spl in self.spills[:n_spills]:

            sparse_i = None
            xy_j = {}

            for t, p, overlap in spl.gen_overlaps_single(self.phones):

                ip = np.argmax(np.array(self.phones) == p)

                f = Spill.get_file(p, t, filetype=filetype)
                    
                intersect = np.ones(f['x'].size, dtype=bool)
                for iphone in self.phones:
                    if iphone == p: continue
                    intersect &= f[iphone]

                x_new = (f['x'][intersect] - self.profile.x_off).astype(int)
                y_new = (f['y'][intersect] - self.profile.y_off).astype(int)

                if ip == 0:
                    interior_new = interior[x_new, y_new]
                    x_int = x_new[interior_new]
                    y_int = y_new[interior_new]

                    sparse_i = csr_matrix((np.ones(x_int.size), (x_int, y_int)), shape=mat_shape)

                else:
                    xy_j[ip-1] = (x_new, y_new)
                    
                
                for indices in np.ndindex(*dim):

                    if sparse_i is None or len(xy_j.keys()) < len(self.phones) - 1: continue

                    sparse_tot = sparse_i.copy()
                    
                    for i,ixy in enumerate(np.array_split(indices, len(indices)//2)):
                        
                        ix, iy = ixy            
                        xj, yj = xy_j[i]
                        
                        xj += ix - r
                        yj += iy - r
                        
                        sparse_tot *= csr_matrix((np.ones(xj.size), (xj, yj)), shape=mat_shape)
                    
                    self._hist[indices] += sparse_tot.sum() # TODO: subtract background
                                    
                f.close()
        

    def show(self, p1, p2):
        plt.figure()
        plt.title('Hit separation: ({}, {})'.format(p1[:6], p2[:6]))
        plt.xlabel(r'$\Delta x$ (pixels)')
        plt.ylabel(r'$\Delta y$ (pixels)')
        
        hist = self._hist_reduce(p1, p2)
        xy_r = hist.shape[0] // 2
        # flip if out of order
        if p1 > p2:
            hist = hist[::-1, ::-1]

        plt.imshow(hist, extent=[-xy_r-0.5, xy_r+0.5, -xy_r-0.5, xy_r+0.5], norm=LogNorm())
        plt.colorbar()
        plt.show()


    def set_conv(self, p1, p2, conv_shape):
        if type(conv_shape) == int:
            conv_shape = (conv_shape, conv_shape) 
        else:
            conv_shape = tuple(conv_shape)
            if not len(conv_shape) == 2:
                raise ValueError('Arguments must be integers or tuples of integers')
        
        self._cshapes[frozenset([p1, p2])] = conv_shape
        # clear cache
        self._csizes = {}


    def show_all(self):
        for p1, p2 in powerset(self.phones, min_size=2, max_size=2):
            self.show(p1, p2)

    def set_conv_all(self, *conv_shapes):
        if len(conv_shapes) != len(self.phones) * (len(self.phones) - 1) // 2:
            raise ValueError('Incorrect number of input arguments')

        for c, conv in zip(powerset(self.phones, min_size=2, max_size=2), conv_shapes):
            self.set_conv(*c, conv)

    def _hist_reduce(self, *phones):
        
        if len(phones) == 1:
            return np.array([1])

        set_phones = set(phones)
        all_phones = set(self.phones)

        if not set_phones.issubset(all_phones):
            raise ValueError

        reduce_phones = np.array(list(all_phones - set_phones))
        reduce_idx = np.argmax(reduce_phones.reshape(-1,1) == np.sort(phones), axis=1)
        
        reduce_axes = []
        for idx in reduce_idx:
            if idx > 0:
                reduce_axes += [2*idx-2, 2*idx-1]
        
        hist_reduced = self._hist.sum(axis=reduce_axes)

        s = hist_reduced.shape[0]
        if 0 in reduce_idx: 
            hist_reordered = np.zeros(hist_reduced.shape[2:])
            n_axes = len(hist_reordered.shape)

            for ix, iy in np.ndindex((s,s)):
                hist_roll = hist_reduced[ix,iy]

                dx = ix-r
                dy = iy-r
                hist_roll = np.roll(hist_roll, -dx, axis=np.arange(0, n_axes, 2))
                hist_roll = np.roll(hist_roll, -dy, axis=np.arange(1, n_axes, 2))
                
                # set elements rolled through to zero
                slx = slice(dx) if dx < 0 else slice(-dx, None)
                sly = slice(dy) if dy < 0 else slice(-dy, None)

                hist_roll[(slx, sly) * (n_axes//2)] = 0

                hist_reordered += hist_roll

            return hist_reordered

        return hist_reduced


    def subset(self, phones, profile=None):
        profile = profile or self.profile

        hist = self._hist_reduce(*phones)
        cshapes = {fs: cshape for fs, cshape in self._cshapes.items() if fs.issubset(set(phones))}
        return Coincidence(spills, profile, hist, cshapes)


    def convolve_sparse(self, x, y, phones, mat_shape):
        phones = frozenset(phones)
        cshape = np.array([1,1]) if len(phones) == 1 else self._cshapes[phones]

        center = np.array(cshape) // 2
        conv_x = []
        conv_y = []
        
        for idx, idy in np.ndindex(cshape):
            ix = x + idx-center[0]
            iy = y + idy-center[1]
            valid = (ix > 0) & (iy > 0) & (ix < mat_shape[0]) & (iy < mat_shape[1])

            conv_x.append(ix[valid])
            conv_y.append(iy[valid])
        
        conv_x = np.hstack(conv_x)
        conv_y = np.hstack(conv_y)
        
        return csr_matrix((np.ones(conv_x.size), (conv_x, conv_y)), shape=mat_shape)
   

    def get_effective_convolution(self, pi, s):

        if not (pi, frozenset(s)) in self._csizes:

            # calculate and cache result
            
            s_conv = np.hstack([self._cshapes[frozenset([pi, si])] for si in sorted(s)])
            hist = self._hist_reduce(s)
            hist /= hist.sum()
                                    
            conv_tot = 0
            
            # iterate over positions for particle to be incident on min_conv sensor
            for mid in map(np.array, np.ndindex(*s_conv)):
                start = np.maximum(mid - s_conv//2, 0)
                end = np.minimum(mid + s_conv//2, s_conv)
                
                slices = []
                for i,f in zip(start, end):
                    slices.append(slice(i,f))
                
                conv_tot += hist[tuple(slices)]
            
            self._csizes[(pi, s)] = conv_tot

        return self._csizes[(pi, frozenset(s))]


    def optimal(self, c):

        c = frozenset(c)

        if not c in self._optimal:
            # determine the fastest way to convolve
            for subset in powerset(c, min_size=2):
                pi_best = None
                for pi in subset:
                    total_conv = 0
                    for pj in subset:
                        if pi is pj: continue
                        total_conv += self.get_effective_convolution(pi, [pj])
                
                    if pi_best == None or total_conv < min_conv:
                        min_conv = total_conv
                        pi_best = pi
                
                self._optimal[subset] = pi_best   


        p_opt = self._optimal[frozenset(c)]
        return p_opt, c - set([p_opt])
 

    def get_efficiency(self, threshold=-1, verbose=True, visualize=True):

        edge_size = max([max(conv) for conv in self._cshapes.values()]) // 2
        noise = {p: self.profile.get_noise(p, edge=edge_size)}
        
        interior = self.profile.cut_edge(edge_size)

        n_all = []
        n_single = []
        
        for n_spills, spl in enumerate(self.spills):
            if verbose:
                print("{:.2f}%".format(100*n_spills/len(self.spills)), end="\r")
            
            all_spl = 0
            single_spl = 0
            
            # keep a cache of results
            n_coinc = {}
            n_hits_interior = {}
            n_hits_corr = {}
            first_last = {p: True for p in self.phones}
            conv_ij = {p: {} for p in self.phones}

            for t, p, overlap in spl.gen_overlaps_single(self.phones):
                
                # clear out old entries
                conv_ij[p] = {}
                
                f = Spill.get_file(p, t, filetype='align')
                
                intersect = np.ones(f['x'].size, dtype=bool)
                for iphone in self.phones:
                    if iphone == p: continue
                    intersect &= f[iphone]

                thresh_pass = f['val'] > threshold
                
                x_new = (f['x'][intersect & thresh_pass] - self.profile.x_off).astype(int)
                y_new = (f['y'][intersect & thresh_pass] - self.profile.y_off).astype(int)
                                
                interior_new = interior[x_new, y_new]
                
                n_coinc[frozenset([p])] = x_new.size
                n_hits_interior[p] = interior_new.sum()
                n_hits_corr[p] = n_hits_interior[p] - noise[p]
                first_last[p] = (t in np.array(spl[p])[[0,-1]])
                                
                f.close()
                
                for subc in powerset(self.phones, min_size=2):
                    if not p in subc: continue
                    
                    # at the very least, we make the convolved matrix for p
                    pi, pj_all = self.optimal(subc)
                    
                    if not pi in conv_ij[p]:
                        self.convolve_sparse(x_new, y_new, (pi, p), shape=(self.profile.x_tot, self.profile.y_tot))
                     
                    if not all([pi in conv_ij[pij] for pij in subc]): continue
                        
                    sparse_i = conv_ij[pi][pi].copy()
                    
                    for pj in pj_all:
                        sparse_i = sparse_i.multiply(conv_ij[pj][pi])
                
                    # now we calculate the noise
                    noise_tot = 0
                    for part in __partition(subc):
                        
                        order = len(part)
                        if order == 1: continue # this is the signal term
                            
                        noise_contribution = self.profile.coeff(order)
                        
                        for s in map(frozenset, part):
                            if s == frozenset([pi]):
                                noise_contribution *= n_hits_interior[pi]
                            else:
                                noise_contribution *= n_coinc[s]
                            
                            if not pi in s:
                                noise_contribution *= self.get_effective_convolution(pi, s) ### conv_factors[(pi, s)]
                        
                        noise_tot += noise_contribution
                    
                    n_coinc[subc] = (sparse_i.sum() - noise_tot)
                
                if not overlap: continue
                
                all_spl += n_coinc[c]
                
                if np.any(list(first_last.values())):

                    if min(n_hits_corr.values()) <= 0 \
                    or min(n_hits_corr.values()) / max(n_hits_corr.values()) > overlap:
                        if np.all(list(first_last.values())):
                            # either we're missing a frame, or this is a statistical fluke
                            single_spl += overlap * max(n_hits_corr.values())
                        else:
                            single_spl += overlap * np.mean([n_hits_corr[p] for p in self.phones if not first_last[p]])
                    else:
                        single_spl += min(n_hits_corr.values())

                else:
                    single_spl += overlap * np.mean(list(n_hits_corr.values()))
                    
            n_all.append(all_spl)
            n_single.append(single_spl)
            
            
        if verbose: print("100% ", end="\r")
        
        # now find an unbiased ratio estimator (to first order) for each spill
        
        if visualize:
        
            plt.figure(figsize=(12, 4))
            plt.title('Efficiency by spill', tuple(map(lambda p: p[:6], self.phones)))
            plt.hist(eff_estimates[c], bins=35)
            plt.xlabel(r'$\epsilon$')

            plt.show()

        return np.array(n_all) / np.array(n_single)


    
def __partition(iterable):
    collection = list(iterable)
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in __partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def __superpartition(collection, n_sets):
    for idx in map(np.array, np.ndindex(*np.repeat(2**n_sets - 1, len(collection)))):
        include = (idx + 1) // 2**np.arange(n_sets).reshape(-1, 1) % 2 == 1
        if not np.all(include.sum(axis=1)): continue
        part = [list(np.array(collection)[include[i]]) for i in range(n_sets)]
        # make sure we only get each grouping once
        if part != sorted(part): continue
        yield part


def powerset(iterable, min_size=0, max_size=None):
    s = list(iterable)
    if not max_size: max_size = len(s)
    return map(frozenset, it.chain.from_iterable(it.combinations(s, r) for r in range(min_size, max_size+1)))




def plot_2_particles(eff_n, particle_frac, lim=((0,1),(0,1))):

    colz = ['r','g','b']

    A = np.linspace(0,1,100)
    X, Y = np.meshgrid(A,A)

    plt.figure(figsize=(7,7))

    plt.xlabel(r'$\epsilon_1$')
    plt.ylabel(r'$\epsilon_2$')

    plt.xlim(*(lim[0]))
    plt.ylim(*(lim[1]))

    contours = []
    labels = []

    for i, eff in enumerate(eff_n):

        def fn(x, y):
            X = np.moveaxis(np.array([x, y]), 0, 2)
            return np.dot(X**(i+2), particle_frac) / np.dot(X, particle_frac)

        contours.append(plt.contour(X, Y, fn(X, Y), [eff], colors=colz[i]))
        labels.append('{} phones'.format(i+2))

    plt.legend(map(lambda cont: cont.legend_elements()[0][0], contours), labels)

    plt.show()


def plot_3_particles(eff_n, particle_frac):
    
    colz=['r','g','b']

    fig = plt.figure(figsize=(10,10))
    ax = plt.gca(projection='3d')

    ax.set_xlabel(r'$\epsilon_1$')
    ax.set_ylabel(r'$\epsilon_2$')
    ax.set_zlabel(r'$\epsilon_3$')

    ax.set_xlim(0,1)
    ax.set_ylim(1,0)
    ax.set_zlim(0,1)
    
    u = np.linspace(0, np.pi / 2, 400)
    v = np.linspace(0, np.pi / 2, 400)
    
    x0 = np.outer(np.sin(v), np.cos(u))
    y0 = np.outer(np.sin(v), np.sin(u))
    z0 = np.outer(np.cos(v), np.ones(u.size))
    
    X = np.moveaxis(np.array([x0, y0, z0]), 0, 2)
    
    
    r = []
    
    for i,eff in enumerate(eff_n):
        
        r.append((eff * np.dot(X, particle_frac) / np.dot(X**(i+2), particle_rac))**(1/(i+1)))
        
    r = np.array(r)
    args = np.argmax(r, axis=0) 
    
    # slice on the max r values 
    idx = list(np.ogrid[[slice(r.shape[ax]) for ax in range(r.ndim) if ax != 0]])
    idx.insert(0, args)
    idx = tuple(idx)
    
    r = r[idx]
    c = np.array([np.full_like(x0, c, dtype=str) for c in colz])[idx]
    
    e1 = r * x0
    e2 = r * y0
    e3 = r * z0

    ax.plot_surface(e1, e2, e3, rstride=1, cstride=1, facecolors=c)
    

    # now do a 2D plot
    plt.figure(figsize=(8,8))
    
    X, Y = np.meshgrid(np.linspace(0, np.pi/2, args.shape[0]), 
                       np.linspace(0, np.pi/2, args.shape[1]))
    
    # create a custom colormap
    cm = LinearSegmentedColormap.from_list('rgb', [(1, 0, 0), (0, 1, 0), (0, 0, 1)], N=3)
    plt.imshow(args[::-1,::-1], cmap=cm, 
               origin='lower', extent=[0, np.pi/2, 0, np.pi/2])
    contour = plt.contour(X, Y, r[::-1, ::-1], cmap='gray')
    plt.clabel(contour, fontsize=8)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta$')

    plt.show()


def plot2eff(eff_n, particle_frac, guess_phi, guess_theta):

    def f(phitheta):
        phi, theta = phitheta
        
        x0 = np.sin(theta) * np.cos(phi)
        y0 = np.sin(theta) * np.sin(phi)
        z0 = np.cos(theta)
        
        X = np.array([x0, y0, z0])

        dot = [np.dot(X**(i+1), particle_frac) for i in range(4)]

        f2 = dot[0] / dot[1] * eff_n[0]
        f3 = dot[1] / dot[2] * eff_n[1] / eff_n[0]
        f4 = dot[2] / dot[3] * eff_n[2] / eff_n[1]
        
        return np.array([f3 - f2, f4 - f2])

    guess_compl = np.pi/2 - np.array(guess)

    phi, theta = scipy.optimize.fsolve(f, guess_compl)

    x0 = np.sin(theta) * np.cos(phi)
    y0 = np.sin(theta) * np.sin(phi)
    z0 = np.cos(theta)

    X = np.array([x0, y0, z0])

    r = np.dot(X, particle_frac) / np.dot(X**2, particle_frac) * eff_n[0]

    e1 = r * np.sin(theta) * np.cos(phi)
    e2 = r * np.sin(theta) * np.sin(phi)
    e3 = r * np.cos(theta)

    return np.pi/2-phi, np.pi/2-theta, e1, e2, e3

