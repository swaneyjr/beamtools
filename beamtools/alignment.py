import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

import os
import functools
import itertools as it

_NDIVS = 13

def optimize_scoring(root, spills, nmin=1, nmax=30, verbose=False):
    from timeit import timeit
    global _NDIVS

    _NDIVS = nmin

    def _test_spills(root_, spills_):
        spl = spills_[0] 
        align = Alignment(spl.res_x, spl.res_y)
        for p in spl.phones():
            if p is root_: continue
            score_spills(root_, p, align, spills_, nmax=3)

    dt_new = timeit(functools.partial(_test_spills, root, spills), number=1)
    if verbose: print('{}: {}'.format(_NDIVS, dt_new))

    while _NDIVS <= nmax:
        _NDIVS += 1
        dt_old = dt_new
        dt_new = timeit(functools.partial(_test_spills, root, spills), number=1)
        print('{}: {}'.format(_NDIVS, dt_new))

        if dt_new > dt_old:
            _NDIVS -= 1
            break

    return _NDIVS

def get_ndivs():
    global _NDIVS
    return _NDIVS

class Alignment():
    def __init__(self, res_x, res_y, x=0, y=0, phi=0, ux=0, uxy=0, uy=0):
        self.res_x = res_x
        self.res_y = res_y
        self.x = x
        self.y = y
        self.phi = phi
        self.ux = ux
        self.uxy = uxy
        self.uy = uy

    def adjust(self, x=None, y=None, phi=None, ux=None, uxy=None, uy=None):
        if x is None: x = self.x
        if y is None: y = self.y
        if phi is None: phi = self.phi
        if ux is None: ux = self.ux
        if uxy is None: uxy = self.uxy
        if uy is None: uy = self.uy
        
        return Alignment(self.res_x, self.res_y, x, y, phi, ux, uxy, uy)


    def to_array(self):
        return np.array([
            self.res_x, 
            self.res_y,
            self.x, 
            self.y, 
            self.phi, 
            self.ux, 
            self.uxy, 
            self.uy])

    def to_npz(self, fname):
        np.savez(fname, 
                res_x=self.res_x,
                res_y=self.res_y,
                x=self.x, 
                y=self.y, 
                phi=self.phi, 
                ux=self.ux, 
                uxy=self.uxy, 
                uy=self.uy)
    
    @staticmethod
    def from_array(arr):
        return Alignment(*arr)

    @staticmethod
    def from_npz(fname):
        f = np.load(fname)
        align = Alignment(
                f.f.res_x, 
                f.f.res_y,
                x=f.f.x, 
                y=f.f.y, 
                phi=f.f.phii, 
                ux=f.f.ux, 
                uxy=f.f.uxy, 
                uy=f.f.uy)
        f.close()
        return align

    # methods for converting between frames

    def sensor_map(self, xs, ys):
        xy_s = np.array([xs, ys])
        phi_mat = np.array([
                [np.cos(self.phi), -np.sin(self.phi)], 
                [np.sin(self.phi), np.cos(self.phi)]
            ])
    
        scale = np.eye(2) - np.array([[self.ux, self.uxy],[self.uxy, self.uy]])
    
        return functools.reduce(np.dot, [scale, phi_mat, xy_s]) + np.array([[self.x], [self.y]])

    def inverse_map(self, xlab, ylab):
        xy_lab = np.array([xlab, ylab]) - np.array([[self.x], [self.y]])
        inv_phi_mat = np.array([
                [np.cos(self.phi), np.sin(self.phi)],
                [-np.sin(self.phi), np.cos(self.phi)]
            ])
    
        inv_scale = np.linalg.inv(np.eye(2) - np.array([[self.ux, self.uxy],[self.uxy, self.uy]]))
    
        return functools.reduce(np.dot, [inv_phi_mat, inv_scale, xy_lab])

    
class AlignmentSet(dict):

    def __init__(self, alignment_dict, p_root=None):
        self.res_x = self.res_y = None

        for k,v in alignment_dict.items():
            if not self.res_x:
                self.res_x = v.res_x
                self.res_y = v.res_y
            elif v.res_x != self.res_x or v.res_y != self.res_y:
                raise ValueError("Alignments must have the same resolution")
            if not k == p_root:
                self[k] = v
        self.root = p_root

    def phones(self):
        keys = list(self.keys()) 
        if self.root: keys += [self.root]
        return keys

    @staticmethod
    def from_npz(fname):
        f = np.load(fname)
        aligns = {}
        root = None
        for key in f.files:
            if key == 'root':
                root = str(f[key])
            else:
                aligns[key] = Alignment.from_array(f[key])
        f.close()

        return AlignmentSet(aligns, root)

    def to_npz(self, fname):
        arrself = {key: self[key].to_array() for key in self}
        if self.root: arrself['root'] = self.root
        np.savez(fname, **arrself)

    def to_dict(self):
        d = {p: self[p] for p in self}
        if self.root:
            d[self.root] = Alignment(self.res_x, self.res_y)
        return d

    # shortcuts for align refinement applied to all
    def _apply_all(self, func, spills, *args, **kwargs):
        aligns_new = {p: func(self.root, p, align, spills, *args, **kwargs) \
                for p, align in self.items() if p != self.root}

        return AlignmentSet(aligns_new, self.root)
    
    def xy_grid(self, spills, *args, **kwargs):
        return self._apply_all(xy_grid, spills, *args, **kwargs)

    def phi_grid(self, spills, *args, **kwargs):
        return self._apply_all(phi_grid, spills, *args, **kwargs)

    def xyphi_grid(self, spills, *args, **kwargs):
        return self._apply_all(xyphi_grid, spills, *args, **kwargs)

    def xyphi_optimize(self, spills, *args, **kwargs):
        return self._apply_all(xyphi_optimize, spills, *args, **kwargs)

    def u_grid(self, spills, *args, **kwargs):
        return self._apply_all(u_grid, spills, *args, **kwargs)

    def u_optimize(self, spills, *args, **kwargs):
        return self._apply_all(u_optimize, spills, *args, **kwargs)

    def align_optimize(self, spills, *args, **kwargs):
        return self._apply_all(align_optimize, spills, *args, **kwargs)

    # output corrected coordinates
    def apply_aligns(self, spills, subdir_in='cluster', subdir_out='align'):

        os.makedirs(os.path.join(spills[0].basedir, subdir_out), exist_ok=True)
        for iphone in self.phones():
            
            for spl in spills:

                lim_x = spl.res_x / 2
                lim_y = spl.res_y / 2

                for t in spl[iphone]:

                    fspl = spl.get_file(iphone, t, subdir_in)

                    x_sensor = fspl['x'] + 0.5 - lim_x
                    y_sensor = fspl['y'] + 0.5 - lim_y
                    
                    if iphone == self.root:
                        x_lab = x_sensor
                        y_lab = y_sensor
                    else:
                        x_lab, y_lab = self[iphone].sensor_map(x_sensor, y_sensor)
                    
                    intersect_dict = {}
                    for jphone in self.keys():
                        if iphone == jphone: continue
                            
                        if jphone == self.root:
                            xj = x_lab
                            yj = y_lab
                        else:
                            xj, yj = self[jphone].inverse_map(x_lab, y_lab)   
                        
                        intersect_dict[jphone] = (np.abs(xj) < lim_x) \
                                & (np.abs(yj) < lim_y)
                        
                    outfile = '{}/{}/{}_p{}_t{}.npz'.format(
                            spl.basedir, 
                            subdir_out, 
                            spl.tag, 
                            iphone, 
                            t)

                    ialign = Alignment(spl.res_x, spl.res_y) \
                            if iphone == self.root else self[iphone]
                    other_keys = set(fspl.keys()) - {'x','y'}
                    other_fields = {key: fspl[key] for key in other_keys}

                    np.savez(outfile,
                            x=x_lab, 
                            y=y_lab,
                            align=ialign,
                            **intersect_dict,
                            **other_fields)
                    fspl.close()

    def subset(self, phones):
        align_dict = {}
        root = None
        for p in phones:
            if p == self.root:
                root = p
            else:
                align_dict[p] = self[p]

        return AlignmentSet(align_dict, root)

    def intersection(self, phones):
        return Intersection.from_alignments(self.subset(phones))


    def visualize(self):

        corner_x, corner_y = np.array([
            [-self.res_x, self.res_x, self.res_x, -self.res_x],
            [-self.res_y, -self.res_y, self.res_y, self.res_y]
            ]) / 2
            

        poly = [Polygon(al.sensor_map(corner_x, corner_y).transpose()) for al in self.values()] \
                + [Polygon(np.transpose([corner_x, corner_y]))]
        p = PatchCollection(poly, linewidths=1, alpha=0.5, cmap='jet')
        colors = np.linspace(0, 100, len(poly))
        p.set_array(colors)
        
        ax = plt.gca()
        ax.add_collection(p)
        plt.xlim(-self.res_x, self.res_x)
        plt.ylim(-self.res_y, self.res_y)
        plt.show()


def _chisq_offset(hist1, hist2, dx, dy):
    if dy > 0:
        hist1 = hist1[dy:,:]
        hist2 = hist2[:-dy, :]
    elif dy < 0:
        hist1 = hist1[:dy,:]
        hist2 = hist2[-dy:, :]
    
    if dx > 0:
        hist1 = hist1[:,dx:]
        hist2 = hist2[:,:-dx]
    elif dx < 0:
        hist1 = hist1[:, :dx]
        hist2 = hist2[:, -dx:]
        
    return np.sum((hist1/hist1.mean() - hist2/hist2.mean())**2/(hist1 + hist2)) / hist1.size


def coarse_align(spills, downsample=97, noise=None, visualize=False, p_root=None, filetype='cluster'):

    # first make histograms
    phones = spills[0].phones()
    res_x = spills[0].res_x
    res_y = spills[0].res_y
    shape_ds = (res_x//downsample, downsample, res_y//downsample, downsample)

    hists = {p:0 for p in phones}
    noise_grids = {p: noise[p].reshape(shape_ds).sum((1,3)).transpose() \
            if noise else 0 for p in phones}

    for spl in spills:
        for p in phones:
            hist_i = spl.histogram(p, downsample=downsample, filetype=filetype)
            hists[p] += hist_i - noise_grids[p]

    ysize = (len(phones) + 1) // 2
    if visualize:
        plt.figure(figsize=(8, 4*ysize))
        plt.tight_layout()
        plt.suptitle(r'Flux in particles / (pixel $\cdot$ s)')
        for i,p in enumerate(hists): 
            plt.subplot(ysize, 2, i+1)
            plt.title(p[:6])
            plt.imshow(hists[p] / downsample**2 / len(spills) / 4.2, cmap='plasma', origin='lower', extent=[0, res_x, 0, res_y]);
            plt.colorbar();
        plt.show()


    # use chi squared comparison as a measure of distance
    min_intersection = 5
    shape = list(hists.values())[0].shape
    sx = downsample*(shape[1] - 2*min_intersection)
    sy = downsample*(shape[0] - 2*min_intersection)

    dxy_all = {} 

    pairs = list(map(sorted, it.combinations(phones, 2)))
    ysize = (len(pairs) + 1) // 2

    if visualize:
        plt.figure(figsize=(8, 4*ysize))
        plt.suptitle('Coarse alignments')
        plt.tight_layout()

    for i, pair in enumerate(pairs):
        p1, p2 = pair
        grid = np.array([[_chisq_offset(hists[p1], hists[p2], x, y) \
                           for x in range(-shape[1]+min_intersection, shape[1]-min_intersection+1)] \
                           for y in range(-shape[0]+min_intersection, shape[0]-min_intersection+1)])

        iy, ix = np.unravel_index(np.argmin(grid), grid.shape)
        x = ix / (grid.shape[1]-1) * 2 * sx - sx
        y = iy / (grid.shape[0]-1) * 2 * sy - sy

        dxy_all[tuple(pair)] = (x, y)
 
        if visualize:
            plt.subplot(ysize, 2, i+1)
            plt.title(r'$\chi^2$: ({}, {})'.format(p1[:6], p2[:6]))
            plt.xlabel('dx (pix)')
            plt.ylabel('dy (pix)')
            plt.imshow(grid, cmap='Spectral', norm=LogNorm(), extent=[-sx, sx, -sy, sy], origin='lower');
            plt.colorbar();


    if not p_root:
        # choose a phone close to all the others
        furthest = {}

        for c,xy in dxy_all.items():
            dsq = xy[0]**2 + xy[1]**2
            for p in c:
                if not p in furthest or dsq > furthest[p]:
                    furthest[p] = dsq 
            
        p_root = min(furthest.keys(), key=(lambda key: furthest[key]))

    dxy_coarse = {}
    for c,xy in dxy_all.items():
        x, y = xy
        if p_root == c[0]:
            dxy_coarse[c[1]] = Alignment(res_x, res_y, x=x, y=y)
        elif p_root == c[1]:
            dxy_coarse[c[0]] = Alignment(res_x, res_y, x=-x, y=-y)

    if visualize:
        plt.show()

    return AlignmentSet(dxy_coarse, p_root)


# method for sorting points for easier computation
def _divide_points(x, y, ndivs, limits=None):

    if limits:
        xmin, xmax, ymin, ymax = limits
    else:
        xmin = np.amin(x)
        xmax = np.amax(x)
        ymin = np.amin(y)
        ymax = np.amin(y)

    divx = (xmax - xmin) / ndivs
    divy = (ymax - ymin) / ndivs

    return np.minimum((x - xmin) // divx, ndivs-1), np.minimum((y - ymin) // divy, ndivs-1)

# scoring functions
def score_points(x1, y1, x2, y2, alpha=0.5):

    global _NDIVS

    xmin = min(np.amin(x1), np.amin(x2))
    xmax = max(np.amax(x1), np.amax(x2))
    ymin = min(np.amin(y1), np.amin(y2))
    ymax = max(np.amax(y1), np.amax(y2))

    divx = (xmax - xmin) / _NDIVS
    divy = (ymax - ymin) / _NDIVS

    # make interlaced cells
    x1_cells, y1_cells = _divide_points(x1, y1, _NDIVS, limits=(xmin, xmax, ymin, ymax))
    x2_cells, y2_cells = _divide_points(x2, y2, _NDIVS+1, limits=(xmin - divx/2, xmax + divx/2, ymin-divy/2, ymax+divy/2))

    total_score = 0

    for i,j in np.ndindex(_NDIVS, _NDIVS):
        group1 = (x1_cells == i) & (y1_cells == j)
        if not group1.sum():
            continue

        group2 = ((x2_cells == i) | (x2_cells == i+1)) & ((y2_cells == j) | (y2_cells == j+1))
        if not group2.sum():
            continue

        rsquared_min = np.amin((x1[group1] - x2[group2].reshape(-1,1))**2 \
                               + (y1[group1] - y2[group2].reshape(-1,1))**2, axis=0)

        # this is just cdf for min Euclidean distance
        density = group2.sum() / (4 * divx * divy)
        survival = np.exp(-rsquared_min * np.pi * density)
        scores = np.maximum(survival - (1 - alpha), 0) / alpha
        total_score += np.sum(scores)

    return total_score / len(x1)


def score_spills(p1, p2, align, spills, filetype='cluster', nmax=None, **kwargs):
    scores = 0
    total = 0
    n = 0

    for spl in spills:
        # to save time in coarse adjustment
        if nmax and n >= nmax: break

        for times, overlap in spl.gen_overlaps((p1, p2)):

            # don't worry about small overlaps
            if overlap < 0.4: continue

            t1, t2 = times

            if nmax and n >= nmax: break

            n += 1

            f1 = spl.get_file(p1, t1, filetype)
            f2 = spl.get_file(p2, t2, filetype)

            lim_x = spl.res_x/2
            lim_y = spl.res_y/2

            x1_sensor1 = f1['x'] + 0.5 - lim_x
            y1_sensor1 = f1['y'] + 0.5 - lim_y
            x2_sensor2 = f2['x'] + 0.5 - lim_x
            y2_sensor2 = f2['y'] + 0.5 - lim_y

            x2_sensor1, y2_sensor1 = align.sensor_map(x2_sensor2, y2_sensor2)

            # limit to hits that map to opposite sensor as well
            x1_sensor2, y1_sensor2 = align.inverse_map(x1_sensor1, y1_sensor1)

            intersect1 = (np.abs(x1_sensor2) < lim_x) & (np.abs(y1_sensor2) < lim_y)
            intersect2 = (np.abs(x2_sensor1) < lim_x) & (np.abs(y2_sensor1) < lim_y)

            x1_sensor1 = x1_sensor1[intersect1]
            y1_sensor1 = y1_sensor1[intersect1]
            x2_sensor1 = x2_sensor1[intersect2]
            y2_sensor1 = y2_sensor1[intersect2]

            score = score_points(x1_sensor1, y1_sensor1, x2_sensor1, y2_sensor1, **kwargs)

            scores += overlap*score
            total += overlap

    return scores / total


# alignment optimizers
def xy_grid(p1, p2, align, spills, delta, res, visualize=False, **kwargs):
    x_bins, y_bins = np.meshgrid(np.linspace(align.x-delta, align.x+delta, res), np.linspace(align.y-delta, align.y+delta, res))
    score_grid = np.array([[score_spills(p1, p2, align.adjust(x=x, y=y), spills, **kwargs) \
                   for x in np.linspace(align.x-delta, align.x+delta, res)] \
                  for y in np.linspace(align.y-delta, align.y+delta, res)])

    if visualize:
        plt.figure()
        plt.imshow(score_grid, extent=[align.x-delta, align.x+delta, align.y-delta, align.y+delta], cmap='seismic', \
                   origin='lower');
        plt.title('({}, {})'.format(p1[:6], p2[:6]))
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.colorbar()
        plt.show()
    return score_grid, x_bins, y_bins


def phi_grid(p1, p2, align, spills, delta, res, visualize=False, **kwargs):
    phi_bins = np.linspace(align.phi-delta, align.phi+delta, res)
    score_grid = np.array([score_spills(p1, p2, align.adjust(phi=phi_i), spills, **kwargs) for phi_i in phi_bins])

    if visualize:
        plt.figure()
        plt.plot(phi_bins, score_grid)
        plt.title('({}, {})'.format(p1[:6], p2[:6]))
        plt.xlabel(r'$\Delta\phi$')
        plt.ylabel('Score')
        plt.show()

    return score_grid, phi_bins


def _get_max(grid, *bins):
    idx = np.unravel_index(np.argmax(grid), grid.shape)
    return tuple(b[idx] for b in bins)


def xyphi_grid(p1, p2, align, spills, delta_xy, delta_phi, \
                res_xy=11, res_phi=15, visualize=None, **kwargs):

    if visualize and visualize != 'best' and visualize != 'all':
        raise ValueError('"visualize" must be one of "best"/"all"')

    dx_result = align.x
    dy_result = align.y
    phi_result = align.phi
    score_grid_best = None

    max_score = 0
    for phi_i in np.linspace(align.phi-delta_phi, align.phi+delta_phi, res_phi):
        if visualize=='all':
            print("phi={:.4f}".format(phi_i))

        score_grid, x_bins, y_bins = xy_grid(p1, p2, align.adjust(phi=phi_i), spills, \
                delta_xy, res_xy, visualize=(visualize=='all'), **kwargs)

        grid_max = np.amax(score_grid)
        if grid_max > max_score:
            max_score = grid_max

            dx_result, dy_result = _get_max(score_grid, x_bins, y_bins)

            phi_result = phi_i

            score_grid_best = score_grid

    if visualize == 'best':
        plt.figure()
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.imshow(score_grid_best, extent=[align.x-delta_xy, align.x+delta_xy, align.y-delta_xy, align.y+delta_xy], \
                   cmap='seismic', origin='lower')
        plt.colorbar()
        plt.title(r'$\phi = {:.4}, ({}, {})$'.format(phi_result, p1[:6], p2[:6]))
        plt.show()

    return align.adjust(x=dx_result, y=dy_result, phi=phi_result)


def u_grid(p1, p2, align, spills, delta, res, visualize=None, **kwargs):

    if visualize and visualize != 'best' and visualize != 'all':
        raise ValueError('"visualize" must be one of "best"/"all"')

    ux_result = align.ux
    uxy_result = align.uxy
    uy_result = align.uy
    score_grid_best = None

    max_score = 0
    for uxy_i in np.linspace(align.uxy-delta, align.uxy+delta, res):

        score_grid = np.array([[score_spills(p1, p2, align.adjust(ux=ux_i, uxy=uxy_i, uy=uy_i), spills, **kwargs) \
                   for ux_i in np.linspace(align.ux-delta, align.ux+delta, res)] \
                  for uy_i in np.linspace(align.uy-delta, align.uy+delta, res)])

        if visualize=='all':
            plt.figure()
            plt.imshow(score_grid, extent=[align.ux-delta, align.ux+delta, align.uy-delta, align.uy+delta], \
                       cmap='seismic', origin='lower')
            plt.colorbar()
            plt.title(r'$u_{xy} =$' + '{:.4}'.format(uxy_i))
            plt.show()

        grid_max = np.amax(score_grid)
        if grid_max > max_score:
            max_score = grid_max

            ux_bins, uy_bins = np.meshgrid(np.linspace(align.ux-delta, align.ux+delta, res), np.linspace(align.uy-delta, align.uy+delta, res))
            ux_result, uy_result = _get_max(score_grid, ux_bins, uy_bins)

            uxy_result = uxy_i

            score_grid_best = score_grid

    if visualize == 'best':
        plt.figure()
        plt.imshow(score_grid_best, extent=[align.ux-delta, align.ux+delta, align.uy-delta, align.uy+delta], \
                   cmap='seismic', origin='lower')
        plt.colorbar()
        plt.title(r'$u_{xy} =$' + '{:.4}'.format(uxy_result))
        plt.xlabel(r'$u_x$')
        plt.ylabel(r'$u_y$')
        plt.show()

    return align.adjust(ux=ux_result, uxy=uxy_result, uy=uy_result)


def xyphi_optimize(p1, p2, align, spills, visualize=False, alpha=0.1, nmax=10, **kwargs):
    def f_xyphi(x, p1, p2, spills):
        return 1 - score_spills(p1, p2, align.adjust(x=x[0], y=x[1], phi=x[2]), 
                spills, alpha=alpha, nmax=nmax, **kwargs)
    
    guess = align.to_array()[2:5]
    progress = [guess]
    callback = lambda xk: progress.append(xk) if visualize else None
    res = minimize(f_xyphi, guess, args=(p1, p2, spills), method='Nelder-Mead', callback=callback)
    
    if visualize:
        progress_arr = np.array(progress).transpose()
        
        xmin = np.amin(progress_arr[0])
        xmax = np.amax(progress_arr[0])
        ymin = np.amin(progress_arr[1])
        ymax = np.amax(progress_arr[1])
        
        plt.xlim(xmin-(xmax-xmin)/6, xmax+(xmax-xmin)/6)
        plt.ylim(ymin-(ymax-ymin)/6, ymax+(ymax-ymin)/6)
        
        plt.title(r'$(x, y, \phi)$ Convergence: ({}, {})'.format(p1[:6], p2[:6]))
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.scatter(progress_arr[0], progress_arr[1], c=progress_arr[2], \
                    s=160*np.arange(progress_arr.shape[1]+1)[::-1] / progress_arr.shape[1], cmap='winter', marker='x')
        plt.colorbar()
        plt.show()
    
    if res.success:
        x, y, phi = res.x
        return align.adjust(x=x, y=y, phi=phi)
    


def u_optimize(p1, p2, align, spills, visualize=False, nmax=10, alpha=0.05, **kwargs):
    
    def f_u(x, p1, p2, spills):
        return 1 - score_spills(p1, p2, align.adjust(ux=x[0],uxy=x[1],uy=x[2]), 
                            spills, alpha=alpha, nmax=nmax, **kwargs)
 
    guess = align.to_array()[5:]
    progress = [guess]
    callback = lambda xk: progress.append(xk) if visualize else None
    res = minimize(f_u, guess, args=(p1, p2, spills), method='Nelder-Mead', \
            options={'xatol': 1e-5}, callback=callback)
    
    if visualize:
        progress_arr = np.array(progress).transpose()
        
        xmin = np.amin(progress_arr[0])
        xmax = np.amax(progress_arr[0])
        ymin = np.amin(progress_arr[2])
        ymax = np.amax(progress_arr[2])
        
        plt.xlim(xmin-(xmax-xmin)/6, xmax+(xmax-xmin)/6)
        plt.ylim(ymin-(ymax-ymin)/6, ymax+(ymax-ymin)/6)
        
        plt.title(r'$(u_x, u_y, u_{xy})$ Convergence: ' + '({}, {})'.format(p1[:6], p2[:6]))
        plt.xlabel(r'$u_x$')
        plt.ylabel(r'$u_y$')
        plt.scatter(progress_arr[0], progress_arr[2], c=progress_arr[1], \
                    s=160*np.arange(progress_arr.shape[1]+1)[::-1] / progress_arr.shape[1], cmap='winter', marker='x')
        plt.colorbar()
        plt.show()
        
    if res.success:
        ux, uxy, uy = res.x
        return align.adjust(ux=ux, uxy=uxy, uy=uy)


def align_optimize(p1, p2, align, spills, verbose=False, nmax=25, alpha=0.05, **kwargs):
    def f_tot(x, p1, p2, spills):
        return 1 - score_spills(
                p1, p2, 
                Alignment(align.res_x, align.res_y, *x), 
                spills, 
                alpha=alpha, 
                nmax=nmax, 
                **kwargs)

    callback = (lambda xk: print(xk)) if verbose else None

    guess = align.to_array()[2:]
    res = minimize(f_tot, guess, args=(p1, p2, spills), method='Nelder-Mead', \
            options={'xatol': 1e-5}, callback=callback)
        
    if res.success:
        if verbose:
            print("Success!")
        return Alignment(align.res_x, align.res_y, *res.x)
    elif verbose: 
        print("Failed:", res.message)

