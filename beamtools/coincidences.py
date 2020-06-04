import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import itertools as it
import os
import copy

from .spills import SpillSet
from .alignment import _divide_points

class CoincidenceCounter():
    def __init__(self, spills, profile):
        self._spills = spills
        self._profile = profile

        self._windows = {}
        self._optimal = {}

        self._factors = {}


    def hist(self, xy_bins=None, survival_bins=100, filetype='align', ndivs=4):
                
        phones = set()
        hists = {}

        for c in it.combinations(self._spills.phones, 2):
            
            if xy_bins:
                xr = yr = xy_bins // 2
            elif not frozenset(c) in self._windows:
                continue
            else:    
                window = np.array(self._windows[frozenset(c)], dtype=int)
                xr, yr = window // 2
            
            phones.update(set(c))

            hist_interior = self._profile.cut_edge(2*max([xr, yr]))
            
            p1, p2 = sorted(c)
            xy_hist = np.zeros((2*yr+1, 2*xr+1))
            survival_hist = np.zeros(survival_bins) 

            for spl in self._spills:
                if not all(p in spl.phones for p in c): continue
                for times, overlap in spl.gen_overlaps((p1, p2)):
                    t1, t2 = times

                    f1 = spl.get_file(p1, t1, filetype=filetype)
                    f2 = spl.get_file(p2, t2, filetype=filetype)

                    intersect1 = np.ones(f1['x'].size, dtype=bool)
                    intersect2 = np.ones(f2['x'].size, dtype=bool)
                    for iphone in self._profile.phones:
                        if iphone != p1:
                            intersect1 &= f1[iphone]
                        if iphone != p2:
                            intersect2 &= f2[iphone]

                    x1 = f1['x'][intersect1] - self._profile.x_off
                    y1 = f1['y'][intersect1] - self._profile.y_off
                    x2 = f2['x'][intersect2] - self._profile.x_off
                    y2 = f2['y'][intersect2] - self._profile.y_off

                    f1.close()
                    f2.close()

                    # now cut out the edges on x1
                    edge_cut = hist_interior[x1.astype(int), y1.astype(int)]

                    x1 = x1[edge_cut]
                    y1 = y1[edge_cut]

                    xmin = 0
                    xmax = self._profile.x_tot
                    ymin = 0
                    ymax = self._profile.y_tot

                    divx = (xmax - xmin) / ndivs
                    divy = (ymax - ymin) / ndivs

                    # make interlaced cells
                    x1_cells, y1_cells = _divide_points(x1, y1, ndivs, limits=(xmin, xmax, ymin, ymax))
                    x2_cells, y2_cells = _divide_points(x2, y2, ndivs+1, limits=(xmin - divx/2, xmax + divx/2, ymin-divy/2, ymax+divy/2))

                    for i,j in np.ndindex(ndivs, ndivs):
                        group1 = (x1_cells == i) & (y1_cells == j)
                        if not group1.sum():
                            continue

                        group2 = ((x2_cells == i) | (x2_cells == i+1)) & ((y2_cells == j) | (y2_cells == j+1))
                        if not group2.sum():
                            continue

                        rsquared_min = np.amin((x1[group1] - x2[group2].reshape(-1,1))**2 \
                                               + (y1[group1] - y2[group2].reshape(-1,1))**2, axis=0)

                        nx = 2
                        ny = 2
                        if i == 0:
                            nx -= 0.5
                        if i == ndivs-1:
                            nx -= 0.5
                        if j == 0:
                            ny -= 0.5
                        if j == ndivs-1:
                            ny -= 0.5

                        density = group2.sum() / (nx * divx * ny * divy)
                        survival = np.exp(-rsquared_min * np.pi * density)

                        survival_hist += np.histogram(survival, bins=survival_bins)[0]

                        xy_vals = np.dstack([x1[group1] - x2[group2].reshape(-1,1), 
                                             y1[group1] - y2[group2].reshape(-1,1)]).reshape(-1, 2).transpose()

                        xy_hist += np.histogram2d(xy_vals[0], xy_vals[1], \
                                        bins=(np.arange(-yr-0.5, yr+1.5), 
                                            np.arange(-xr-0.5, xr+1.5)))[0]

                    xy_hist -= x1.size * x2.size * self._profile.coeff(2)

            # compute factors if we are using these windows
            hists[frozenset(c)] = xy_hist

            plt.figure(figsize=(9, 4))
            plt.subplot(121)
            plt.title('Min distances: ({}, {})'.format(p1[:6], p2[:6]))
            plt.xlabel('1-cdf')
            plt.hist(np.linspace(0, 1, survival_hist.size), weights=survival_hist, bins=survival_hist.size)

            plt.subplot(122)
            plt.title('Hit separation: ({}, {})'.format(p1[:6], p2[:6]))
            plt.xlabel(r'$\Delta x$ (pixels)')
            plt.ylabel(r'$\Delta y$ (pixels)')

            plt.imshow(xy_hist, extent=[-xr-0.5, xr+0.5, -yr-0.5, yr+0.5], norm=LogNorm())
            plt.colorbar()
            plt.show()

        # now compute noise factors
        if xy_bins: return
        
        for pi in phones:
            p_other = phones.copy()
            p_other.remove(pi)
            for pj_all in map(list, powerset(p_other, min_size=1)):
                j_conv = np.array([self._windows[frozenset([pi, pj])] for pj in pj_all])
                min_conv_idx = np.argmin(np.product(j_conv, axis=1))
                min_conv = j_conv[min_conv_idx]
                                        
                conv_tot = 0
                
                # iterate over positions for particle to be incident on min_conv sensor
                for ixy in map(lambda a: np.array(a) - (min_conv // 2), np.ndindex(tuple(min_conv))):
                    conv_contribs = [1]
                    
                    # now multiply by probabilities for particle to be in convolution windows on other sensors
                    for j_idx, pj in enumerate(pj_all):
                        if j_idx == min_conv_idx: continue
                        
                        xy_idx = frozenset([pj_all[min_conv_idx], pj])
                        xy_hist = hists[xy_idx]
                        xy_size = np.array(self._windows[xy_idx]) 
                        
                        xstart, ystart = np.array(xy_hist.shape) // 2 - ixy \
                                - j_conv[j_idx] // 2
                        xend = xstart + j_conv[j_idx][0]
                        yend = ystart + j_conv[j_idx][1]
                        xstart = max(xstart, 0)
                        ystart = max(ystart, 0)
                        
                        hist_cut = (xy_hist[xstart:xend, ystart:yend]).sum() / xy_hist.sum()

                        conv_contribs.append(hist_cut)
                                
                    # mean of upper (perfect correlation) and lower (no correlation) bounds
                    conv_tot += (np.product(conv_contribs) + min(conv_contribs)) / 2
                
                self._factors[(pi, frozenset(pj_all))] = conv_tot
                #min_convolutions[(pi, frozenset(s))] = np.product(min_conv)
                    

    
    def _find_optimal(self, c):

        c = frozenset(c)
        p_opt = self._optimal[frozenset(c)]
        return p_opt, c - set([p_opt])
 

    def set_windows(self, windows):
        
        # convert to set type
        windows = {frozenset(k): v for k,v in windows.items()}

        # first make sure we have valid keys
        phones = set()
        for p1, p2 in windows:
            phones.update({p1, p2})

        # make sure all of the keys are present
        for c in it.combinations(phones, 2):
            if not frozenset(c) in windows:
                raise ValueError('Missing key {}'.format(c))

        self._windows = windows

        # determine the fastest way to convolve for all subsets
        for subset in powerset(phones, min_size=2):
            pi_best = None
            for pi in subset:
                total_conv = 0
                for pj in subset:
                    if pi is pj: continue

                    total_conv += np.product(windows[frozenset([pi, pj])])
                
                if pi_best == None or total_conv < min_conv:
                    min_conv = total_conv
                    pi_best = pi
                
            self._optimal[subset] = pi_best


    def count_coincidences(self, filetype='align', verbose=False):

        def _process_counts():
            # calculate the run adjustment
            counts_all = np.array(counts_neg[::-1] + counts_pos)
            noise_all = np.array(noise_neg[::-1] + noise_pos)
            net_all = counts_all - noise_all 

            for idx, net in enumerate(net_all):
                t = (idx - len(counts_neg) + 0.5) * 1000/spl.fps + offset
                print(t, net)
                                
            argmax = np.argmax(net_all)
            countmax = counts_all[argmax]
            noisemax = noise_all[argmax]
            netmax = net_all[argmax]
            tmax = (argmax - len(counts_neg) + 0.5) * 1000/spl.fps + offset

            if net_all[argmax-1] > net_all[argmax+1]:
                argsecond = argmax-1
            else:
                argsecond = argmax+1

            countsecond = counts_all[argsecond]
            noisesecond = noise_all[argsecond]
            netsecond = net_all[argsecond]
            tsecond = (argsecond - len(counts_neg) + 0.5) * 1000/spl.fps + offset
                                
            center_new = (tmax*netmax + tsecond*netsecond) \
                    / (netmax + netsecond)
            var_new = (1000/spl.fps)**2 * netmax**2 * netsecond**2 \
                    * ((countmax + noisemax)/netmax**2 \
                    + (countsecond + noisesecond)/netsecond**2) \
                    * (netmax + netsecond)**-4 

            print('{} +/- {}'.format(center_new, np.sqrt(var_new)))
            print()

            run_centers.append(center_new)
            run_weights.append(1 / var_new) 

            # clear lists
            counts_pos.clear()
            counts_neg.clear()

            noise_pos.clear()
            noise_neg.clear()
 


        def _process_runs():
            
            # first handle remaining data
            _process_counts()

            key = frozenset([(p1, run1), (p2, run2)])
            
            # take a weighted average of individual spill results
            center = np.average(run_centers, weights=run_weights)
            weight = np.sum(run_weights)

            distances[key] = center
            weights[key] = weight

            if verbose: 
                print('Adjustment: {} +/- {}'.format(center, np.sqrt(1/weight)))
                print()


        distances = {}
        weights = {} # inverse variance
        runs_all = set()
        offset = None

        # doubly open-ended lists
        counts_pos = []
        counts_neg = []

        noise_pos = []
        noise_neg = []

        for c in it.combinations(self._spills.phones, 2):
            p1, p2 = sorted(c)
            if not frozenset([p1, p2]) in self._windows: continue
            if verbose: print(p1, p2) 

            run1 = 0
            run2 = 0

            run_centers = []
            run_weights = [] 

            for spl in self._spills:
                if not p1 in spl.phones or not p2 in spl.phones \
                        or max(len(spl[p]) for p in (p1,p2)) < 2:
                    continue

                # keep counts per intersection of runs
                if spl.run[p1] != run1 or spl.run[p2] != run2:
                    
                    if run1 and run2:
                        _process_runs()

                    run1 = spl.run[p1]
                    run2 = spl.run[p2]
                    
                    if verbose:
                        print('Runs:')
                        print('{}: {}'.format(p1[:6], run1))
                        print('{}: {}'.format(p2[:6], run2))

                    runs_all.add((p1, run1))
                    runs_all.add((p2, run2))

                    run_centers = []
                    run_weights = []
                    offset = None

                # group according to time differences
                diffs = np.diff(list(it.product(spl[p2], spl[p1])), axis=1)
                diffmin = min(diffs[diffs > 0]) if np.any(diffs > 0) else diffs.max()
                while diffmin < 0: diffmin += 1000 / spl.fps

                # check whether to readjust for slight FPS differences
                offset_new = diffmin - 500/spl.fps
                
                if offset is None:
                    # first iteration of run
                    offset = offset_new
                elif abs(offset - offset_new) > 10:
                    _process_counts()
                    
                    offset = offset_new

                for t1, t2 in it.product(spl[p1], spl[p2]):

                    f1 = spl.get_file(p1, t1, filetype=filetype)
                    f2 = spl.get_file(p2, t2, filetype=filetype)
                    
                    intersect1 = np.ones(f1['x'].size, dtype=bool)
                    intersect2 = np.ones(f2['x'].size, dtype=bool)
                    for iphone in self._profile.phones:
                        if iphone != p1:
                            intersect1 &= f1[iphone]
                        if iphone != p2:
                            intersect2 &= f2[iphone]

                    x1 = (f1['x'][intersect1] - self._profile.x_off).astype(int)
                    y1 = (f1['y'][intersect1] - self._profile.y_off).astype(int)
                    x2 = (f2['x'][intersect2] - self._profile.x_off).astype(int)
                    y2 = (f2['y'][intersect2] - self._profile.y_off).astype(int)

                    f1.close()
                    f2.close()

                    sparse1 = csr_matrix((np.ones(x1.size), (x1, y1)),
                            shape=(self._profile.x_tot, self._profile.y_tot))
                    sparse2 = csr_matrix((np.ones(x2.size), (x2, y2)),
                            shape=(self._profile.x_tot, self._profile.y_tot))   

                    window = self._windows[frozenset(c)]
                    count = 0

                    for dx, dy in np.ndindex(window):
                            
                        dx -= window[0]//2
                        dy -= window[1]//2
                       
                        sparse_i = sparse1
                        sparse_j = sparse2

                        if dx > 0:
                            sparse_i = sparse_i[dx:, :]
                            sparse_j = sparse_j[:-dx, :]
                        elif dx < 0:
                            sparse_i = sparse_i[:dx, :]
                            sparse_j = sparse_j[-dx:, :]
                        if dy > 0:
                            sparse_i = sparse_i[:, dy:]
                            sparse_j = sparse_j[:, :-dy]
                        elif dy < 0:
                            sparse_i = sparse_i[:, :dy]
                            sparse_j = sparse_j[:, -dy:]

                        count += sparse_i.multiply(sparse_j).sum()

                    t = t1 - t2 - offset
                    idx = abs(t) * spl.fps / 1000

                    count_list = counts_pos if t > 0 else counts_neg
                    noise_list = noise_pos if t > 0 else noise_neg
                    while idx >= len(count_list):
                        count_list.append(0)
                        noise_list.append(0)
                    count_list[int(idx)] += count
                    noise_list[int(idx)] += self._profile.coeff(2) * x1.size * x2.size * window[0] * window[1] 

            # process final pair
            _process_runs()

        # now compute best fit drift values
        runs_list = sorted(runs_all)
        n_runs = len(runs_list)

        A = np.zeros((n_runs, n_runs))
        total_weights = np.zeros(n_runs)
        total_distances = np.zeros(n_runs)

        for runs in weights:

            run_i, run_j = sorted(runs)
            
            # find coordinates of runs in matrix    
            for r, run in enumerate(runs_list):
                if run_i == run:
                    i = r
                    break
            
            for r, run in enumerate(runs_list[i+1:]):
                if run_j == run:
                    j = r + i + 1
                    break
                
            # construct matrix elements for linear system
            A[i,j] = A[j,i] = -weights[runs]
            total_weights[i] += weights[runs]
            total_weights[j] += weights[runs]
            total_distances[i] += distances[runs] * weights[runs]
            total_distances[j] -= distances[runs] * weights[runs]

        A += np.diagflat(total_weights)
        b = total_distances

        # this is a singular matrix, but we can replace the first row as
        # x1 + ... xN = 0
        A[0] = np.ones(n_runs)
        b[0] = 0

        A_inv = np.linalg.inv(A)
        t_estimates = np.matmul(A_inv, b)

        # now make corrections
        corrections = {r: t for r,t in zip(runs_list, t_estimates)}
        spills_new = copy.deepcopy(self._spills)
        for spl in spills_new:
            for p in spl.phones:
                r = spl.run[p]
                if not (p, r) in corrections: continue
                correction = int(corrections[(p, r)] + 0.5)
                spl._t[p] = list(np.add(spl._t[p], -correction))
                spl._drift[p] += correction

        return spills_new, corrections

            
    def save_coincidences(self, subdir_in='align', subdir_out='coinc', verbose=False):

        for n_spills, spl in enumerate(self._spills):
            if verbose:
                print("{:.2f}%".format(100*n_spills/len(self._spills)), end="\r")

            os.makedirs(os.path.join(
                spl.basedir,
                spl.tag,
                'coincidences',
                subdir_out), exist_ok=True)

            # keep a cache of results
            sparse_hits = {}
            sparse_val = {}
            sparse_npix = {}

            t_nominal = {}
            coincidences = {}

            for t, p, overlap in spl.gen_overlaps_single(spl.phones):
                #print(p, t)

                f = spl.get_file(p, t, filetype=subdir_in)
                
                intersect = np.ones(f['x'].size, dtype=bool)
                for iphone in self._profile.phones:
                    if iphone == p: continue
                    intersect &= f[iphone]
                
                x_new = (f['x'][intersect] - self._profile.x_off).astype(int)
                y_new = (f['y'][intersect] - self._profile.y_off).astype(int)
                val_new = f['max_val'][intersect]
                npix_new = f['n_pix'][intersect]

                t_nominal[p] = t + spl._drift[p]

                # now construct sparse matrices to find coincidences
                sparse_shape = (self._profile.x_tot, self._profile.y_tot)

                # boolean matrix for efficient multiplication
                sparse_hits[p] = csr_matrix((np.ones(x_new.size, dtype=bool), \
                        (x_new, y_new)),
                        shape=sparse_shape)
                
                sparse_val[p] = csr_matrix((val_new, (x_new, y_new)), shape=sparse_shape)
                sparse_npix[p] = csr_matrix((npix_new, (x_new, y_new)), shape=sparse_shape)
                

                f.close()

                # check for hits across all pairs of phones
                
                for pi in self._profile.phones:

                    if p == pi or not pi in sparse_hits: continue
                    
                    if not frozenset([pi, p]) in self._windows: continue
                    cx, cy = self._windows[frozenset([pi, p])]

                    xi = []
                    xj = []
                    yi = []
                    yj = []
                    val_i = []
                    val_j = []
                    npix_i = []
                    npix_j = []
                    
                    for dx, dy in np.ndindex((cx, cy)):
                        
                        dx -= cx//2
                        dy -= cy//2

                        sparse_i = sparse_hits[pi]
                        sparse_j = sparse_hits[p]

                        if dx > 0:
                            sparse_i = sparse_i[dx:, :]
                            sparse_j = sparse_j[:-dx, :]
                        elif dx < 0:
                            sparse_i = sparse_i[:dx, :]
                            sparse_j = sparse_j[-dx:, :]
                        if dy > 0:
                            sparse_i = sparse_i[:, dy:]
                            sparse_j = sparse_j[:, :-dy]
                        elif dy < 0:
                            sparse_i = sparse_i[:, :dy]
                            sparse_j = sparse_j[:, -dy:]

                        x_hits, y_hits = sparse_i.multiply(sparse_j).nonzero()
                        if not x_hits.size: continue 

                        if dx > 0:
                            x_hits_i = x_hits + dx
                            x_hits_j = x_hits
                        else:
                            x_hits_i = x_hits
                            x_hits_j = x_hits - dx
                        if dy > 0:
                            y_hits_i = y_hits + dy
                            y_hits_j = y_hits
                        else:
                            y_hits_i = y_hits
                            y_hits_j = y_hits - dy

                        xi.append(x_hits_i)
                        xj.append(x_hits_j)

                        yi.append(y_hits_i)
                        yj.append(y_hits_j)
                        
                        val_i.append(np.array(sparse_val[pi][x_hits_i, y_hits_i]).flatten())
                        val_j.append(np.array(sparse_val[p][x_hits_j, y_hits_j]).flatten())
                        
                        npix_i.append(np.array(sparse_npix[pi][x_hits_i, y_hits_i]).flatten())
                        npix_j.append(np.array(sparse_npix[p][x_hits_j, y_hits_j]).flatten())

                    if xi:
                        xi = np.hstack(xi)
                        yi = np.hstack(yi)
                        val_i = np.hstack(val_i)
                        npix_i = np.hstack(npix_i)
                    else:
                        xi = np.array([])
                        yi = np.array([])
                        val_i = np.array([])
                        npix_i = np.array([])

                    if xj:
                        xj = np.hstack(xj)
                        yj = np.hstack(yj)
                        val_j = np.hstack(val_j)
                        npix_j = np.hstack(npix_j)
                    else:
                        xj = np.array([])
                        yj = np.array([])
                        val_j = np.array([])
                        npix_j = np.array([])


                    t_coinc = {k: t_nominal[k] for k in (p, pi)}

                    x_coinc = {pi: xi, p: xj}
                    y_coinc = {pi: yi, p: yj}
                    val_coinc = {pi: val_i, p: val_j}
                    npix_coinc = {pi: npix_i, p: npix_j}
                    w_coinc = {pi: (1,1), p: (cx, cy)}

                    coinc = Coincidence(t_coinc, 
                            x_coinc, 
                            y_coinc, 
                            val_coinc, 
                            npix_coinc,
                            w_coinc)
                    
                    outname = '_'.join(sorted(['{}_{}'.format(k,v) for k,v in t_coinc.items()]))
                    outfile = os.path.join(
                            spl.basedir,
                            spl.tag,
                            'coincidences',
                            subdir_out,
                            outname + '.npz')

                    coinc.to_npz(outfile)
                    #print(t_coinc, np.hstack(xi).size)
                    coincidences[frozenset([p,pi])] = coinc
                
                # now that we have pairwise hits, we can construct longer tracks
                for c in powerset(self._profile.phones, min_size=3):
                    if not p in c or not all(pi in sparse_hits for pi in c) \
                            or not all(frozenset(s) in self._windows for s in it.combinations(c, 2)): 
                        continue

                    pi, pj_all = self._find_optimal(c)
                    if not pi: continue

                    #print(p, c)
                    pj_all = list(pj_all)

                    idxi = None
                    val_i = None
                    npix_i = None
                    
                    xyvn_arrays = {}

                    for pj in pj_all:
                        cj = coincidences[frozenset([pi, pj])]
                        idxi_j_all = cj.x[pi] + cj.y[pi] * spl.res_x
                        idxi_j = np.unique(idxi_j_all)

                        idx_arg = [np.argwhere(idx == idxi_j_all).flatten() for idx in idxi_j]

                        xyvn_j = [np.vstack([cj.x[pj][arg], 
                                cj.y[pj][arg], 
                                cj.val[pj][arg], 
                                cj.npix[pj][arg]]).transpose() for arg in idx_arg]
                        xyvn_j = np.array(xyvn_j)
                        #print(idxi_j % 4656, idxi_j // 4656, xyvn_j)

                        if idxi is None:
                            idxi = idxi_j
                            val_i = cj.val[pi]
                            npix_i = cj.npix[pi]
                            xyvn_arrays[pj] = xyvn_j
                        else:
                            idxi, in_i, in_j = np.intersect1d(idxi, idxi_j, return_indices=True)
                            
                            val_i = val_i[in_i]
                            npix_i = npix_i[in_i]

                            for pk in xyvn_arrays:
                                xyvn_arrays[pk] = xyvn_arrays[pk][in_i]
                                
                            xyvn_arrays[pj] = xyvn_j[in_j]

                    
                    # now convert to standard arrays 
                    if not idxi.size: 
                        x_coinc = {k: [] for k in c}
                        y_coinc = {k: [] for k in c}
                        val_coinc = {k: [] for k in c}
                        npix_coinc = {k: [] for k in c}
                    else:

                        xyvn_flat = []
                        for i,idx in enumerate(idxi):
                            xi = idx % spl.res_x
                            yi = idx // spl.res_x
                            xyvn = [np.array([[xi, yi, val_i[i], npix_i[i]]])]

                            for pj in pj_all:
                                xyvn_j = xyvn_arrays[pj] 
                                xyvn.append(xyvn_j[i])
                            
                            #print(xyvn)
                            xyvn_flat.append(list(map(np.hstack, it.product(*xyvn))))

                        xyvn_flat = np.vstack(xyvn_flat).transpose()
                    
                        x_coinc = {pi: xyvn_flat[0]}
                        y_coinc = {pi: xyvn_flat[1]}
                        val_coinc = {pi: xyvn_flat[2]}
                        npix_coinc = {pi: xyvn_flat[3]}
                        for j, pj in enumerate(pj_all):
                            x_coinc[pj] = xyvn_flat[4*j + 4]
                            y_coinc[pj] = xyvn_flat[4*j + 5]
                            val_coinc[pj] = xyvn_flat[4*j + 6]
                            npix_coinc[pj] = xyvn_flat[4*j + 7]

                    t_coinc = {pk: t_nominal[pk] for pk in c} 
                    w_coinc = {pj: self._windows[frozenset([pi, pj])] for pj in pj_all}
                    w_coinc[pi] = (1,1)

                    coinc = Coincidence(t_coinc,
                            x_coinc,
                            y_coinc,
                            val_coinc,
                            npix_coinc,
                            w_coinc)

                    outname = '_'.join(sorted(['{}_{}'.format(k,v) for k,v in t_coinc.items()]))
                    outfile = os.path.join(
                            spl.basedir,
                            spl.tag,
                            'coincidences',
                            subdir_out,
                            outname)

                    #print(t_coinc, idxi.size, xyvn_flat[0].size if idxi.size else 0)
                    coinc.to_npz(outfile)
                                
        if verbose: print("100% ", end="\r")


    def get_efficiency(self, single_dir='align', coinc_dir='coinc', verbose=True, thresh=0, bin_sz=10):

        phones = set()
        for p1, p2 in self._windows:
            phones.update({p1, p2})

        edge_size = max([max(conv) for conv in self._windows.values()]) // 2
        noise = {p: self._profile.get_noise(p, edge=edge_size).sum() for p in phones}
        
        n_all = {c: [] for c in powerset(phones, min_size=2)}
        n_single = {c: [] for c in powerset(phones, min_size=2)}

        interior = self._profile.cut_edge(edge_size)   

        for n_spills, spl in enumerate(self._spills):

            if verbose:
                print("{:.2f}%".format(100*n_spills/len(self._spills)),end="\r")
            
            # make counters for the spill
            all_spl = {c: 0 for c in n_all.keys()}
            single_spl = {c: 0 for c in n_single.keys()}

            # keep a cache of results
            n_coinc = {}
            n_hits_interior = {}
            n_hits_corr = {}
            first_last = {p: True for p in spl.phones}
            times = {}

            for t, p, _ in spl.gen_overlaps_single(spl.phones):
                if not p in phones: continue

                times[p] = t

                # first get the data from the individual phone
                f = spl.get_file(p, t, filetype=single_dir)
                
                intersect = np.ones(f['x'].size, dtype=bool)
                for iphone in self._profile.phones:
                    if iphone == p: continue
                    intersect &= f[iphone]
                
                x_new = (f['x'][intersect] - self._profile.x_off).astype(int)
                y_new = (f['y'][intersect] - self._profile.y_off).astype(int)
                
                if thresh:
                    above_thresh = (f['max_val'][intersect] > thresh)
                    x_new = x_new[above_thresh]
                    y_new = y_new[above_thresh]

                interior_new = interior[x_new, y_new]
                
                n_coinc[frozenset([p])] = x_new.size
                n_hits_interior[p] = interior_new.sum()
                n_hits_corr[p] = n_hits_interior[p] - noise[p]
                first_last[p] = (t in np.array(spl[p])[[0,-1]])
                                
                f.close()
                
                for c in powerset(times.keys(), min_size=2):
                    if not p in c: continue

                    # find root phone
                    pi, pj_all = self._find_optimal(c)
                    
                    t_coinc = {pc: times[pc] for pc in c}

                    # now calculate the fraction of single-frame hits in the
                    # overlap window
                    overlap = spl.calculate_overlap(*t_coinc.values())
                    if not overlap: continue

                    coinc = spl.get_coinc(t_coinc, filetype=coinc_dir)
                    if thresh:
                        coinc = coinc.threshold(thresh)
                
                    # now we calculate the noise
                    noise_tot = 0
                    for part in _partition(c):
                        
                        order = len(part)
                        if order == 1: continue # this is the signal term
                            
                        noise_contribution = self._profile.coeff(order, bin_sz=bin_sz)
                        
                        for s in map(frozenset, part):
                            if s == frozenset([pi]):
                                noise_contribution *= n_hits_interior[pi]
                            else:
                                noise_contribution *= n_coinc[s]
                            
                            if not pi in s:
                                noise_contribution *= self._factors[(pi, s)]
                        
                        noise_tot += noise_contribution
                        #print(noise_contribution)
                    
                    n_coinc[c] = coinc.x[p].size - noise_tot
                    #print(len(c), coinc.x[p].size, noise_tot)
                    all_spl[c] += n_coinc[c]  

                    corr_c = [n_hits_corr[pc] for pc in c]
                    first_last_c = [first_last[pc] for pc in c]
                    if any(first_last_c):

                        if min(corr_c) <= 0 or min(corr_c) / max(corr_c) > overlap:
                            
                            if all(first_last_c):
                                # either we're missing a frame, or this is a statistical fluke
                                single_spl[c] += overlap * max(corr_c)
                            else:
                                single_spl[c] += overlap * np.mean([n_hits_corr[pc] for pc in c if not first_last[pc]])
                        else:
                            single_spl[c] += min(corr_c)

                    else:
                        single_spl[c] += overlap * np.mean(corr_c)

            for c in n_all:
                n_all[c].append(all_spl[c])
                n_single[c].append(single_spl[c])
            
        if verbose: print("100% ", end="\r")
        return n_all, n_single


class Coincidence():

    def __init__(self, times, x, y, val, npix, windows):
        self.phones = set(times.keys())
        if not set(x.keys()) == self.phones \
                or not set(y.keys()) == self.phones \
                or not set(val.keys()) == self.phones \
                or not set(npix.keys()) == self.phones \
                or not set(windows.keys()) == self.phones:

            raise ValueError('All entries should be dictionaries with identical keys')

        self.times = times
        self.x = x
        self.y = y
        self.val = val
        self.npix = npix
        self.windows = windows

    @staticmethod
    def from_npz(fname):
        f = np.load(fname)
        
        phones = [k[2:] for k in f.keys() if k.startswith('t')]
        t = {p: f['t:{}'.format(p)] for p in phones}
        x = {p: f['x:{}'.format(p)] for p in phones}
        y = {p: f['y:{}'.format(p)] for p in phones}
        val = {p: f['val:{}'.format(p)] for p in phones}
        npix = {p: f['npix:{}'.format(p)] for p in phones}
        windows = {p: f['window:{}'.format(p)] for p in phones}
        coincidence = Coincidence(t, x, y, val, npix, windows)
        
        f.close()
        return coincidence

    def to_npz(self, fname):
        t_out = {'t:{}'.format(p): self.times[p] for p in self.phones}
        x_out = {'x:{}'.format(p): self.x[p] for p in self.phones}
        y_out = {'y:{}'.format(p): self.y[p] for p in self.phones}
        val_out = {'val:{}'.format(p): self.val[p] for p in self.phones}
        npix_out = {'npix:{}'.format(p): self.npix[p] for p in self.phones}
        windows_out = {'window:{}'.format(p): np.array(self.windows[p]) for p in self.phones}
        np.savez(fname, 
                **t_out, 
                **x_out, 
                **y_out, 
                **val_out, 
                **npix_out,
                **windows_out)

    def threshold(self, thresh):
        above_thresh = np.all([self.val[p] > thresh for p in self.phones],
                axis=0)

        x_new = {p: self.x[p][above_thresh] for p in self.phones}
        y_new = {p: self.y[p][above_thresh] for p in self.phones}
        val_new = {p: self.val[p][above_thresh] for p in self.phones}
        npix_new = {p: self.npix[p][above_thresh] for p in self.phones}

        return Coincidence(self.times, 
                x_new, 
                y_new, 
                val_new, 
                npix_new, 
                self.windows)


class Efficiency(CoincidenceCounter):
    def __init__(self, spills, profile, hist=None, cshapes={}):

        self.spills = spills
        self.profile = profile
        self.phones = sorted(self.profile.phones)

        self._hist = hist
        self._cshapes = cshapes
        self._csizes = {}
        self._optimal = {}
    
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
 


def _partition(iterable):
    collection = list(iterable)
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in _partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller


def _superpartition(collection, n_sets):
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

