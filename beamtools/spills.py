import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.stats import linregress


class Spill():

    def __init__(self, res_x, res_y, fps, basedir, tag):
        self.res_x = res_x
        self.res_y = res_y
        self.fps = fps
        self.basedir = basedir
        self.tag = tag
        self._d = {}
        self._tmin = np.inf
        self._tmax = -np.inf

    def __getitem__(self, phone):
        return self._d[phone]

    def phones(self):
        return list(self._d.keys())

    def append(self, phone, t):
        if not phone in self._d.keys():
            self._d[phone] = [t]
        else:
            self._d[phone].append(t)

        self._tmin = min(self._tmin, t)
        self._tmax = max(self._tmax, t)
            
    def get_file(self, phone, t, filetype):
        fname = os.path.join(self.basedir, self.tag, phone, filetype, '{}.npz'.format(t))
        return np.load(fname)
    
    def histogram(self, phone, filetype, downsample=4):
        hist = 0
        
        for t in self._d[phone]:
            f = self.get_file(phone, t, filetype)
            res_y_down = self.res_y // downsample
            res_x_down = self.res_x // downsample
            hist += np.histogram2d(f.f.y, f.f.x, bins=(res_y_down, res_x_down), range=((0, self.res_y),(0, self.res_x)))[0]
            f.close()
        return hist


    def _calculate_overlap(self, *t):
        diff = max(t) - min(t)
        return max(1 - self.fps * diff / 1000, 0)

    # iterator through overlapping frames
    def gen_overlaps(self, phones):

        phones = np.array(phones)
        times = [self._d[p].copy() for p in phones]
        t_ijk = np.array([t.pop(0) for t in times]).astype(int)
        while True:
            overlap = self._calculate_overlap(*t_ijk)        
            if overlap:
                yield t_ijk, overlap

            # replace the earliest time, if possible
            p_earliest = np.argmin(t_ijk)
            if not times[p_earliest]: break
            t_ijk[p_earliest] = times[p_earliest].pop(0)
        

    def gen_overlaps_single(self, phones):
        phones = list(phones)
        times = {p: self._d[p].copy() for p in phones}
        t_ijk = [times[p].pop(0) for p in phones]
        
        sort = np.argsort(t_ijk)
        phones = [phones[arg] for arg in sort]
        t_ijk = [t_ijk[arg] for arg in sort]
        
        initial_overlap = _calculate_overlap(*t_ijk)
        for t, phone in zip(t_ijk[:-1], phones[:-1]):
            yield t, phone, 0
        yield t_ijk[-1], phones[-1], initial_overlap
        
        while True:
            # replace the earliest time, if possible
            t_ijk = t_ijk[1:]
            p_earliest = phones.pop(0)
            
            if not times[p_earliest]: break
            t_new = times[p_earliest].pop(0)
            
            t_ijk.append(t_new)
            phones.append(p_earliest)
            overlap = _calculate_overlap(*t_ijk)
        
            yield t_new, p_earliest, overlap



class RawSpillSet():

    def __init__(self, res, fps):
        self.res_x, self.res_y = res
        self.fps = fps
        self.dir = None
        self.subdir = None
        self._df = {}
        self.flux = {}
        self._intervals = {}
        self._noise = {}


    def __getitem__(self, iphone):
        return self._df[iphone]

    def __setitem__(self, key, val):
        self._df[key] = val

    def __iter__(self):
        return self._df.__iter__()

    def add_files(self, *files):

        phones = []
        t_nominal = []
        t_filename = []
        tags = []

        for fname in map(os.path.realpath, files):

            # first sort through filename
            head, fbase = os.path.split(fname)
            head, subdir = os.path.split(head)
            head, iphone = os.path.split(head)
            root_dir, tag = os.path.split(head)

            t = int(os.path.splitext(fbase)[0])

            # now make sure directory structure is consistent

            if not self.dir:
                self.dir = root_dir
                self.subdir = subdir
            elif self.dir != root_dir or self.subdir != subdir:
                raise
                        
            
            # in case this is run multiple times, we use the original timestamp saved in the raw file
            f = np.load(fname)
            
            phones.append(iphone)
            t_nominal.append(int(f.f.t))
            t_filename.append(t)
            tags.append(tag)

            f.close()
        
        sorting = np.argsort(t_nominal)
        t_nominal = np.array(t_nominal)[sorting]
        t_filename = np.array(t_filename)[sorting]
        phones = np.array(phones)[sorting]
        tags = np.array(tags)[sorting]
        
        for p in set(phones):
            cut = (phones == p)
            self._df[p] = pd.DataFrame(data={
                'tag': tags[cut], 
                'nominal': t_nominal[cut], 
                'filename': t_filename[cut],
                'corr': t_filename[cut]
                })

        self.flux = {p: RawSpillSet.FluxEstimate(self, p) for p in self} 
        for p in self: 
            self._calculate_intervals(p)


    def _get_filename(self, phone, row):
        return os.path.join(self.dir, row['tag'], phone, self.subdir, '{}.npz'.format(row['filename']))
 

    class FluxEstimate():

        def __init__(self, raw_spill_set, iphone):
            n_hits = []
            df = raw_spill_set[iphone]
            for _, row in df.iterrows():
                f = np.load(raw_spill_set._get_filename(iphone, row))
                n_hits.append(f.f.n_clusters) 
                f.close()
                
            # exclude the first and last frame of each beam dump
            t_diffs = np.hstack([[1e5], np.diff(df['nominal']), [1e5]]) 
                
            cut = (t_diffs[1:] < 2.5e3 / raw_spill_set.fps) \
                    & (t_diffs[:-1] < 2.5e3 / raw_spill_set.fps)

            self._phone = iphone
            self._times = df['nominal'].to_numpy()
            self._vals = np.array(n_hits)
            self._timesx = self._times[cut]
            self._valsx = self._vals[cut]

            self.split()


        def visualize(self, vistype='timeplot', xlim=None, **kwargs):
            plt.figure(figsize=(11,4))
            plt.title(self._phone[:6])
            if vistype == 'timeplot':
                plt.xlabel('t')
                default_kwargs = {
                        'marker': '.',
                        'ls': ''
                        }
                default_kwargs.update(kwargs)
                plt.plot(self._times, self._vals, label='Observed flux', **default_kwargs)
                plt.plot(self._times, self.get_flux(self._times), lw=2.0, label='Median flux')
                plt.legend()

            elif vistype == 'histplot':
                plt.xlabel('Hits per frame')
                splits_all = self._splits + [self._times.max() + 1]
                for i, start, end in zip(range(len(self._splits)), splits_all[:-1], splits_all[1:]):
                    flux = self._valsx[(self._timesx >= start) \
                            & (self._timesx < end)]

                    default_kwargs = {
                            'bins': 50,
                            'histtype': 'step',
                            'log': True
                            }
                    default_kwargs.update(kwargs)
                    plt.hist(flux, label=i+1, **default_kwargs) 
            

                plt.legend()

            if xlim:
                plt.xlim(*xlim)
            
            plt.show()


        def split(self, split=()):
            
            flux_vals = []

            bounds = [self._timesx[0]]
            bounds += list(split)
            bounds.append(self._timesx[-1]+1)

            self._splits = bounds[:-1]
            tf = bounds[1:]

            for start, end in zip(self._splits, tf):
                flux = self._valsx[(self._timesx >= start) \
                        & (self._timesx < end)]

                flux_vals.append(np.median(flux))
                
            self._flux_vals = np.array(flux_vals)


        def get_flux(self, t):
            t = np.array(t).reshape(-1,1)
            return self._flux_vals[np.argmin(t >= self._splits, axis=1)]
 

    def add_noise(self, noise={}):
        self._noise.update(noise)
        for p in noise:
            self._calculate_intervals(p)


    def _calculate_intervals(self, iphone, cutoff=5e3):

        t_diffs = np.hstack([[1e5], np.diff(self[iphone]['nominal']), [1e5]])
        
        t_len = self[iphone]['nominal'].size
        start_args = np.arange(t_len)[(t_diffs[1:] <= cutoff) \
                & (t_diffs[:-1] > 5e3)]
        end_args = np.arange(t_len)[(t_diffs[1:] > cutoff) \
                & (t_diffs[:-1] <= 5e3)]
        
        t_start_corr = []
        t_end_corr = []

        flux = self.flux[iphone]
        noise = self._noise[iphone] if iphone in self._noise else 0
        
        for arg in start_args:
            t_series = self[iphone].iloc[arg]
            t_nominal = t_series['nominal']
            f = np.load(self._get_filename(iphone, t_series))
            t_start_corr.append(t_nominal + 1000 / self.fps * (1 - f.f.n_clusters - noise) / (flux.get_flux(t_nominal) - noise))
            f.close()
            
        for arg in end_args: 
            t_series = self[iphone].iloc[arg]
            t_nominal = t_series['nominal']
            f = np.load(self._get_filename(iphone, t_series))
            t_end_corr.append(t_nominal + 1000 / self.fps * (f.f.n_clusters - noise) / (flux.get_flux(t_nominal) - noise))
            f.close()
            
        self._intervals[iphone] = np.vstack([t_start_corr, t_end_corr])
    
    
    def calibrate(self, p_root=None, visualize=False):

        if not p_root:
            p_root = list(self._df.keys())[0]

        times_root = self._intervals[p_root].flatten()
        times_root.sort()

        for p_other in self:
            if p_other == p_root:
                self[p_root]['corr'] = self[p_root]['nominal']
                continue
        
            times_other = self._intervals[p_other].flatten()
            times_other.sort()

            m,b,r,p,s = linregress(times_other, times_root)
            self[p_other]['corr'] = (m * self[p_other]['nominal'] + b).astype(int)

            if visualize:
                plt.xlabel(p_other[:6])
                plt.ylabel(p_root[:6])
                plt.plot(times_other[-8:], times_other[-8:], 'r.', markersize=5, label='Nominal')
                plt.plot(times_other[-8:], times_root[-8:], 'b.', markersize=5, label='Actual')
                plt.plot(times_other[-8:], m * times_other[-8:] + b, lw=0.5, label='Correction')
                plt.legend()
                plt.show()

     
    def visualize(self, metric='corr', delta=False, aggregate=False, **kwargs):
        if aggregate:
            times = {'Total': np.hstack([t[metric] for t in self._df.values()])}
        else:
            times = {phone: self[phone][metric] for phone in self._df}

        for phone, t in times.items():
            if delta:
                dt = np.diff(np.sort(t))
                plt.title(r'$\Delta t$ between images')
                default_kwargs = {
                        'bins': 50,
                        'log': True,
                        'histtype': 'step'
                        }
                default_kwargs.update(kwargs)
                plt.hist(dt, label=phone[:6], **default_kwargs)
            else:
                plt.title('Image times')
                default_kwargs = {
                        'bins': 50,
                        'histtype': 'step',
                        }
                default_kwargs.update(kwargs)
                plt.hist(t, label=phone[:6], **default_kwargs)
        
        plt.legend()
        plt.show()


    def prune(self, phone, dtmin, dtmax, metric='corr'):
        sorted_args = np.argsort(self[phone][metric])

        # find inverse sorting
        n_args = sorted_args.size
        rev_sorted_args = np.zeros(n_args, dtype=int)
        rev_sorted_args[sorted_args] = np.arange(n_args)
        
        dt = np.diff(self[phone][metric][sorted_args])
        dt_cut = (dt > dtmin) & (dt < dtmax)

        prune_args = np.hstack([[dt_cut[0] & np.logical_not(dt_cut[1])],
            dt_cut[1:] & dt_cut[:-1],
            dt_cut[-1] & np.logical_not(dt_cut[-2])])

        self[phone] = self[phone][rev_sorted_args[np.logical_not(prune_args)]]


    def apply_calibration(self, dtmax=5e3):
        
        # rename files
        for iphone in self:
            for _, row in self[iphone].iterrows():
                f_full = self._get_filename(iphone, row)
                f_corr = f_full.replace(str(row['filename']), str(row['corr']))
                os.rename(f_full, f_corr)
            self[iphone]['filename'] = self[iphone]['corr']
        
        # create SpillSet
        tag_sorted = []
        phone_sorted = []
        time_sorted = []

        for phone, df in self._df.items():

            time_sorted.append(df['filename'])
            tag_sorted.append(df['tag'])
            phone_sorted.append([phone] * len(df))

        time_sorted = np.hstack(time_sorted)
        phone_sorted = np.hstack(phone_sorted)
        tag_sorted = np.hstack(tag_sorted)

        sorting = np.argsort(time_sorted)

        time_sorted = time_sorted[sorting]
        phone_sorted = phone_sorted[sorting]
        tag_sorted = tag_sorted[sorting]

        dt = np.diff(time_sorted)

        def _get_tuple(ientry):
            return tag_sorted[ientry], phone_sorted[ientry], time_sorted[ientry]
        
        spills = []
        spill_lengths = []
        imin = 0
 
        tag, iphone, t = _get_tuple(0)
        ispill = Spill(self.res_x,
                self.res_y,
                self.fps,
                self.dir, 
                tag)
        ispill.append(iphone, t)

        for i, delta in enumerate(dt):
            tag, iphone, t = _get_tuple(i+1)
            if delta > dtmax:
                spill_lengths.append(time_sorted[i] - time_sorted[imin])
                spills.append(ispill)

                imin = i+1
                # create a new spill
                ispill = Spill(self.res_x, 
                        self.res_y, 
                        self.fps,
                        self.dir,
                        tag)
            
            elif tag != ispill.tag:
                print('Skipping file with incorrect tag.')
                continue

            ispill.append(iphone, t)

        spills.append(ispill)

        return SpillSet(spills)



class SpillSet():

    def __init__(self, spills=[]):
        for spl in spills:
            if not isinstance(spl, Spill):
                raise ValueError
        self._spills = np.array(spills)
        self._tag = np.array([spl.tag for spl in spills])
        self._tmin = np.array([spl._tmin for spl in spills])
        self._tmax = np.array([spl._tmax for spl in spills])

    def __getitem__(self, key):
        return self._spills[key]

    def __iter__(self):
        return self._spills.flat

    def __len__(self):
        return self._spills.size

    def to_npz(self, fname):
        np.savez(fname, spills=np.array(self))

    @staticmethod
    def from_npz(fname):
        f = np.load(fname, allow_pickle=True)
        spill_set = SpillSet(f.f.spills)
        f.close()
        return spill_set

    def gen_overlaps(self, phones):
        for spl in self:
            for yld in spl.gen_overlaps(phones):
                yield yld

    def gen_overlaps_single(self, phones):
        for spl in self:
            for yld in spl.gen_overlaps(phones):
                yield yld

    # slicing methods

    def slice(self, tag=None, t_range=None, dt=None):
        cut = np.ones(self._spills.size)
        if tag:
            cut &= (self._tag == tag)
        if t_range:
            tmin, tmax = t_range
            cut &= (self._tmin > tmin) & (self._tmax < tmax)
        if dt:
            cut &= (self._tmax - self._tmin < dt)

        return SpillSet(self._spills[cut])

    def tag(self, tag):
        return self.slice(tag=tag)

    def t_range(self, t_start, t_end):
        return self.slice(t_range=(t_start, t_end))

    def dt(self, dt):
        return self.slice(dt=dt)

     
