import rawpy as rp
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from glob import glob

import os
import gzip

class ZeroBiasSet():
    
    def __init__(self, root_dir, tags, phones, times):

        self._dir = root_dir
        self._tags = np.array(tags)
        self._phones = np.array(phones)
        self._times = np.array(times)

    @staticmethod
    def from_files(files):
        times = []
        phones = []
        tags = []

        root_dir = None

        for fname in map(os.path.realpath, files):
            head, fbase = os.path.split(fname)
            head, this_subdir = os.path.split(head)
            head, phone = os.path.split(head)
            this_root_dir, tag = os.path.split(head)

            if not root_dir:
                root_dir = this_root_dir
            elif root_dir != this_root_dir:
                raise ValueError('File outside directory structure')

            t = int(fbase.split('.')[0])

            times.append(t)
            phones.append(phone)
            tags.append(tag)
            
        times = np.array(times)
        phones = np.array(phones)
        tags = np.array(tags)

        return ZeroBiasSet(root_dir, tags, phones, times)

    def files(self, filetype='cluster', phone=None):
        ext = '{}.dng.gz' if filetype=='triggered_image' else '{}.npz'
        for tag, p, t in zip(self._tags, self._phones, self._times):
            if phone and p != phone: continue
            yield os.path.join(
                self._dir, 
                tag, 
                p, 
                filetype, 
                ext.format(t))
        

    # slicing methods

    def slice(self, tag=None, t_range=None):
        cut = np.ones(self._times.size, dtype=bool)
        if tag:
            cut &= (self._tags == tag)
        if t_range:
            tmin, tmax = t_range
            cut &= (self._times > tmin) & (self._times < tmax)
        
        return ZeroBiasSet(self._dir, 
                self._tags[cut], 
                self._phones[cut],
                self._times[cut])

    def tag(self, tag):
        return self.slice(tag=tag)

    def t_range(self, t_start, t_end):
        return self.slice(t_range=(t_start, t_end))

    def cut_beam(self, val, nmax, phone=None, filetype='cluster'):
        files = []
        for fname in self.files():

            if filetype == 'triggered_image':
                f = gzip.open(fname)
                im = rp.imread(f)
                if np.sum(im.raw_image > val) <= nmax:
                    files.append(fname)
                
                im.close()
                f.close()

            else:
                f = np.load(fname)
                vals = f.f.val if 'val' in f.keys() else f.f.max_val
                if np.sum(vals >= val) <= nmax:
                    files.append(fname)
                f.close()

        return ZeroBiasSet.from_files(files)


    # exporting results

    def profile(self, thresh, filetype='cluster', **kwargs): 
        profile = NoiseProfile(thresh, **kwargs)
        profile.add_zerobias(self.files(filetype=filetype))
        return profile
    
    @staticmethod
    def from_npz(fname):
        f = np.load(fname, allow_pickle=True)
        zb_set = ZeroBiasSet(str(f.f.dir),
                f.f.tags,
                f.f.phones,
                f.f.times)
        f.close()
        return zb_set

    def to_npz(self, fname):
        np.savez(fname, 
                dir=self._dir,
                tags=self._tags,
                phones=self._phones,
                times=self._times)

    def visualize(self, thresh, filetype='cluster'):

        for p in set(self._phones):    
            times = []
            counts = []
            for fname in self.files(filetype=filetype, phone=p):
                times.append(int(os.path.basename(fname).split('.')[0]))

                if filetype == 'triggered_image':
                    f = gzip.open(fname)
                    im = rp.imread(f)

                    counts.append(np.sum(im.raw_image > thresh))

                    im.close()
                    f.close()

                else:
                    f = np.load(fname)
                    vals = f.f.val if 'val' in f.keys() else f.f.max_val

                    counts.append(np.sum(vals >= thresh))
                
                    f.close()

            times = np.array(times)
            counts = np.array(counts)
            sort = np.argsort(times)
            
            plt.plot(times[sort], counts[sort], '.', ls='', label=p[:6])

        plt.legend()
        plt.show()


class NoiseProfile():
   
    def __init__(self, thresh):
        self.thresh = thresh
        self._sparse = {}
        self._nframes = {}

    def __getitem__(self, phone):
        return self._sparse[phone].toarray() / self._nframes[phone]

    @property
    def phones(self):
        return list(self._sparse.keys())

    def add_zerobias(self, files, do_unmunge=False):
        n_frames = {}

        for fname in map(os.path.realpath, files):
            if not fname.endswith('.npz'): 
                raise ValueError('Filetype not supported')
            head, fbase = os.path.split(fname)
            head, subdir = os.path.split(head)
            head, phone = os.path.split(head)
            root_dir, tag = os.path.split(head)

            if not phone in self._sparse:
                self._sparse[phone] = 0
                self._nframes[phone] = 0
            
            self._nframes[phone] += 1

            f = np.load(fname)
            
            val = f.f.val if 'val' in f.keys() else f.f.max_val
            x = f.f.x[val > self.thresh]
            y = f.f.y[val > self.thresh]

            self._sparse[phone] += csr_matrix((np.ones(val.size), (x, y)), 
                    shape=(4656, 3492))
           
            f.close()

    def _convolve_single(self, phone, ds=97):
        downsampled = self._get_downsampled(phone, ds)
        downsampled = scipy.signal.convolve2d(downsampled, np.ones((3,3))/9, mode='same', boundary='symm')
        res_y, res_x = ds * np.array(downsampled.shape)
        
        return cv2.resize(downsampled, (res_y, res_x)) / n_frames[p] / self._ds**2

    def convolve(self, phone=None, ds=97):
        if phone:
            return self._convolve_single(phone, ds)
        else:
            return {self._convolve_single(p, ds) for p in self.phones}

    def get_total(self, phone):
        if not phone in self._histograms: return 0
        return self._sparse[phone].sum() / self._nframes[iphone]

    def visualize(self, phone=None, downsample=97):
        if phone:
            self._vis_phone(phone, downsample)
        else:
            for p in self.phones:
                self._vis_phone(p, downsample)

    def _vis_phone(self, phone, ds):
        plt.imshow(self._get_downsampled(phone, ds) / self._nframes[iphone])
        plt.show()

    def _get_downsampled(self, phone, ds):
        arr = self._sparse[iphone].toarray()
        return np.sum(arr.reshape(arr.shape[0]//ds, ds, arr.shape[1]//ds, ds), 
                axis=(1,3))

