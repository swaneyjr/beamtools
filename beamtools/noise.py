from glob import glob
import rawpy as rp
import gzip

class ZeroBiasSet():
    
    def __init__(self, files):
        times = []
        phones = []
        tags = []

        self._dir = None
        self._subdir = None

        for fname in map(os.path.realpath(files)):
            head, fbase = os.path.split(fname)
            head, subdir = os.path.split(head)
            head, phone = os.path.split(head)
            root_dir, tag = os.path.split(head)

            if not self._dir:
                self._dir = root_dir
            elif self._dir != root_dir:
                raise ValueError('File outside directory structure')

            if not self._subdir:
                self._subdir = subdir
            elif self._subdir != subdir:
                raise ValueError('Invalid mixing of file types')

            t = int(os.path.splitext(fbase)[0])

            times.append(t)
            phones.append(phone)
            tags.append(tag)
            
        self._times = np.array(times)
        self._phones = np.array(phones)
        self._tags = np.array(tags)


    def files(self):
        return [os.path.join(self._dir, tag, phone, self._subdir, t, '.npz') \
                for tag, phone, t in zip(self._tags, self._phones, self._times)]

    # slicing methods

    def slice(self, tag=None, t_range=None):
        cut = np.ones(self._spills.size, dtype=bool)
        if tag:
            cut &= (self._tags == tag)
        if t_range:
            tmin, tmax = t_range
            cut &= (self._times > tmin) & (self._times < tmax)
        
        return ZeroBiasSet(self._spills[cut])

    def tag(self, tag):
        return self.slice(tag=tag)

    def t_range(self, t_start, t_end):
        return self.slice(t_range=(t_start, t_end))

    def profile(self, threshrange, **kwargs): 
        profile = NoiseProfile(threshrange, **kwargs)
        profile.add_zerobias(self.files())
        return profile


class NoiseProfile():
   
    def __init__(self, threshrange, downsample=97):
        self._histograms = {}
        self._thresh_min, self._thresh_max = threshrange
        self._ds = downsample
        self._nframes = 0

    def phones(self):
        return list(self._histograms.keys())

    def add_zerobias(self, files):
        n_frames = {}

        bins = np.arange(self._thresh_min, self._thresh_max+1)

        for fname in map(os.path.realpath(files)):
            head, fbase = os.path.split(fname)
            head, subdir = os.path.split(head)
            head, phone = os.path.split(head)
            root_dir, tag = os.path.split(head)

            if not phone in self._histograms:
                self._histograms[phone] = 0
                self._nframes[phone] = 0
            
            self._nframes[phone] += 1

            f = gzip.open(fname) if f.endswith('.gz') else open(fname)
            image = rawpy.imread(f)
            raw_image = unmunge(image.raw_image.transpose()) if unmunge else image.raw_image.transpose()

            res_y, res_x = raw_image.shape
            raw_image = raw_image.reshape(res_y // self._ds, 
                    self._ds,
                    res_x // self._ds,
                    self._ds)

            raw_image = np.minimum(raw_image, self._thresh_max)

            self._histograms[phone] += np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 
                    0, 
                    raw_image)
           
            image.close()
            f.close()

    def get_profile(self, phone, thresh, convolve=True):
        downsampled = self._get_downsampled(thresh_phone)
        if convolve:
            downsampled = scipy.signal.convolve2d(downsampled, np.ones((3,3))/9, mode='same', boundary='symm')
        
        res_y, res_x = self._ds * np.array(downsampled.shape)
        
        return cv2.resize(downsampled, (res_y, res_x)) / n_frames[p] / self._ds**2

        

    def get_total(self, thresh, phone):
        if not phone in self._histograms: return 0
        return np.sum(self._get_downsampled(thresh, phone))

    def visualize(self, thresh, phone=None):
        if phone:
            self._vis_phone(thresh, phone)
        else:
            for p in self.phones():
                self._vis_phone(thresh, p)

    def _vis_phone(self, thresh, phone):
        plt.imshow(self._get_downsampled(thresh, phone) / self._nframes[iphone])
        plt.show()

    def _get_downsampled(self, thresh, phone):
        if thresh >= self._thresh_max:
            raise ValueError('Out of specified threshold range: [{},{})'.format(self._thresh_min, self._thresh_max))
        idx_min = max(thresh - self._thresh_min + 1, 0)
        return np.sum(self._histograms[iphone][idx_min:], axis=0)

