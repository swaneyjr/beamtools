import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering


def hotcells(image, hotc): 

    x_res = image.shape[1]
    hotx = hotc % (x_res)
    hoty = hotc // (x_res)

    maskedarray = np.ma.masked_array(image)
    maskedarray[hoty,hotx] = np.ma.masked
    return np.ma.filled(maskedarray,0)

def apply_threshold(image, threshold, hotc):

    if not hotc is None:
        image = hotcells(image, hotc)

    y, x = np.array(np.nonzero(image > threshold))
    val = image[y, x]
   
    return x, y, val 


def unmunge(data):
    p0 = np.bitwise_and(data[0::4], 3)
    p1 = np.bitwise_and(data[1::4], 3)
    p2 = np.bitwise_and(data[2::4], 3)
    p3 = np.bitwise_and(data[3::4], 3)

    data[1::4] = np.bitwise_and(data[1::4], 1020) + p2
    data[2::4] = np.bitwise_and(data[2::4], 1020)
    data[3::4] = np.bitwise_and(data[3::4], 1020)

    return data


def cluster(x, y, val, threshold, clusteringOption=1):

    # create clusters
    xyval = np.column_stack((x,y,val))
        
    if clusteringOption == 1:
        clustering = DBSCAN(eps = threshold, min_samples = 1) 
    elif clusteringOption == 2:
        clustering = AgglomerativeClustering(n_clusters = None, compute_full_tree = True, distance_threshold = threshold)

    clustering.fit(xyval[:,:-1])

    ordered_indices = np.argsort(clustering.labels_)
    ordered_labels = clustering.labels_[ordered_indices]

    # split into groups
    diff = np.diff(ordered_labels)
    locations_to_split = (np.argwhere(diff != 0) + 1).flatten()

    groups = np.array_split(ordered_indices, locations_to_split)
    
    return [xyval[gp] for gp in groups]


def hist_zerobias(res, *dng, ds=97, visualize=False):
    
    histograms = {}
    n_frames = {}
    noise = {}

    res_x, res_y = res

    for fname in dng:
        p = fname.split('_')[0]
        if not p in histograms:
            histograms[p] = 0
            n_frames[p] = 0
        
        f = np.load(fname)
        
        histograms[p] += csr_matrix((np.ones(f['x'].size), (f['x'], f['y'])), shape=res)
        n_frames[p] += 1
        
        f.close()

    for p,h in histograms.items():
        downscale = h.toarray().reshape(res_x//ds, ds, res_y//ds, ds).sum((1,3))
        downscale = scipy.signal.convolve2d(downscale, np.ones((3,3))/9, mode='same', boundary='symm')
        noise[p] = cv2.resize(downscale, (res_y, res_x)) / n_frames[p] / downsample**2
        
        if visualize:
            plt.figure()
            plt.imshow(noise[p].transpose(), cmap='viridis')
            plt.colorbar()

    return noise
