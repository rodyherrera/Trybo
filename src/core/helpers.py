import numpy as np
import mmap

mm_global = None

def init_worker(filename):
    global mm_global
    f = open(filename, 'r+b')
    mm_global = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def _load_segment(range_args):
    start, end, n_cols = range_args
    seg = mm_global[start:end]
    flat = np.fromstring(seg, sep=' ')
    return flat.reshape(-1, n_cols)
