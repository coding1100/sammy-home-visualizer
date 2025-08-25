from typing import Tuple
import numpy as np

def bbox_from_poly(poly: np.ndarray):
    xs = poly[:,0]; ys = poly[:,1]
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    return [float(x0), float(y0), float(x1-x0), float(y1-y0)]