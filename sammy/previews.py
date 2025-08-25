import cv2
import numpy as np
from typing import List, Tuple

COL = (255, 0, 0)


def draw_outlines(im_bgr: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    canvas = im_bgr.copy()
    for poly in polygons:
        pts = poly.reshape(-1,1,2).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=COL, thickness=2)
    return canvas