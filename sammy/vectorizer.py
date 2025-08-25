from typing import List, Dict
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union


def mask_to_polygons(mask: np.ndarray, simplify_eps: float = 2.0, min_area: float = 200.0) -> List[np.ndarray]:
    # mask: HxW bool
    m = (mask.astype('uint8')*255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, simplify_eps, True)
        poly = approx.reshape(-1, 2).astype(float)
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def polygon_to_points_attr(poly: np.ndarray) -> str:
    return " ".join([f"{x:.1f},{y:.1f}" for x,y in poly])