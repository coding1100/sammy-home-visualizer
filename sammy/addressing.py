from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class PolyInstance:
    category: str
    score: float
    polygon: np.ndarray  # Nx2 float (in resized image coords)


def _centroid(poly: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(poly[:,0])), float(np.mean(poly[:,1]))


def stable_order(instances: List[PolyInstance]) -> List[PolyInstance]:
    # Order by row (top→bottom), then left→right for stability
    def key(pi: PolyInstance):
        cx, cy = _centroid(pi.polygon)
        return (round(cy/32), cx)  # bucket rows by 32px height bands
    return sorted(instances, key=key)


def assign_ids(instances: List[PolyInstance]) -> List[tuple]:
    # Returns list of (id_str, instance)
    out = []
    grouped = {}
    for pi in instances:
        grouped.setdefault(pi.category, []).append(pi)
    for cat, items in grouped.items():
        ordered = stable_order(items)
        for idx, it in enumerate(ordered):
            out.append((f"{cat.replace(' ', '_')}-{idx}", it))
    return out


def address_for(cat: str, idx: int) -> str:
    return f"house/{cat.replace(' ', '_')}/{idx}"