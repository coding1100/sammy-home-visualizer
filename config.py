from dataclasses import dataclass
from typing import List

DEFAULT_PROMPTS = [
    "window", "main door", "garage door", "roof", "wall", "gutter", "chimney", "porch", "trim", "shutter"
]

@dataclass
class SegmenterConfig:
    text_prompts: List[str] = None
    box_threshold: float = 0.25  # GroundingDINO text-box match
    text_threshold: float = 0.25
    mask_threshold: float = 0.0   # SAM will produce masks; we keep all and post-filter
    device: str = "cuda"
    max_side: int = 1536          # resize long side for inference

@dataclass
class VectorizeConfig:
    simplify_epsilon: float = 2.0  # px in resized space
    min_area: float = 200.0        # drop tiny fragments

@dataclass
class AddressingConfig:
    order_weight_y: float = 1.0
    order_weight_x: float = 0.7

@dataclass
class ValidateConfig:
    min_width: int = 640
    min_height: int = 480
    allowed_exts = {".jpg", ".jpeg", ".png"}
    enable_house_classifier: bool = False  # ← disable for now

SEG_CFG = SegmenterConfig(text_prompts=DEFAULT_PROMPTS)
VEC_CFG = VectorizeConfig()
ADDR_CFG = AddressingConfig()
VAL_CFG = ValidateConfig()
