import os
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Tuple
import torch
from transformers import CLIPProcessor, CLIPModel

HOUSE_LABELS = [
    "house facade", "residential house front", "front of a house",
    "apartment facade", "building facade"
]
NON_HOUSE_LABELS = [
    "interior room", "kitchen", "bathroom", "bedroom", "garden close-up",
    "car interior", "people portrait", "sky only"
]

@dataclass
class ValidationReport:
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    house_score: float = 0.0

# sammy/validators.py
_clip_cache = {"model": None, "processor": None, "device": None}

def _get_clip(device: str = "cpu"):
    # lazy imports to avoid loading torch at module import
    import torch
    from transformers import CLIPProcessor, CLIPModel

    if _clip_cache["model"] is None or _clip_cache["device"] != device:
        _clip_cache["model"] = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True  # ← crucial: avoids .bin + torch.load
        ).to(device)
        _clip_cache["processor"] = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        _clip_cache["device"] = device
    return _clip_cache["model"], _clip_cache["processor"]

def basic_file_checks(path: Path, min_w: int, min_h: int, exts) -> Tuple[bool, str, int, int]:
    if path.suffix.lower() not in exts:
        return False, f"Unsupported extension {path.suffix}", 0, 0
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:
        return False, f"Cannot open image: {e}", 0, 0
    if w < min_w or h < min_h:
        return False, f"Image too small ({w}x{h})", w, h
    return True, "ok", w, h


def classify_house_front(path: Path, device: str = "cuda") -> float:
    model, processor = _get_clip(device)
    image = Image.open(path).convert("RGB")
    texts = [f"a photo of {t}" for t in HOUSE_LABELS + NON_HOUSE_LABELS]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits_per_image.squeeze(0).softmax(dim=-1).cpu().tolist()
    # Aggregate: sum probs over house labels
    house_prob = sum(logits[:len(HOUSE_LABELS)])
    return house_prob


def validate_input(path: Path, min_w: int, min_h: int, exts, device: str = "cpu", enable_classifier: bool = False) -> ValidationReport:
    ok, reason, w, h = basic_file_checks(path, min_w, min_h, exts)
    if not ok:
        return ValidationReport(False, reason, w, h, 0.0)
    if not enable_classifier:
        return ValidationReport(True, "ok (classifier skipped)", w, h, 1.0)
    score = classify_house_front(path, device)
    if score < 0.4:
        return ValidationReport(False, f"Not a clear house-front (score={score:.2f})", w, h, score)
    return ValidationReport(True, "ok", w, h, score)

