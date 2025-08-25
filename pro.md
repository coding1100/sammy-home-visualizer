# Sammy Home Visualizer — Vision-to-SVG Prototype (v0.1)

This is a **runnable skeleton** for converting a house-front image (`.jpg/.jpeg/.png`) into an **annotated SVG** + **sidecar JSON** with addressable façade elements.

> **Stack choice (POC):** Grounded-SAM (GroundingDINO + SAM) → masks → polygons (OpenCV/Shapely) → SVG writer. Optional CLIP classifier for attributes.

---

## 📁 Repo Layout

```
sammy-vision/
├─ README.md
├─ requirements.txt
├─ analyze.py                 # CLI entrypoint: image → svg + json
├─ api.py                     # FastAPI service exposing /analyze
├─ config.py                  # Model + thresholds + category prompts
├─ sammy/
│  ├─ __init__.py
│  ├─ validators.py           # file checks + “house-front” classifier (CLIP zero-shot)
│  ├─ segmenter.py            # Grounded-SAM wrapper (GroundingDINO + SAM)
│  ├─ vectorizer.py           # masks → polygons (contours + simplification)
│  ├─ addressing.py           # stable IDs (L→R, T→B), address schema
│  ├─ svg_writer.py           # SVG generation with <g>, <polygon>, data-*
│  ├─ schema.py               # JSON sidecar + dataclasses
│  ├─ previews.py             # outline/labeled preview assets
│  └─ utils.py                # IO, geometry helpers, timing, hashing
└─ tests/
   └─ test_smoke.py
```

---

## 🧩 requirements.txt

```txt
# Core CV + geometry
opencv-python>=4.9
numpy<2 ; platform_system == "Darwin" and platform_machine == "x86_64"  # Intel mac needs NumPy 1.x with Torch 2.2
numpy>=1.26 ; platform_system != "Darwin" or platform_machine != "x86_64"
shapely>=2.0
Pillow>=10.0
scikit-image>=0.23

# Models
# PyTorch matrix (see notes):
# - macOS Intel (x86_64): last supported torch==2.2.*, torchvision==0.17.*
# - Apple Silicon (arm64) & Linux/Windows: newer torch ok
torch==2.2.* ; platform_system == "Darwin" and platform_machine == "x86_64"
torchvision==0.17.* ; platform_system == "Darwin" and platform_machine == "x86_64"

# Non-Intel macs & others (adjust as desired)
torch==2.8.* ; platform_system != "Darwin" or platform_machine != "x86_64"
torchvision==0.19.* ; platform_system != "Darwin" or platform_machine != "x86_64"

transformers>=4.41
huggingface_hub>=0.23

# SAM (install from GitHub)
# uv/pip will fetch this repo directly
git+https://github.com/facebookresearch/segment-anything.git

# API & CLI
fastapi>=0.111
uvicorn>=0.30
pydantic>=2.7
click>=8.1
python-multipart>=0.0.9

# Utils
tqdm>=4.66
```

> **Note:** Some model repos (e.g., GroundingDINO) may require install from source. You can vendor their minimal inference modules into `sammy/third_party/` if needed.

---

## ⚙️ config.py

```python
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

SEG_CFG = SegmenterConfig(text_prompts=DEFAULT_PROMPTS)
VEC_CFG = VectorizeConfig()
ADDR_CFG = AddressingConfig()
VAL_CFG = ValidateConfig()
```

---

## 🧪 validators.py

```python
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

_clip_cache = {"model": None, "processor": None}

def _get_clip(device: str = "cuda"):
    if _clip_cache["model"] is None:
        _clip_cache["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_cache["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
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


def validate_input(path: Path, min_w: int, min_h: int, exts, device: str = "cuda") -> ValidationReport:
    ok, reason, w, h = basic_file_checks(path, min_w, min_h, exts)
    if not ok:
        return ValidationReport(False, reason, w, h, 0.0)
    try:
        score = classify_house_front(path, device)
    except Exception as e:
        return ValidationReport(False, f"Classifier failed: {e}", w, h, 0.0)
    if score < 0.4:
        return ValidationReport(False, f"Not a clear house-front (score={score:.2f})", w, h, score)
    return ValidationReport(True, "ok", w, h, score)
```

---

## 🧠 segmenter.py — HF Grounding DINO + SAM (CPU‑friendly)

```python
# sammy/segmenter.py
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry, SamPredictor

@dataclass
class DetInstance:
    category: str
    score: float
    bbox_xyxy: List[float]
    mask: np.ndarray  # HxW bool

class GroundedSAMPredictor:
    """
    Grounding DINO (HF Transformers) → boxes, then SAM (vit_b by default) → masks.
    Works on CPU; set device="cuda" if you have a GPU.
    """
    def __init__(self, device: str = "cpu", box_thr: float = 0.25, text_thr: float = 0.25, max_side: int = 1280, sam_variant: str = "vit_b"):
        self.device = torch.device(device)
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.max_side = max_side

        # --- Grounding DINO via Transformers (auto-downloads weights)
        self.gdino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(self.device)
        self.gdino_model.eval()

        # --- SAM weights via HF Hub (vit_b is relatively light & OK on CPU)
        if sam_variant == "vit_b":
            repo_id, filename, sam_key = "facebook/segment-anything", "sam_vit_b_01ec64.pth", "vit_b"
        elif sam_variant == "vit_h":
            repo_id, filename, sam_key = "facebook/segment-anything", "sam_vit_h_4b8939.pth", "vit_h"
        else:
            raise ValueError(f"Unsupported sam_variant: {sam_variant}")

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        sam = sam_model_registry[sam_key](checkpoint=ckpt_path)
        sam.to(self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

    def predict(self, image_path: str, text_prompts: List[str]) -> List[DetInstance]:
        # 0) Load image (keep original size so GDINO post-process can map boxes correctly)
        im = Image.open(image_path).convert("RGB")
        W, H = im.size

        # 1) Text-prompted detection → boxes (xyxy in original pixel space)
        inputs = self.gdino_proc(images=im, text=text_prompts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.gdino_model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=self.device)
        det = self.gdino_proc.post_process_grounded_object_detection(
            outputs,
            inputs,
            box_threshold=self.box_thr,
            text_threshold=self.text_thr,
            target_sizes=target_sizes,
        )[0]
        boxes_xyxy = det["boxes"]            # (N,4) torch, in original pixels
        labels = det["labels"]               # list[str]
        scores = det.get("scores", torch.ones(len(labels), device=self.device))

        # 2) For each box, get a fine-grained mask from SAM
        np_img = np.array(im)
        self.sam_predictor.set_image(np_img)
        # Map boxes into SAM's transformed coordinate frame
        transformed = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, np_img.shape[:2])

        results: List[DetInstance] = []
        for i, (tbox, lab, sc) in enumerate(zip(transformed, labels, scores)):
            # SAM expects a numpy box (xyxy), single instance
            box_np = tbox.cpu().numpy()
            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=False,
            )
            mask = masks[0].astype(bool)  # HxW bool in original image coords
            results.append(DetInstance(
                category=str(lab),
                score=float(sc),
                bbox_xyxy=boxes_xyxy[i].tolist(),
                mask=mask,
            ))
        return results
```

---

## 🧭 addressing.py

```python
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
```

---

## ✏️ vectorizer.py

```python
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
```

---

## 🖼️ svg\_writer.py

```python
from typing import List, Dict, Any
from xml.etree.ElementTree import Element, SubElement, tostring
from dataclasses import dataclass
import base64

@dataclass
class SVGItem:
    id: str
    category: str
    address: str
    points: str
    score: float

SVG_HEADER_STYLE = """
  .interactive { pointer-events: all; cursor: pointer; }
  [data-selected="true"] { outline: 2px dashed #4AA3FF; }
"""


def build_svg(width: int, height: int, items: List[SVGItem], origin_name: str) -> str:
    svg = Element('svg', attrib={
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f"0 0 {width} {height}",
    })
    defs = SubElement(svg, 'defs')
    style = SubElement(defs, 'style')
    style.text = SVG_HEADER_STYLE

    house = SubElement(svg, 'g', attrib={'id': 'house', 'data-type': 'house', 'data-origin': origin_name})

    groups: Dict[str, Element] = {}
    for it in items:
        grp = groups.get(it.category)
        if grp is None:
            grp = SubElement(svg, 'g', attrib={'id': it.category.replace(' ', '_'), 'class': 'interactive', 'data-type': it.category})
            groups[it.category] = grp
        SubElement(grp, 'polygon', attrib={
            'id': it.id,
            'points': it.points,
            'data-type': it.category,
            'data-address': it.address,
            'data-confidence': f"{it.score:.2f}",
        })

    return tostring(svg, encoding='unicode')
```

---

## 📦 schema.py

```python
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ElementJSON:
    id: str
    address: str
    category: str
    polygon: list  # [[x,y], ...]
    bbox: list     # [x,y,w,h]
    confidence: float
    attributes: Dict[str, Any]

@dataclass
class AnalysisJSON:
    image: Dict[str, Any]
    elements: List[ElementJSON]
    relations: List[Dict[str, str]]
    version: str = "0.1.0"

    def to_dict(self):
        return {
            'image': self.image,
            'elements': [asdict(e) for e in self.elements],
            'relations': self.relations,
            'version': self.version,
        }
```

---

## 🧰 utils.py

```python
from typing import Tuple
import numpy as np

def bbox_from_poly(poly: np.ndarray):
    xs = poly[:,0]; ys = poly[:,1]
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    return [float(x0), float(y0), float(x1-x0), float(y1-y0)]
```

---

## 🖼️ previews.py

```python
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
```

---

## 🚀 analyze.py (CLI)

```python
import json, os
from pathlib import Path
import click
from PIL import Image
import numpy as np

from config import SEG_CFG, VEC_CFG, ADDR_CFG, VAL_CFG
from sammy.validators import validate_input
from sammy.segmenter import GroundedSAMPredictor, DetInstance
from sammy.vectorizer import mask_to_polygons, polygon_to_points_attr
from sammy.addressing import PolyInstance, assign_ids, address_for
from sammy.svg_writer import build_svg, SVGItem
from sammy.schema import AnalysisJSON, ElementJSON
from sammy.utils import bbox_from_poly

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--outdir', default='out', help='Output directory')
@click.option('--device', default=SEG_CFG.device)
@click.option('--prompts', multiple=True, help='Override text prompts (repeatable)')
def main(image_path, outdir, device, prompts):
    ip = Path(image_path)
    os.makedirs(outdir, exist_ok=True)

    # 0) Validate
    rep = validate_input(ip, VAL_CFG.min_width, VAL_CFG.min_height, VAL_CFG.allowed_exts, device)
    if not rep.ok:
        raise SystemExit(f"Validation failed: {rep.reason}")

    # 1) Segment
    seg = GroundedSAMPredictor(device=device, box_thr=SEG_CFG.box_threshold, text_thr=SEG_CFG.text_threshold, max_side=SEG_CFG.max_side)
    cats = list(prompts) if prompts else SEG_CFG.text_prompts
    detections = seg.predict(str(ip), cats)  # List[DetInstance]

    # 2) Vectorize each mask → polygons
    poly_items = []
    for det in detections:
        polys = mask_to_polygons(det.mask, simplify_eps=VEC_CFG.simplify_epsilon, min_area=VEC_CFG.min_area)
        for poly in polys:
            poly_items.append(PolyInstance(det.category, det.score, poly))

    # 3) Assign IDs + addresses
    id_pairs = assign_ids(poly_items)
    svg_items = []
    json_elems = []
    for id_str, pi in id_pairs:
        cat = pi.category.replace(' ', '_')
        idx = int(id_str.split('-')[-1])
        addr = address_for(cat, idx)
        pts_attr = polygon_to_points_attr(pi.polygon)
        svg_items.append(SVGItem(id=id_str, category=cat, address=addr, points=pts_attr, score=pi.score))
        json_elems.append(ElementJSON(
            id=id_str,
            address=addr,
            category=cat,
            polygon=pi.polygon.astype(float).tolist(),
            bbox=bbox_from_poly(pi.polygon),
            confidence=float(pi.score),
            attributes={},
        ))

    # 4) SVG + JSON
    with Image.open(ip) as im:
        w, h = im.size
    svg = build_svg(w, h, svg_items, origin_name=ip.name)
    (Path(outdir)/f"{ip.stem}.svg").write_text(svg)

    aj = AnalysisJSON(image={"width": w, "height": h, "dpi": 96, "file": ip.name}, elements=json_elems, relations=[])
    (Path(outdir)/f"{ip.stem}.json").write_text(json.dumps(aj.to_dict(), indent=2))

    print(f"✅ Wrote: {ip.stem}.svg and {ip.stem}.json in {outdir}")

if __name__ == '__main__':
    main()
```

---

## 🌐 api.py (FastAPI service)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from pathlib import Path
import uvicorn
import base64

from analyze import main as analyze_cli  # Or refactor analyze logic into callable

app = FastAPI(title="Sammy Vision API", version="0.1")

@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    # Call into pipeline (refactor analyze.py into functions for reuse)
    # For brevity, we shell out to CLI or import function
    try:
        # TODO: replace with direct call and return in-memory SVG/JSON
        outdir = tmp_path.parent / "out"
        outdir.mkdir(exist_ok=True)
        # analyze_image(tmp_path, outdir)
        return JSONResponse({"message": "OK", "note": "Wire analyze_image() here."})
    finally:
        pass

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

---

## 🧪 tests/test\_smoke.py

```python
from pathlib import Path
import subprocess

def test_cli_smoke(tmp_path: Path):
    img = Path('samples/house1.jpg')
    if not img.exists():
        return
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    cp = subprocess.run(['python', 'analyze.py', str(img), '--outdir', str(out)], capture_output=True)
    assert cp.returncode == 0
    svg = out / f"{img.stem}.svg"
    js = out / f"{img.stem}.json"
    assert svg.exists() and js.exists()
```

---

## ▶️ How to Run (POC)

1. **Create venv & install deps**

```bash
python -m venv .venv && source .venv/bin/activate
uv pip install --upgrade pip
uv sync  # or: uv add -r requirements.txt
```

2. **(Intel mac)** Ensure ABI-safe versions:

```bash
uv add "numpy<2" "torch==2.2.*" "torchvision==0.17.*"
```

3. **Run CLI**

```bash
python analyze.py path/to/house.jpg --outdir out/
```

4. **Run API**

```bash
python api.py
# Test JSON
curl -s -X POST "http://localhost:8000/analyze?return_format=json&device=cpu" \
  -F "file=@/absolute/path/to/house.jpg" | jq .
# Test raw SVG
curl -X POST "http://localhost:8000/analyze?return_format=svg&device=cpu" \
  -H "Accept: image/svg+xml" \
  -F "file=@/absolute/path/to/house.jpg"
```

> Notes:
>
> * First run will download Grounding DINO and SAM weights from Hugging Face.
> * CPU is fine for testing (SAM **vit\_b**). For speed/quality on GPU, set `device=cuda` and/or switch `sam_variant` to `vit_h` in `segmenter.py` constructor.

2. **Place model weights** (GroundingDINO + SAM) and wire them in `segmenter.py`.
3. **Run CLI**

```bash
python analyze.py path/to/house.jpg --outdir out/
```

4. Outputs: `out/house.svg`, `out/house.json`

---

## 🔁 Output Contract (example JSON)

```json
{
  "image": {"width": 2048, "height": 1365, "dpi": 96, "file": "house.jpg"},
  "elements": [
    {
      "id": "window-0",
      "address": "house/window/0",
      "category": "window",
      "polygon": [[310.5,420.1],[402.0,420.3],[403.1,505.0],[309.8,504.2]],
      "bbox": [309.8,420.1,93.3,84.9],
      "confidence": 0.92,
      "attributes": {"pane_count": 6}
    }
  ],
  "relations": [],
  "version": "0.1.0"
}
```

---

## 🧭 Next Steps

* Wire real GroundingDINO + SAM inference in `segmenter.py`.
* Add **outline/labeled previews** (PNG) for your compare-toggle.
* Optional: add **CLIP-based attribute classifier** for window styles & roof types.
* Add **stable-ID seeding** from image hash to improve determinism across runs.
* Integrate a **materials taxonomy** (`data-material`) aligned with your editor sidebar.
