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
    rep = validate_input(ip, VAL_CFG.min_width, VAL_CFG.min_height, VAL_CFG.allowed_exts, device, VAL_CFG.enable_house_classifier)
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

# --- ADD THIS TO analyze.py (above the __main__ guard) ---
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image

from config import SEG_CFG, VEC_CFG, ADDR_CFG, VAL_CFG
from sammy.validators import validate_input
from sammy.segmenter import GroundedSAMPredictor
from sammy.vectorizer import mask_to_polygons, polygon_to_points_attr
from sammy.addressing import PolyInstance, assign_ids, address_for
from sammy.svg_writer import build_svg, SVGItem
from sammy.schema import AnalysisJSON, ElementJSON
from sammy.utils import bbox_from_poly

def analyze_image_to_memory(
    image_path: str,
    prompts: Optional[List[str]] = None,
    device: str = SEG_CFG.device
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the pipeline and return (svg_string, analysis_json_dict)
    """
    ip = Path(image_path)

    # 0) Validate
    rep = validate_input(ip, VAL_CFG.min_width, VAL_CFG.min_height, VAL_CFG.allowed_exts, device)
    if not rep.ok:
        raise ValueError(f"Validation failed: {rep.reason}")

    # 1) Segment (GroundingDINO → boxes, SAM → masks)
    seg = GroundedSAMPredictor(
        device=device,
        box_thr=SEG_CFG.box_threshold,
        text_thr=SEG_CFG.text_threshold,
        max_side=SEG_CFG.max_side
    )
    cats = list(prompts) if prompts else SEG_CFG.text_prompts
    detections = seg.predict(str(ip), cats)  # List[DetInstance]

    # 2) Vectorize each mask → polygons
    poly_items: List[PolyInstance] = []
    for det in detections:
        polys = mask_to_polygons(det.mask, simplify_eps=VEC_CFG.simplify_epsilon, min_area=VEC_CFG.min_area)
        for poly in polys:
            poly_items.append(PolyInstance(det.category, det.score, poly))

    # 3) Assign stable IDs + addresses
    id_pairs = assign_ids(poly_items)

    svg_items: List[SVGItem] = []
    json_elems: List[ElementJSON] = []
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

    # 4) Build SVG + JSON dict
    with Image.open(ip) as im:
        w, h = im.size
    svg_str = build_svg(w, h, svg_items, origin_name=ip.name)
    aj = AnalysisJSON(
        image={"width": w, "height": h, "dpi": 96, "file": ip.name},
        elements=json_elems,
        relations=[]
    )
    return svg_str, aj.to_dict()


if __name__ == '__main__':
    main()