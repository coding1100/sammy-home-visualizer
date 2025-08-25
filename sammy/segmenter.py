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
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base",
            use_safetensors=True   # ← avoids .bin
        ).to(self.device)

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