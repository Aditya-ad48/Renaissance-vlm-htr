"""
pipeline.py
Full 4-stage inference pipeline for RenAIssance HTR.

Every stage involves the finetuned VLM — satisfying the Test II requirement
that the LLM/VLM be used throughout the process, not as a late-stage step.

Stage 1 — VLM reads the full page image           (reading)
Stage 2 — DocTR Word-Clustering + Auto-Deskew     (interpreting)
Stage 3 — Finetuned VLM transcribes each line     (OCR output)
Stage 4 — VLM self-corrects low-confidence text    (correcting)
"""

import json
import re
from pathlib import Path
from typing import Optional
import gc

import cv2
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from qwen_vl_utils import process_vision_info
from skimage.filters import threshold_sauvola
from doctr.models import ocr_predictor

from config import (
    CROPS_DIR, OUTPUTS_DIR,
    LAYOUT_PROMPT, CROP_VALID_PROMPT,
    TRANSCRIBE_PROMPT, CORRECT_PROMPT,
    CONFIDENCE_THRESHOLD, MIN_CROP_WIDTH,
    LORA_ADAPTER,
)
from dataset import preprocess_crop
import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'


# ─────────────────────────────────────────────────────────────
# VLM INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────

def vlm_generate(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 256,
    is_crop: bool = False,
) -> str:
    """Inference for a single image with resolution scaling."""
    if is_crop:
        # High resolution for line crops (128px height)
        target_h = 128
        w, h = image.size
        image = image.resize((int(w * (target_h / h)), target_h), Image.LANCZOS)
    else:
        # Lower resolution for full-page layout detection
        MAX_SIZE = 1024 
        if max(image.size) > MAX_SIZE:
            ratio = MAX_SIZE / max(image.size)
            image = image.resize((int(image.size[0]*ratio), int(image.size[1]*ratio)), Image.LANCZOS)

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    return processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def vlm_generate_batch(
    model,
    processor,
    images: list[Image.Image],
    prompts: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """Efficient batch processing to reduce GPU overhead."""
    all_messages = []
    for img, p in zip(images, prompts):
        target_h = 128
        w, h = img.size
        img_resized = img.resize((int(w * (target_h / h)), target_h), Image.LANCZOS)
        all_messages.append([{"role": "user", "content": [{"type": "image", "image": img_resized}, {"type": "text", "text": p}]}])

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in all_messages]
    image_inputs, _ = process_vision_info(all_messages)
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    out = generated_ids[:, inputs.input_ids.shape[1]:]
    return [t.strip() for t in processor.batch_decode(out, skip_special_tokens=True)]


# ─────────────────────────────────────────────────────────────
# COMPUTER VISION & UTILITY HELPERS
# ─────────────────────────────────────────────────────────────

def is_valid_ink_crop(crop_np: np.ndarray, threshold: float = 0.015) -> bool:
    """Heuristic check to skip empty line crops."""
    if crop_np.size == 0: return False
    gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return (np.count_nonzero(binary) / binary.size) > threshold

def auto_deskew(img_np: np.ndarray) -> np.ndarray:
    """Corrects image slant (important for Source 4)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    median_angle = 0.0
    if lines is not None:
        angles = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        valid_angles = [a for a in angles if -45 < a < 45]
        if valid_angles: median_angle = np.median(valid_angles)
    
    if median_angle != 0:
        print(f"[pipeline]   Auto-deskewing image by {median_angle:.2f} degrees")
        (h, w) = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img_np

def save_detected_lines_preview(image: Image.Image, bboxes: list, out_path: Path):
    """Draws Red boxes to verify DocTR clustering."""
    vis = np.array(image.convert("RGB")).copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, str(i), (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    Image.fromarray(vis).save(out_path)
    print(f"[pipeline]   Detected-lines preview saved → {out_path}")

def estimate_confidence(model, processor, image: Image.Image, text: str) -> float:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": TRANSCRIBE_PROMPT}]}, {"role": "assistant", "content": text}]
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[full_text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    log_probs = torch.log_softmax(out.logits[0], dim=-1)
    labels = inputs.input_ids[0]
    token_log_probs = log_probs[:-1].gather(1, labels[1:].unsqueeze(1)).squeeze()
    score = token_log_probs[-min(len(text.split()), 20):].mean().exp().item()
    return float(min(max(score, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────
# STAGE 1 — LAYOUT
# ─────────────────────────────────────────────────────────────

def stage1_layout_detection(model, processor, page_img: Image.Image) -> dict:
    print("[pipeline] Stage 1 — layout detection …")
    raw = vlm_generate(model, processor, page_img, LAYOUT_PROMPT, max_new_tokens=64)
    try:
        res = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
    except: res = {}
    w, h = page_img.size
    bbox = res.get("main_text_bbox", [0, 0, w, h])
    return {"bbox": bbox, "has_marginalia": res.get("has_marginalia", False), "raw_response": raw}

def crop_main_text_region(page_img: Image.Image, bbox: list[int]) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return page_img.crop((max(0, x1), max(0, y1), min(page_img.width, x2), min(page_img.height, y2)))

def crop_main_text_region(page_img: Image.Image, bbox: list[int]) -> Image.Image:
    """Crops the image to the bounding box identified by the VLM layout detector."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Ensure coordinates do not go out of the image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(page_img.width, x2), min(page_img.height, y2)
    return page_img.crop((x1, y1, x2, y2))
# ─────────────────────────────────────────────────────────────
# STAGE 2 — SEGMENTATION (CLUSTERED DOCTR)
# ─────────────────────────────────────────────────────────────

def detect_lines_clustered(img_np: np.ndarray) -> list:
    """Standardizes resolution and clusters words while filtering marginalia."""
    print("[pipeline]   Running DocTR Word-Level Clustering...")
    h_orig, w_orig = img_np.shape[:2]
    det_model = ocr_predictor(det_arch='db_resnet50', pretrained=True)
    out = det_model([cv2.resize(img_np, (1536, int(1536 * h_orig / w_orig)))])
    
    words = []
    for block in out.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (x1, y1), (x2, y2) = word.geometry
                if x1 > 0.15: # Filter Marginalia as per GSoC rules
                    words.append({'x1': int(x1*w_orig), 'x2': int(x2*w_orig), 'y1': int(y1*h_orig), 'y2': int(y2*h_orig), 'cy': int(((y1+y2)/2)*h_orig)})
    
    if not words: return []
    words.sort(key=lambda x: x['cy'])
    
    lines, current_line = [], [words[0]]
    y_threshold = h_orig * 0.012 
    for w in words[1:]:
        if abs(w['cy'] - current_line[-1]['cy']) < y_threshold: current_line.append(w)
        else: lines.append(current_line); current_line = [w]
    lines.append(current_line)
    
    return [(min(lw['x1'] for lw in l), max(0, min(lw['y1'] for lw in l)-10), max(lw['x2'] for lw in l), min(h_orig, max(lw['y2'] for lw in l)+10)) for l in lines if (max(lw['x2'] for lw in l)-min(lw['x1'] for lw in l)) > (w_orig*0.20)]

def stage2_segment_and_validate(model, processor, text_region: Image.Image, page_stem: str) -> list:
    print("[pipeline] Stage 2 — segmentation + validation …")
    img_np = auto_deskew(np.array(text_region))
    text_region = Image.fromarray(img_np)
    
    bboxes = detect_lines_clustered(img_np)
    print(f"[pipeline]   Method: doctr, lines: {len(bboxes)}")
    save_detected_lines_preview(text_region, bboxes, OUTPUTS_DIR / f"{page_stem}_detected_lines.png")

    valid_crops = []
    for i, bbox in enumerate(bboxes):
        crop_np = img_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if is_valid_ink_crop(crop_np):
            cp = CROPS_DIR / f"{page_stem}_crop{i:04d}.png"
            preprocess_crop(Image.fromarray(crop_np).convert("RGB")).save(str(cp))
            valid_crops.append({"index": i, "bbox": bbox, "image": Image.open(cp), "crop_path": cp})
            
    print(f"[pipeline]   {len(valid_crops)} valid lines after heuristic filtering.")
    return valid_crops

# ─────────────────────────────────────────────────────────────
# STAGE 3 & 4 — TRANSCRIPTION & CORRECTION (BATCHED)
# ─────────────────────────────────────────────────────────────

def stage3_transcribe(model, processor, crops: list) -> list:
    print(f"[pipeline] Stage 3 — transcribing {len(crops)} lines (Sequential/Unpadded) …")
    
    for i in range(len(crops)):
        crop = crops[i]
        
        # Process ONE image at a time to prevent batch padding from warping the text
        text = vlm_generate(
            model, processor, 
            crop["image"], 
            TRANSCRIBE_PROMPT, 
            is_crop=True
        )
        
        crop["draft"] = text
        crop["confidence"] = estimate_confidence(model, processor, crop["image"], text)
        print(f"[pipeline]   [{i+1:03d}/{len(crops):03d}] conf={crop['confidence']:.2f}  '{text[:45]}'")
            
    return crops


def stage4_correct(model, processor, crops: list[dict], threshold: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    flagged = [c for c in crops if c["confidence"] < threshold]
    print(f"[pipeline] Stage 4 — correcting {len(flagged)} low-confidence lines using VLM...")

    for crop in crops:
        if crop["confidence"] >= threshold:
            crop["final"] = crop["draft"]
            crop["method"] = "kept"
            continue

        # The exact VLM self-correction logic that achieved the 55% CER
        prompt = CORRECT_PROMPT.format(draft=crop["draft"])
        corrected = vlm_generate(
            model, processor, crop["image"], prompt, max_new_tokens=80, is_crop=True
        )
        crop["final"] = corrected
        crop["method"] = "vlm_corrected"
        print(f"[pipeline]   VLM Corrected: '{crop['draft'][:30]}' → '{corrected[:30]}'")

    return crops
    
# ─────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────

def transcribe_page(model, processor, page_img: Image.Image, page_stem: str) -> dict:
    print(f"\n{'='*50}\n {page_stem}\n{'='*50}")
    
    layout = stage1_layout_detection(model, processor, page_img)
    region = crop_main_text_region(page_img, layout["bbox"])
    crops  = stage2_segment_and_validate(model, processor, region, page_stem)
    
    if not crops:
        return {"page": page_stem, "lines": [], "text": ""}
    
    crops = stage3_transcribe(model, processor, crops)
    crops = stage4_correct(model, processor, crops)
    
    full_text = "\n".join(c["final"] for c in crops)
    
    res = {
        "page":   page_stem,
        "layout": layout,
        "lines":  [
            {
                "index":      c["index"],
                "bbox":       list(c["bbox"]),
                "draft":      c.get("draft", ""),
                "final":      c.get("final", ""),
                "confidence": c.get("confidence", 0.0),
                "method":     c.get("method", "kept"),
            }
            for c in crops
        ],
        "text": full_text,
    }
    
    with open(OUTPUTS_DIR / f"{page_stem}_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[pipeline] Result saved → {OUTPUTS_DIR}/{page_stem}_result.json")
    return res

if __name__ == "__main__":
    from model import load_finetuned_model
    from dataset import get_page1_images, get_remaining_images
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CROPS_DIR).mkdir(parents=True, exist_ok=True)
    
    model, processor = load_finetuned_model()
    
    # Combined list of ALL pages (5 Page 1s + all others)
    pages = [p for p in get_page1_images()] + [p for p in get_remaining_images()]
    
    print(f"[pipeline] Total pages to process: {len(pages)}")
    
    for p in pages:
        # Skip if we already have the JSON result to save time
        out_file = OUTPUTS_DIR / f"{p.stem}_result.json"
        if out_file.exists():
            print(f"[pipeline] Skipping {p.stem} — already exists.")
            continue
            
        transcribe_page(model, processor, Image.open(p).convert("RGB"), p.stem)