"""
dataset.py — RenAIssance HTR

Rodrigo dataset format :
  images/Rodrigo_00006_00.png        ← line image
  text/transcriptions.txt            ← single file, format:
      Rodrigo_00008_13 pre la su gracia. Señor ymbio vos la obra...
      Rodrigo_00008_14 copilar de las historias Antiguas de los Reyes...
  partition/train.txt                ← list of stems for train split
  partition/test.txt
  partition/validation.txt

RenAIssance GT format (confirmed from actual .docx files):
  NOTES block (paleographer rules) → strip, inject into prompts
  PDF p1 marker
  Each paragraph = one transcription line (~30 lines per source)
  GT exists for PAGE 1 ONLY across all 5 sources (~150 lines total)
"""

import re
import random
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset, Subset
import fitz  # PyMuPDF

from config import (
    RAW_DIR, IMAGES_DIR, CROPS_DIR, GT_DIR, RODRIGO_DIR,
    PAGE_DPI, MIN_CROP_WIDTH, MAX_CROP_WIDTH, CROP_HEIGHT,
    TRANSCRIBE_PROMPT, SEED,
)

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# RODRIGO LOADER  — confirmed format
# ─────────────────────────────────────────────────────────────

def load_rodrigo(rodrigo_dir: Path) -> list[tuple[Path, str]]:
    """
    Load Rodrigo dataset using their official train split only.

    Format:
      images/Rodrigo_XXXXX_YY.png
      text/transcriptions.txt  →  "Rodrigo_XXXXX_YY transcription text"
      partition/train.txt      →  list of stems in train split

    Returns list of (image_path, transcription_text) tuples.
    Only includes stems listed in partition/train.txt to avoid
    leaking their test/validation data into our finetuning.
    """
    rodrigo_dir = Path(rodrigo_dir)
    images_dir  = rodrigo_dir / "images"
    trans_file  = rodrigo_dir / "text" / "transcriptions.txt"
    train_file  = rodrigo_dir / "partition" / "train.txt"

    # validate structure
    for path in [images_dir, trans_file, train_file]:
        if not path.exists():
            print(f"[dataset] WARNING: Rodrigo path not found: {path}")
            return []

    # load official train split stems
    train_stems = set()
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            stem = line.strip()
            if stem:
                train_stems.add(stem)
    print(f"[dataset] Rodrigo train split: {len(train_stems)} stems")

    # parse transcriptions.txt
    # format: "{stem} {transcription text}"
    stem_to_text = {}
    with open(trans_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)   # split on FIRST space only
            if len(parts) == 2:
                stem_to_text[parts[0]] = parts[1]

    print(f"[dataset] Rodrigo transcriptions loaded: {len(stem_to_text)}")

    # match train stems → image paths → transcriptions
    samples = []
    missing_images = 0
    for stem in sorted(train_stems):
        if stem not in stem_to_text:
            continue

        # try both .png and .tif extensions
        img_path = images_dir / f"{stem}.png"
        if not img_path.exists():
            img_path = images_dir / f"{stem}.tif"
        if not img_path.exists():
            missing_images += 1
            continue

        text = stem_to_text[stem].strip()
        if text:
            samples.append((img_path, text))

    print(f"[dataset] Rodrigo: {len(samples)} train samples loaded. "
          f"({missing_images} images missing — normal if small)")
    return samples


# ─────────────────────────────────────────────────────────────
# GT PARSER  — RenAIssance .docx format
# ─────────────────────────────────────────────────────────────

def extract_docx_text(docx_path: Path) -> str:
    """Extract plain text from .docx using pandoc."""
    result = subprocess.run(
        ["pandoc", str(docx_path), "-t", "plain"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pandoc failed on {docx_path}.\n"
            "Install pandoc: brew install pandoc (Mac) or "
            "apt install pandoc (Linux/Kaggle)"
        )
    return result.stdout


def parse_gt_docx(docx_path: Path) -> dict:
    """
    Parse one RenAIssance GT .docx file.

    Returns:
      {
        "source": str,
        "notes":  list[str],   — paleographer rules (injected into prompts)
        "pages":  {1: [line1, line2, …]}   — page 1 only has GT
      }
    """
    raw   = extract_docx_text(docx_path)
    lines = raw.split("\n")

    notes    = []
    pages    = {}
    cur_page = None
    in_notes = True

    for raw_line in lines:
        line = raw_line.strip()

        # page marker: "PDF p1", "PDF p2", "PDF  p1" etc.
        m = re.match(r"^PDF\s+p(\d+)$", line, re.IGNORECASE)
        if m:
            in_notes = False
            cur_page = int(m.group(1))
            pages[cur_page] = []
            continue

        if in_notes:
            clean = re.sub(
                r"^NOTES:\s*", "", line, flags=re.IGNORECASE
            ).strip()
            if clean:
                notes.append(clean)
            continue

        # transcription line
        if cur_page is not None and line:
            pages[cur_page].append(line)

    total = sum(len(v) for v in pages.values())
    print(f"[dataset]  {docx_path.name}: "
          f"{len(pages)} page(s) GT, {total} lines, "
          f"{len(notes)} paleographer notes")
    return {"notes": notes, "pages": pages}


def load_all_gt(gt_dir: Path = GT_DIR) -> dict:
    """
    Load all GT .docx files from gt_dir.
    Returns {source_stem: parsed_dict}
    """
    gt_map     = {}
    docx_files = list(gt_dir.glob("*.docx"))

    if not docx_files:
        print(f"[dataset] WARNING: No .docx GT files in {gt_dir}")
        print("          Place the 5 RenAIssance GT .docx files there.")
        return gt_map

    for f in docx_files:
        # strip _transcription suffix if present
        stem = re.sub(r"_transcription$", "", f.stem, flags=re.IGNORECASE)
        parsed = parse_gt_docx(f)
        parsed["source"] = stem
        gt_map[stem] = parsed

    print(f"[dataset] GT loaded for {len(gt_map)} source(s).")
    return gt_map


def build_line_gt_map(gt_map: dict) -> dict:
    """
    Flatten GT into line-level lookup.

    Key:   "{source_stem}_p{page}_l{line_idx:04d}"
    Value: transcription string

    Example:
      "AHPG-GPAH_1_1716_A_35___1744_p1_l0003"
      → "soy hijo Leg.mo de Domingo de Muguruza y Antonia"
    """
    flat = {}
    for source_stem, parsed in gt_map.items():
        for page_num, lines in parsed["pages"].items():
            for idx, text in enumerate(lines):
                key = f"{source_stem}_p{page_num}_l{idx:04d}"
                flat[key] = text
    return flat


# ─────────────────────────────────────────────────────────────
# PDF → PAGE IMAGES
# ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: Path, out_dir: Path,
                  dpi: int = PAGE_DPI) -> list[Path]:
    """Convert every page of a PDF to a high-res PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc   = fitz.open(str(pdf_path))
    scale = dpi / 72.0
    mat   = fitz.Matrix(scale, scale)
    saved = []
    for i, page in enumerate(doc):
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        name = out_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
        pix.save(str(name))
        saved.append(name)
    doc.close()
    return saved


def convert_all_pdfs(raw_dir: Path = RAW_DIR,
                     images_dir: Path = IMAGES_DIR):
    """Convert every PDF in raw_dir to page images. Safe to re-run."""
    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[dataset] No PDFs in {raw_dir}.")
        print("          Place the 5 handwritten RenAIssance PDFs there.")
        return
    for pdf in pdfs:
        dest = images_dir / pdf.stem
        if dest.exists() and any(dest.glob("*.png")):
            print(f"[dataset] Skip {pdf.name} — already converted.")
            continue
        print(f"[dataset] Converting {pdf.name} …")
        paths = pdf_to_images(pdf, dest)
        print(f"[dataset]   → {len(paths)} pages saved.")


def get_page1_images(images_dir: Path = IMAGES_DIR) -> list[Path]:
    """Page 1 images only — these have GT for quantitative eval."""
    return sorted(images_dir.glob("**/*_p001.png"))


def get_remaining_images(images_dir: Path = IMAGES_DIR) -> list[Path]:
    """Pages 2+ — no GT, used for qualitative inference in notebook."""
    all_pages = sorted(images_dir.glob("**/*.png"))
    return [p for p in all_pages if not p.name.endswith("_p001.png")]


# ─────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING + AUGMENTATION
# ─────────────────────────────────────────────────────────────

def preprocess_crop(pil_img: Image.Image,
                    target_height: int = CROP_HEIGHT) -> Image.Image:
    """
    Normalise a line crop:
      grayscale → CLAHE adaptive contrast → RGB → resize to fixed height
    CLAHE helps with uneven ink density common in old manuscripts.
    """
    gray  = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    rgb   = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img   = Image.fromarray(rgb)
    w, h  = img.size
    ratio = target_height / h
    new_w = max(int(w * ratio), MIN_CROP_WIDTH)
    return img.resize((new_w, target_height), Image.LANCZOS)


def augment_crop(pil_img: Image.Image) -> Image.Image:
    """Subtle augmentation — rotation, brightness, contrast, mild blur."""
    pil_img = pil_img.rotate(
        random.uniform(-2.0, 2.0), fillcolor=255, expand=False)
    pil_img = ImageEnhance.Brightness(pil_img).enhance(
        random.uniform(0.85, 1.15))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(
        random.uniform(0.85, 1.15))
    if random.random() < 0.2:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return pil_img


# ─────────────────────────────────────────────────────────────
# LINE CROP EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_line_crops(image_path: Path, crops_dir: Path,
                       bboxes: list[tuple]) -> list[Path]:
    """Save cropped line images from bounding boxes."""
    img  = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    crops_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cw = x2 - x1
        if cw < MIN_CROP_WIDTH or cw > MAX_CROP_WIDTH:
            continue
        crop  = img[y1:y2, x1:x2]
        fpath = crops_dir / f"{image_path.stem}_crop{i:04d}.png"
        cv2.imwrite(str(fpath), crop)
        saved.append(fpath)
    return saved


# ─────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────

class HandwrittenLineDataset(Dataset):
    """
    Combined dataset for VLM finetuning.

    Sources:
      1. RenAIssance GT crops  (~150 lines, target domain)
      2. Rodrigo train split   (~thousands of lines, same language)

    Without Rodrigo, 150 lines is far too small to finetune.
    """

    def __init__(self,
                 crops_dir: Path = CROPS_DIR,
                 gt_map: Optional[dict] = None,
                 rodrigo_dir: Optional[Path] = None,
                 augment: bool = False):
        self.augment      = augment
        self.samples      = []   # (Path, str)
        self.source_flags = []   # "renaissance" | "rodrigo"

        # ── RenAIssance GT crops (page 1 only) ──────────────
        flat_gt = gt_map if gt_map is not None else \
                  build_line_gt_map(load_all_gt())

        matched = 0
        for crop_path in sorted(crops_dir.glob("**/*.png")):
            if crop_path.stem in flat_gt:
                self.samples.append((crop_path, flat_gt[crop_path.stem]))
                self.source_flags.append("renaissance")
                matched += 1
        print(f"[dataset] RenAIssance GT crops matched: {matched}")

        # ── Rodrigo official train split ─────────────────────
        rod_dir = Path(rodrigo_dir) if rodrigo_dir else RODRIGO_DIR
        rod_samples = load_rodrigo(rod_dir)
        for img_path, text in rod_samples:
            self.samples.append((img_path, text))
            self.source_flags.append("rodrigo")

        print(f"[dataset] Total samples: {len(self.samples)}")

        if len(self.samples) == 0:
            print("\n[dataset] NOTHING LOADED. Checklist:")
            print("  1. PDFs in data/raw/ ?")
            print("  2. GT .docx files in data/ground_truth/ ?")
            print("  3. Ran convert_all_pdfs() ?")
            print("  4. Ran pipeline Stage 1+2 to get line crops ?")
            print("  5. Rodrigo unzipped in data/rodrigo/ ?")
            print("     Expected: data/rodrigo/images/, "
                  "data/rodrigo/text/transcriptions.txt, "
                  "data/rodrigo/partition/train.txt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        crop_path, gt_text = self.samples[idx]
        img = Image.open(crop_path).convert("RGB")
        img = preprocess_crop(img)
        if self.augment:
            img = augment_crop(img)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text":  TRANSCRIBE_PROMPT},
                ],
            },
            {"role": "assistant", "content": gt_text},
        ]
        return {"image": img, "text": gt_text, "messages": messages}


# ─────────────────────────────────────────────────────────────
# TRAIN / VAL SPLIT
# ─────────────────────────────────────────────────────────────

def split_dataset(dataset: HandwrittenLineDataset,
                  val_ratio: float = 0.15,
                  seed: int = SEED):
    """
    TRAIN = all Rodrigo + 85% of RenAIssance GT
    VAL   = 15% of RenAIssance GT only

    Val is RenAIssance-only — the target domain.
    Rodrigo in val would give falsely optimistic loss numbers.
    """
    ren_idx = [i for i, f in enumerate(dataset.source_flags)
               if f == "renaissance"]
    rod_idx = [i for i, f in enumerate(dataset.source_flags)
               if f == "rodrigo"]

    random.seed(seed)
    random.shuffle(ren_idx)

    val_size  = max(1, int(len(ren_idx) * val_ratio))
    val_idx   = ren_idx[:val_size]
    train_idx = ren_idx[val_size:] + rod_idx

    print(f"[dataset] Train: {len(train_idx)} "
          f"({len(ren_idx)-val_size} renaissance + {len(rod_idx)} rodrigo)")
    print(f"[dataset] Val:   {len(val_idx)} (renaissance only)")

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ─────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────

def get_page1_gt_pairs(gt_map: dict, results_dir: Path) -> tuple:
    """
    Match pipeline output JSONs to GT for quantitative evaluation.
    Only page 1 lines that have both a result AND GT are included.
    Returns (predictions, references).
    """
    import json
    flat_gt = build_line_gt_map(gt_map)
    predictions, references = [], []

    for result_json in sorted(results_dir.glob("*_result.json")):
        with open(result_json, encoding="utf-8") as f:
            result = json.load(f)

        page_stem = result.get("page", "")
        # only evaluate page 1 results (have GT)
        if "_p001" not in page_stem:
            continue

        source = re.sub(r"_p\d+$", "", page_stem)
        for line in result.get("lines", []):
            key = f"{source}_p1_l{line['index']:04d}"
            if key in flat_gt:
                predictions.append(line["final"])
                references.append(flat_gt[key])

    print(f"[dataset] Quantitative eval pairs: {len(predictions)}")
    return predictions, references


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(" dataset.py smoke test")
    print("=" * 55)

    # 1. Rodrigo loader
    print("\n[1] Testing Rodrigo loader …")
    rod = load_rodrigo(RODRIGO_DIR)
    if rod:
        print(f"  Loaded {len(rod)} train samples")
        img_path, text = rod[0]
        print(f"  Sample image : {img_path.name}")
        print(f"  Sample text  : {text[:60]}")
    else:
        print("  Rodrigo not found — check RODRIGO_DIR in config.py")

    # 2. GT parser
    print("\n[2] Testing GT parser …")
    gt_map = load_all_gt()
    if gt_map:
        for source, parsed in gt_map.items():
            print(f"  Source : {source}")
            print(f"  Notes  : {len(parsed['notes'])}")
            for n in parsed["notes"][:3]:
                print(f"           • {n}")
            for pg, lines in parsed["pages"].items():
                print(f"  Page {pg} : {len(lines)} GT lines")
                for ln in lines[:2]:
                    print(f"           {ln[:65]}")

    # 3. Flat GT map
    flat = build_line_gt_map(gt_map)
    print(f"\n[3] Flat GT map: {len(flat)} line entries")
    for k, v in list(flat.items())[:3]:
        print(f"  {k}  →  {v[:50]}")

    # 4. PDF conversion
    print("\n[4] Converting PDFs …")
    convert_all_pdfs()

    p1  = get_page1_images()
    p2p = get_remaining_images()
    print(f"  Page 1 images (GT exists)  : {len(p1)}")
    print(f"  Pages 2+ images (qualitative): {len(p2p)}")

    print("\ndataset.py OK")