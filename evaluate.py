"""
evaluate.py
Evaluation of the HTR pipeline output against ground truth transcriptions.

Metrics used and WHY each one matters for this specific task:

  CER  (Character Error Rate)    — PRIMARY metric. HTR is character-level;
                                    a single wrong letter in a word matters.
  WER  (Word Error Rate)         — Word-level, important for readability.
  NED  (Normalised Edit Distance)— Handles partial matches gracefully.
  BERTScore                      — Semantic similarity. CRITICAL for historical
                                    Spanish where archaic spellings like 'vna',
                                    'fazer' are CORRECT, not errors. Penalising
                                    them as wrong would misrepresent accuracy.
  BLEU                           — Sequence-level fluency.

Usage:
    from evaluate import evaluate_results
    scores = evaluate_results(predicted_lines, ground_truth_lines)
"""

import json
import difflib
from pathlib import Path
from typing import Optional

import numpy as np
from jiwer import wer, cer                    
from bert_score import score as bert_score    
from nltk.translate.bleu_score import (      
    corpus_bleu, SmoothingFunction
)
import nltk
nltk.download("punkt", quiet=True)

from config import OUTPUTS_DIR, BERTSCORE_MODEL, BERTSCORE_LANG


# ─────────────────────────────────────────────────────────────
# TEXT NORMALISATION
# ─────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """
    Light normalisation before metric computation.
    We do NOT modernise spelling — that would change the target.
    We only strip extra whitespace and lowercase.
    """
    return " ".join(text.lower().split())


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL METRICS
# ─────────────────────────────────────────────────────────────

def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Character Error Rate = edit_distance(pred, ref) / len(ref)
    Lower is better. 0.0 = perfect. Reported as percentage.
    """
    preds = [normalise(p) for p in predictions]
    refs  = [normalise(r) for r in references]
    # Handle empty lists gracefully
    if not preds or not refs: return 100.0
    return cer(refs, preds) * 100


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """
    Word Error Rate. Lower is better. Reported as percentage.
    """
    preds = [normalise(p) for p in predictions]
    refs  = [normalise(r) for r in references]
    if not preds or not refs: return 100.0
    return wer(refs, preds) * 100


def compute_ned(predictions: list[str], references: list[str]) -> float:
    """
    Normalised Edit Distance = edit_distance / max(len(pred), len(ref))
    Averaged over all pairs. Lower is better.
    """
    def ned_single(pred: str, ref: str) -> float:
        p, r = normalise(pred), normalise(ref)
        if not p and not r:
            return 0.0
        d = edit_distance(p, r)
        return d / max(len(p), len(r), 1)

    scores = [ned_single(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(scores)) * 100 if scores else 100.0


def edit_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance at character level."""
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return int(dp[m][n])


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = BERTSCORE_LANG,
    model_type: str = BERTSCORE_MODEL,
) -> dict:
    """
    BERTScore F1 using a Spanish BERT model.
    """
    # 1. Force identical lengths using zip to guarantee BERT never crashes
    preds, refs = [], []
    for p, r in zip(predictions, references):
        p_norm = normalise(p) if p.strip() else "[MISSING]"
        r_norm = normalise(r) if r.strip() else "[MISSING]"
        preds.append(p_norm)
        refs.append(r_norm)
        
    if not preds or not refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    P, R, F1 = bert_score(
        preds, refs,
        lang       = lang,
        model_type = model_type,
        verbose    = False,
    )

    return {
        "precision": float(P.mean()) * 100,
        "recall":    float(R.mean()) * 100,
        "f1":        float(F1.mean()) * 100,
    }

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """
    Corpus BLEU-4 with smoothing.
    """
    smooth = SmoothingFunction().method1
    
    hyps, refs = [], []
    for p, r in zip(predictions, references):
        h = nltk.word_tokenize(normalise(p)) if p.strip() else ["[MISSING]"]
        ref = [nltk.word_tokenize(normalise(r))] if r.strip() else [["[MISSING]"]]
        hyps.append(h)
        refs.append(ref)
        
    return corpus_bleu(refs, hyps, smoothing_function=smooth) * 100

# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────

def evaluate_results(
    predictions: list[str],
    references: list[str],
    save_path: Optional[Path] = None,
) -> dict:
    """
    Compute all metrics on parallel lists of predicted and reference lines.
    """
    assert len(predictions) == len(references), (
        f"Length mismatch: {len(predictions)} predictions, "
        f"{len(references)} references"
    )

    scores = {}

    print("[evaluate] Computing CER …")
    scores["CER (%)"] = round(compute_cer(predictions, references), 3)

    print("[evaluate] Computing WER …")
    scores["WER (%)"] = round(compute_wer(predictions, references), 3)

    print("[evaluate] Computing NED …")
    scores["NED (%)"] = round(compute_ned(predictions, references), 3)

    print("[evaluate] Computing BERTScore (Spanish) …")
    bs = compute_bertscore(predictions, references)
    scores["BERTScore Precision (%)"] = round(bs["precision"], 3)
    scores["BERTScore Recall (%)"]    = round(bs["recall"],    3)
    scores["BERTScore F1 (%)"]        = round(bs["f1"],        3)

    print("[evaluate] Computing BLEU …")
    scores["BLEU-4 (×100)"] = round(compute_bleu(predictions, references), 3)

    # print table
    print("\n" + "=" * 45)
    print(" EVALUATION RESULTS")
    print("=" * 45)
    for k, v in scores.items():
        direction = "↓ lower=better" if any(
            x in k for x in ["CER", "WER", "NED"]
        ) else "↑ higher=better"
        print(f"  {k:<32} {v:>7.3f}   {direction}")
    print("=" * 45)

    # per-line analysis (for notebook visualisation)
    per_line = []
    for pred, ref in zip(predictions, references):
        per_line.append({
            "prediction": pred,
            "reference":  ref,
            "cer":        round(cer([normalise(ref)], [normalise(pred)]) * 100, 2) if ref and pred else 100.0,
            "wer":        round(wer([normalise(ref)], [normalise(pred)]) * 100, 2) if ref and pred else 100.0,
        })
    scores["per_line"] = per_line

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"[evaluate] Report saved → {save_path}")

    return scores


# ─────────────────────────────────────────────────────────────
# ALIGNMENT LOGIC (THE FIX FOR INDEX SHIFTING)
# ─────────────────────────────────────────────────────────────

def align_lines_by_similarity(pred_lines: list[str], gt_lines: list[str]) -> tuple[list[str], list[str]]:
    """
    Prevents the 'Index Shift Trap' by mathematically matching predicted lines 
    to their most similar ground truth line, regardless of line order.
    """
    aligned_preds = []
    aligned_gts = []
    used_gt_indices = set()
    
    # 1. Match predictions to the closest Ground Truth
    for pred in pred_lines:
        best_gt = ""
        best_ratio = 0.0
        best_idx = -1
        
        for i, gt in enumerate(gt_lines):
            if i in used_gt_indices:
                continue
            # Calculate string similarity
            ratio = difflib.SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
                best_gt = gt
                
        # If they are at least 30% similar, consider it a match
        if best_ratio > 0.30:
            aligned_preds.append(pred)
            aligned_gts.append(best_gt)
            used_gt_indices.add(best_idx)
        else:
            # The model hallucinated a line (e.g., read a margin smudge)
            aligned_preds.append(pred)
            aligned_gts.append("") 
            
    # 2. Catch lines that DocTR completely missed
    for i, gt in enumerate(gt_lines):
        if i not in used_gt_indices:
            aligned_preds.append("") # Blank prediction
            aligned_gts.append(gt)   # Target GT
            
    return aligned_preds, aligned_gts


# ─────────────────────────────────────────────────────────────
# LOAD PIPELINE OUTPUT AND EVALUATE
# ─────────────────────────────────────────────────────────────

def evaluate_from_json(result_json: Path, flat_gt: dict) -> dict:
    """
    Load pipeline output JSON, align with fuzzy matching, and evaluate.
    """
    with open(result_json, encoding="utf-8") as f:
        result = json.load(f)

    page = result.get("page", result_json.stem.replace("_result", ""))

    # get predicted lines
    if "lines" in result and result["lines"]:
        pred_lines = [l.get("final", l.get("draft", "")) for l in result["lines"]]
    elif "text" in result:
        pred_lines = [l.strip() for l in result["text"].split("\n") if l.strip()]
    else:
        print(f"[evaluate] No text found in {result_json.name}")
        return {}

    # get GT lines for this source
    source = page.replace("_p001", "").replace("_p1", "")
    gt_lines = []
    for i in range(100): # Increased range just in case of longer documents
        key = f"{source}_p1_l{i:04d}"
        if key in flat_gt:
            gt_lines.append(flat_gt[key])
        else:
            # If we hit a gap but there might be more lines, keep checking up to 100
            # Usually, documents are contiguous, but just to be safe.
            pass

    # Filter out empty strings if any crept into GT
    gt_lines = [gt for gt in gt_lines if gt.strip()]

    if not gt_lines:
        print(f"[evaluate] No GT found for {source}")
        return {}

    # --- THE CRITICAL FIX: Fuzzy Alignment ---
    pred_lines, gt_lines = align_lines_by_similarity(pred_lines, gt_lines)

    n = len(pred_lines)
    print(f"[evaluate] {result_json.name}: {n} dynamically aligned line pairs")

    # If we only have empty strings after alignment (edge case), return empty
    if all(not p for p in pred_lines) and all(not g for g in gt_lines):
        return {}

    return evaluate_results(
        pred_lines, gt_lines,
        save_path=OUTPUTS_DIR / f"{result_json.stem}_eval.json",
    )


if __name__ == "__main__":
    from dataset import load_all_gt, build_line_gt_map

    gt_map  = load_all_gt()
    flat_gt = build_line_gt_map(gt_map)

    result_files = list(OUTPUTS_DIR.glob("*_result.json"))
    if not result_files:
        print("No result JSONs found. Run pipeline.py first.")
    else:
        all_scores = []
        for rj in sorted(result_files):
            print(f"\nEvaluating {rj.name} …")
            s = evaluate_from_json(rj, flat_gt)
            if s:
                all_scores.append(s)

        if all_scores:
            mean_cer = np.mean([s["CER (%)"] for s in all_scores])
            mean_wer = np.mean([s["WER (%)"] for s in all_scores])
            mean_bs  = np.mean([s["BERTScore F1 (%)"] for s in all_scores])
            print(f"\n{'='*45}")
            print(f" AGGREGATE RESULTS ({len(all_scores)} sources)")
            print(f"{'='*45}")
            print(f"  CER          : {mean_cer:.2f}%")
            print(f"  WER          : {mean_wer:.2f}%")
            print(f"  BERTScore F1 : {mean_bs:.2f}%")
            print(f"{'='*45}")