#!/usr/bin/env python3
# =============================================================================
# ArabicRAG-Eval — rouge_eval.py
# Automatic evaluation using ROUGE-L and Refusal Accuracy.
# No API key required — runs entirely locally.
#
# Metrics computed:
#   ROUGE-L        — Longest common subsequence overlap with ground truth
#   Refusal Rate   — On UNANSWERABLE questions, did the model correctly refuse?
#   RAG Benefit    — ROUGE-L delta: with_context - no_context
# =============================================================================

import re
import argparse
import pandas as pd
from pathlib import Path

# ── Inline ROUGE-L (no external dependency) ──────────────────────────────────

def _lcs_length(x: list, y: list) -> int:
    """Dynamic programming LCS length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE-L F1 between prediction and reference.
    Tokenised on whitespace (suitable for Arabic — no stemming needed).
    Returns dict with precision, recall, f1.
    """
    pred_tokens = prediction.strip().split()
    ref_tokens  = reference.strip().split()

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4)}


# ── Refusal detection ─────────────────────────────────────────────────────────

REFUSAL_PHRASE = "لا تتوفر"   # opening of the canonical refusal sentence

def is_refusal(text: str) -> bool:
    return REFUSAL_PHRASE in str(text)


def _strip_think(text: str) -> str:
    """Remove Qwen's <think>…</think> block before scoring."""
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL).strip()


# ── Main evaluation ───────────────────────────────────────────────────────────

MODELS = ["llama", "qwen", "allam"]

MODEL_DISPLAY = {
    "llama": "LLaMA 3.3 70B",
    "qwen":  "Qwen3 32B",
    "allam": "ALLaM 2 7B",
}


def evaluate(results_path: str, dataset_path: str) -> None:
    res = pd.read_csv(results_path)
    ds  = pd.read_csv(dataset_path)

    # Merge ground truth and context_dependency from dataset into results
    # (results may not have these columns if generated from original 30 rows)
    merge_cols = ["id", "ground_truth_answer", "context_dependency"]
    if "ground_truth_answer" not in res.columns:
        res = res.merge(ds[merge_cols], on="id", how="left")
    elif "context_dependency" not in res.columns:
        res = res.merge(ds[["id", "context_dependency"]], on="id", how="left")

    print("=" * 70)
    print("  ArabicRAG-Eval — ROUGE-L + Refusal Accuracy")
    print("=" * 70)
    print(f"  Results file : {Path(results_path).name}")
    print(f"  Dataset rows : {len(ds)}  |  Results rows : {len(res)}")
    print()

    # Separate answerable vs unanswerable rows in the results
    answerable   = res[res["context_dependency"] != "UNANSWERABLE"]
    unanswerable = res[res["context_dependency"] == "UNANSWERABLE"]

    print(f"  Answerable rows    : {len(answerable)}")
    print(f"  Unanswerable rows  : {len(unanswerable)}")
    print()

    summary_rows = []

    for model in MODELS:
        col_with = f"{model}_answer"
        col_no   = f"{model}_answer_no_context"

        if col_with not in res.columns:
            print(f"  [{model}] column '{col_with}' not found — skipping")
            continue

        # ── ROUGE-L on answerable questions ──────────────────────────────────
        scores_with = []
        scores_no   = []

        for _, row in answerable.iterrows():
            ref  = str(row.get("ground_truth_answer", "")).strip()
            pred = _strip_think(row.get(col_with, ""))
            scores_with.append(rouge_l(pred, ref)["f1"])

            if col_no in res.columns:
                pred_no = _strip_think(row.get(col_no, ""))
                scores_no.append(rouge_l(pred_no, ref)["f1"])

        avg_with = sum(scores_with) / len(scores_with) if scores_with else 0
        avg_no   = sum(scores_no)   / len(scores_no)   if scores_no   else None
        rag_benefit = round(avg_with - avg_no, 4) if avg_no is not None else None

        # ── Refusal accuracy on unanswerable questions ────────────────────────
        refusal_rate = None
        if len(unanswerable) > 0 and col_with in unanswerable.columns:
            n_refused = unanswerable[col_with].apply(
                lambda t: is_refusal(_strip_think(t))
            ).sum()
            refusal_rate = round(n_refused / len(unanswerable), 4)

        # ── Print per-model results ───────────────────────────────────────────
        print(f"  ── {MODEL_DISPLAY[model]} ──")
        print(f"     ROUGE-L (with context)  : {avg_with:.4f}")
        if avg_no is not None:
            print(f"     ROUGE-L (no  context)  : {avg_no:.4f}")
            direction = ("✅ context helps" if rag_benefit > 0.01
                         else "⚠️  minimal benefit" if rag_benefit >= 0
                         else "❌ context hurts")
            print(f"     RAG Benefit (Δ ROUGE-L) : {rag_benefit:+.4f}  {direction}")
        if refusal_rate is not None:
            correct = int(refusal_rate * len(unanswerable))
            indicator = ("✅" if refusal_rate >= 0.80
                         else "⚠️ " if refusal_rate >= 0.40
                         else "❌")
            print(f"     Refusal Accuracy        : {refusal_rate:.0%}  "
                  f"({correct}/{len(unanswerable)})  {indicator}")
        print()

        summary_rows.append({
            "Model":                 MODEL_DISPLAY[model],
            "ROUGE-L (with ctx)":    round(avg_with, 4),
            "ROUGE-L (no ctx)":      round(avg_no, 4) if avg_no is not None else "N/A",
            "RAG Benefit (Δ)":       f"{rag_benefit:+.4f}" if rag_benefit is not None else "N/A",
            "Refusal Accuracy":      f"{refusal_rate:.0%}" if refusal_rate is not None else "N/A",
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    if summary_rows:
        print("=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        print()

        out_path = Path(results_path).parent / "rouge_eval_results.csv"
        summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Saved: {out_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-L and Refusal Accuracy for ArabicRAG-Eval results."
    )
    parser.add_argument(
        "--results",  default="ArabicRAG-Eval_Results_v2.csv",
        help="Path to results CSV (default: ArabicRAG-Eval_Results_v2.csv)"
    )
    parser.add_argument(
        "--dataset",  default="ArabicRAG-Eval_Dataset_v2.csv",
        help="Path to dataset CSV (default: ArabicRAG-Eval_Dataset_v2.csv)"
    )
    args = parser.parse_args()
    evaluate(args.results, args.dataset)
