#!/usr/bin/env python3
"""
ArabicRAG-Eval — evaluate_groq_judge.py
Free LLM-as-Judge evaluation using Groq (LLaMA 3.3 70B)
No OpenAI required!

Implements simplified versions of:
  - Faithfulness: Does the answer stick to facts in the context?
  - Answer Relevancy: Does the answer address the question?
  - RAG Benefit: Compare with-context vs no-context scores
"""

import os, sys, argparse, json, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ── Load .env files ───────────────────────────────────────────────────────────
def load_env_files() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)

load_env_files()

try:
    from groq import Groq
except ImportError:
    print("Missing groq — run: pip install groq")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
# Cross-evaluation: use DIFFERENT model family as judge to avoid self-bias
# Based on ArabicNLP 2024: avoid letting a model judge its own outputs
# LLaMA answers → judged by Qwen
# Qwen answers  → judged by LLaMA  
# ALLaM answers → judged by Qwen (different family)
JUDGE_MODELS = {
    "llama": "qwen/qwen3-32b",           # Qwen judges LLaMA
    "qwen":  "llama-3.1-8b-instant",     # LLaMA judges Qwen
    "allam": "qwen/qwen3-32b",           # Qwen judges ALLaM
}
DEFAULT_JUDGE = "qwen/qwen3-32b"

MODELS = ["llama", "qwen", "allam"]
MODEL_DISPLAY = {
    "llama": "LLaMA 3.3 70B",
    "qwen":  "Qwen3 32B", 
    "allam": "ALLaM 2 7B (Arabic)",
}

# ── Judge Prompts ─────────────────────────────────────────────────────────────
FAITHFULNESS_PROMPT = """You are an expert evaluator for Arabic question-answering systems.

Given:
- Question: {question}
- Context/Passage: {context}
- Answer: {answer}

Evaluate FAITHFULNESS: Does the answer only contain information that can be verified from the given context? 
- Score 1.0: Answer is fully supported by the context
- Score 0.5: Answer is partially supported, some claims are not in context
- Score 0.0: Answer contains claims that contradict or are not in the context

Respond with ONLY a JSON object: {{"score": <number>, "reason": "<brief explanation>"}}"""

RELEVANCY_PROMPT = """You are an expert evaluator for Arabic question-answering systems.

Given:
- Question: {question}
- Answer: {answer}

Evaluate ANSWER RELEVANCY: Does the answer actually address what was asked?
- Score 1.0: Answer directly and completely addresses the question
- Score 0.5: Answer partially addresses the question or includes irrelevant info
- Score 0.0: Answer does not address the question at all

Respond with ONLY a JSON object: {{"score": <number>, "reason": "<brief explanation>"}}"""


def call_judge(client: Groq, prompt: str, judge_model: str, retries: int = 3) -> dict:
    """Call Groq judge with retry logic."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON from response
            if "{" in text and "}" in text:
                json_str = text[text.find("{"):text.rfind("}")+1]
                return json.loads(json_str)
            return {"score": 0.5, "reason": "Could not parse response"}
        except json.JSONDecodeError:
            return {"score": 0.5, "reason": "Invalid JSON response"}
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"score": 0.0, "reason": f"Error: {str(e)[:50]}"}
    return {"score": 0.0, "reason": "Max retries exceeded"}


def evaluate_row(client: Groq, row: pd.Series, answer_col: str, judge_model: str) -> dict:
    """Evaluate a single row for faithfulness and relevancy."""
    answer = str(row.get(answer_col, "")).strip()
    if not answer or answer.startswith("ERROR:"):
        return {"faithfulness": None, "relevancy": None, "error": True}
    
    question = str(row["question"]).strip()
    context = str(row["passage"]).strip()
    
    # Faithfulness
    faith_prompt = FAITHFULNESS_PROMPT.format(
        question=question, context=context, answer=answer
    )
    faith_result = call_judge(client, faith_prompt, judge_model)
    
    # Small delay to avoid rate limits
    time.sleep(0.5)
    
    # Relevancy
    rel_prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)
    rel_result = call_judge(client, rel_prompt, judge_model)
    
    return {
        "faithfulness": faith_result.get("score"),
        "faithfulness_reason": faith_result.get("reason", ""),
        "relevancy": rel_result.get("score"),
        "relevancy_reason": rel_result.get("reason", ""),
        "error": False
    }


def score_model(client: Groq, df: pd.DataFrame, model_key: str, use_no_context: bool = False) -> pd.DataFrame:
    """Score all rows for one model."""
    suffix = "_answer_no_context" if use_no_context else "_answer"
    answer_col = f"{model_key}{suffix}"
    condition = "NO-CONTEXT" if use_no_context else "WITH-CONTEXT"
    
    # Cross-evaluation: use different model as judge
    judge_model = JUDGE_MODELS.get(model_key, DEFAULT_JUDGE)
    print(f"\n📊 Scoring {MODEL_DISPLAY[model_key]} — {condition}")
    print(f"   Judge: {judge_model} (cross-evaluation)")
    
    results = []
    valid_rows = df[
        ~df[answer_col].str.startswith("ERROR:", na=True) &
        df[answer_col].str.strip().ne("")
    ]
    
    for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc=f"  {model_key}"):
        scores = evaluate_row(client, row, answer_col, judge_model)
        scores["id"] = row["id"]
        scores["model"] = model_key
        scores["judge"] = judge_model
        scores["condition"] = "no_context" if use_no_context else "with_context"
        results.append(scores)
        time.sleep(0.5)  # Rate limit protection
    
    return pd.DataFrame(results)


def compute_summary(all_with: dict, all_no: dict) -> pd.DataFrame:
    """Build summary table with RAG benefit."""
    rows = []
    for model_key in MODELS:
        with_df = all_with.get(model_key, pd.DataFrame())
        no_df = all_no.get(model_key, pd.DataFrame())
        
        row = {"Model": MODEL_DISPLAY[model_key]}
        
        for metric in ["faithfulness", "relevancy"]:
            if metric in with_df.columns:
                w_mean = with_df[metric].dropna().mean()
                row[f"{metric}_with"] = round(w_mean, 3) if pd.notna(w_mean) else None
            if metric in no_df.columns:
                n_mean = no_df[metric].dropna().mean()
                row[f"{metric}_no"] = round(n_mean, 3) if pd.notna(n_mean) else None
            # RAG Benefit
            w = row.get(f"{metric}_with")
            n = row.get(f"{metric}_no")
            if w is not None and n is not None:
                benefit = w - n
                row[f"{metric}_benefit"] = round(benefit, 3)
                row[f"{metric}_interp"] = (
                    "✅ Uses context" if benefit > 0.05
                    else "⚠️ Low benefit" if benefit >= 0
                    else "❌ Context hurts"
                )
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic RAG with Groq LLM Judge (free)")
    parser.add_argument("--input", "-i", default="inference_results_v2.csv",
                        help="Input CSV with inference results")
    parser.add_argument("--output", "-o", default="eval_groq_judge.csv",
                        help="Output CSV for evaluation results")
    parser.add_argument("--models", "-m", nargs="+", default=MODELS,
                        choices=MODELS, help="Models to evaluate")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows for testing (default: all)")
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set. Add it to .env file.")
        sys.exit(1)
    
    client = Groq(api_key=api_key)
    
    # Load data
    input_path = Path(__file__).parent / args.input
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print("  Run inference_v2.py first to generate answers.")
        sys.exit(1)
    
    df = pd.read_csv(input_path)
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"📂 Loaded {len(df)} rows from {input_path.name}")
    
    all_with = {}
    all_no = {}
    
    for model_key in args.models:
        # WITH context
        with_col = f"{model_key}_answer"
        if with_col in df.columns:
            all_with[model_key] = score_model(client, df, model_key, use_no_context=False)
        
        # NO context (baseline)
        no_col = f"{model_key}_answer_no_context"
        if no_col in df.columns:
            all_no[model_key] = score_model(client, df, model_key, use_no_context=True)
    
    # Summary
    summary = compute_summary(all_with, all_no)
    
    print("\n" + "="*70)
    print("📊 EVALUATION SUMMARY (Groq LLM Judge)")
    print("="*70)
    print(summary.to_string(index=False))
    
    # Save results
    output_path = Path(__file__).parent / args.output
    
    # Combine all scores
    all_scores = []
    for model_key in args.models:
        if model_key in all_with:
            all_scores.append(all_with[model_key])
        if model_key in all_no:
            all_scores.append(all_no[model_key])
    
    if all_scores:
        combined = pd.concat(all_scores, ignore_index=True)
        combined.to_csv(output_path, index=False)
        print(f"\n✅ Detailed scores saved to: {output_path}")
    
    summary_path = output_path.with_name("eval_summary_groq.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")
    
    print("\n🎉 Evaluation complete! (No OpenAI required)")


if __name__ == "__main__":
    main()
