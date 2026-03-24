#!/usr/bin/env python3
# =============================================================================
# ArabicRAG-Eval — inference_v2.py
# Implements Abdellah's 5 strategies (refinement guide, March 2026):
#   Strategy 2 — No-Context Baseline (runs each question WITH and WITHOUT passage)
#   Strategy 4 — Named models with justification
# =============================================================================

import os, sys, time, argparse, logging, csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from groq import Groq, RateLimitError, APIError, APIConnectionError


def load_env_files() -> None:
    """Load key=value pairs from common .env locations if present."""
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
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


load_env_files()

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("arabicrag_inference_v2")

# ── Models (Strategy 4 — each chosen for a specific hypothesis) ───────────────
# Based on ArabicNLP 2024 findings:
#   - GPT-4 is best, but open-source models "not far behind"
#   - Chain-of-Thought is LESS effective in Arabic (avoid CoT prompts)
#   - Arabic-specialized models (ALLaM, AraLlama) competitive for dialect tasks
MODELS = {
    "llama": "llama-3.3-70b-versatile",  # H1: Best general multilingual (70B scale)
    "qwen":  "qwen/qwen3-32b",            # H2: Strong reasoning model
    "allam": "allam-2-7b",                # H3: Arabic-centric (Saudi ALLaM) — replaces Jais
}

MODEL_HYPOTHESIS = {
    "llama": "General multilingual upper bound — does scale beat specialisation?",
    "qwen":  "Reasoning-focused model — NOTE: CoT less effective in Arabic per ArabicNLP 2024",
    "allam": "Arabic-centric model (ALLaM) — dedicated Arabic training, comparable to Jais",
}

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """أنتَ مساعد متخصص في الإجابة عن الأسئلة باللغة العربية الفصحى.
مهمتك الوحيدة هي الإجابة عن السؤال المطروح بناءً حصرياً على المقطع النصي المُقدَّم إليك.

القواعد الإلزامية التي يجب عليك الالتزام بها:
١. استنِد في إجابتك إلى المعلومات الواردة في المقطع فحسب، ولا تُضِف أي معلومات خارجية.
٢. إذا كانت الإجابة غير موجودة في المقطع، فأجب بالضبط بالعبارة التالية: "لا تتوفر في المقطع معلومات كافية للإجابة عن هذا السؤال."
٣. اكتب إجابتك باللغة العربية الفصحى (MSA) دون لهجات أو مصطلحات أجنبية.
٤. اجعل إجابتك موجزة: جملة واحدة أو جملتان على الأكثر.
٥. لا تبدأ إجابتك بعبارات تمهيدية مثل "بناءً على المقطع..." أو "وفقاً للنص...". ابدأ مباشرةً بالإجابة."""

# WITH context — standard RAG prompt
WITH_CONTEXT_TEMPLATE = """المقطع النصي:
\"\"\"
{passage}
\"\"\"

السؤال:
{question}

الإجابة:"""

# WITHOUT context — Strategy 2 baseline
# No passage provided — tests parametric memory
NO_CONTEXT_SYSTEM = """أنتَ مساعد متخصص في الإجابة عن الأسئلة باللغة العربية الفصحى.
أجب عن السؤال التالي بناءً على معرفتك العامة.
اجعل إجابتك موجزة: جملة واحدة أو جملتان على الأكثر.
ابدأ مباشرةً بالإجابة دون مقدمات."""

NO_CONTEXT_TEMPLATE = """السؤال:
{question}

الإجابة:"""


def build_messages_with_context(passage: str, question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": WITH_CONTEXT_TEMPLATE.format(
            passage=passage.strip(), question=question.strip())}
    ]

def build_messages_no_context(question: str) -> list:
    return [
        {"role": "system", "content": NO_CONTEXT_SYSTEM},
        {"role": "user", "content": NO_CONTEXT_TEMPLATE.format(
            question=question.strip())}
    ]


def build_messages(passage: str, question: str) -> list:
    """Compatibility wrapper: defaults to WITH-context prompt."""
    return build_messages_with_context(passage, question)


def query_model(client, model_id, messages, max_retries=4, base_wait=5.0) -> str:
    attempt = 0
    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model_id, messages=messages,
                temperature=0, max_tokens=256, top_p=1, stream=False)
            answer = response.choices[0].message.content
            return answer.strip() if answer else "ERROR: empty response"
        except RateLimitError:
            wait = base_wait * (2 ** attempt)
            log.warning("Rate limit — waiting %.0fs (attempt %d/%d)", wait, attempt+1, max_retries+1)
            time.sleep(wait); attempt += 1
        except APIConnectionError as e:
            wait = base_wait * (2 ** attempt)
            log.warning("Connection error — waiting %.0fs: %s", wait, str(e))
            time.sleep(wait); attempt += 1
        except (APIError, Exception) as e:
            log.error("API error for %s: %s", model_id, str(e))
            return f"ERROR: {str(e)}"
    return f"ERROR: max retries exhausted"


def run_inference(dataset_path, output_path, models=None, delay=2.5, resume=True):
    log.info("Loading dataset: %s", dataset_path)
    df = pd.read_csv(dataset_path, encoding="utf-8")
    log.info("Dataset: %d rows", len(df))

    active_models = models or list(MODELS.keys())

    # Initialise answer columns — both WITH and WITHOUT context per model
    for m in active_models:
        for suffix in ["_answer", "_answer_no_context"]:
            col = f"{m}{suffix}"
            if col not in df.columns:
                df[col] = ""

    # Resume from existing output - copy ALL model columns, not just active ones
    if resume and Path(output_path).exists():
        log.info("Resuming from: %s", output_path)
        existing = pd.read_csv(output_path, encoding="utf-8")
        # Copy all answer columns from existing file
        for col in existing.columns:
            if "_answer" in col and col not in df.columns:
                df[col] = existing[col].fillna("").values
            elif "_answer" in col:
                df[col] = existing[col].fillna("").values

    # Init Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY not set. Export it first.")
        sys.exit(1)
    client = Groq(api_key=api_key)

    log.info("Starting inference: %d triplets × %d models × 2 conditions = %d calls",
             len(df), len(active_models), len(df)*len(active_models)*2)

    # Print model hypotheses
    log.info("── Model selection rationale (Strategy 4) ──")
    for m in active_models:
        log.info("  %-6s: %s", m.upper(), MODEL_HYPOTHESIS.get(m, ""))

    for model_key in active_models:
        model_id = MODELS[model_key]
        log.info("─── Model: %s (%s) ───", model_key.upper(), model_id)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_key:>5}", unit="triplet"):

            # ── WITH context ──────────────────────────────────────────────
            col_with = f"{model_key}_answer"
            if not (resume and isinstance(df.at[idx, col_with], str) and df.at[idx, col_with].strip()):
                msgs = build_messages_with_context(str(row["passage"]), str(row["question"]))
                df.at[idx, col_with] = query_model(client, model_id, msgs)
                df.to_csv(output_path, index=False, encoding="utf-8")
                time.sleep(delay)

            # ── WITHOUT context (Strategy 2 — no-context baseline) ────────
            col_no = f"{model_key}_answer_no_context"
            if not (resume and isinstance(df.at[idx, col_no], str) and df.at[idx, col_no].strip()):
                msgs = build_messages_no_context(str(row["question"]))
                df.at[idx, col_no] = query_model(client, model_id, msgs)
                df.to_csv(output_path, index=False, encoding="utf-8")
                time.sleep(delay)

        log.info("✓ %s complete.", model_key.upper())

    # Final column order
    output_cols = ["id", "passage", "question", "ground_truth_answer",
                   "context_dependency", "contamination_modified",
                   "llama_answer", "llama_answer_no_context",
                   "qwen_answer",  "qwen_answer_no_context",
                   "allam_answer",  "allam_answer_no_context",
                   "source_url"]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(output_path, index=False, encoding="utf-8")
    log.info("Results saved to: %s", output_path)

    # Summary
    log.info("─── Summary ───")
    for m in active_models:
        with_col = f"{m}_answer"
        no_col   = f"{m}_answer_no_context"
        if with_col in df.columns:
            errs_with = df[with_col].str.startswith("ERROR:", na=True).sum()
            errs_no   = df[no_col].str.startswith("ERROR:", na=True).sum() if no_col in df.columns else 0
            log.info("  %-6s  with_context: %d/%d errors  |  no_context: %d/%d errors",
                     m.upper(), errs_with, len(df), errs_no, len(df))


def parse_args():
    parser = argparse.ArgumentParser(
        description="ArabicRAG-Eval v2: inference with no-context baseline (Strategy 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="ArabicRAG-Eval_Dataset_v2.csv")
    parser.add_argument("--output",  default="ArabicRAG-Eval_Results_v2.csv")
    parser.add_argument("--models",  nargs="+", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--delay",   type=float, default=2.5, help="Delay between API calls (increase if rate limited)")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        dataset_path=args.dataset,
        output_path=args.output,
        models=args.models,
        delay=args.delay,
        resume=not args.no_resume,
    )
