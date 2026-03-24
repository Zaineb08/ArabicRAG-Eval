# ArabicRAG-Eval v2

**A Faithfulness Benchmark for Retrieval-Augmented Generation in Arabic**

> Arabic NLP School 2026 · EACL 2026 · Rabat, Morocco · March 24, 2026  
> Team: Data Resources & Benchmarking — Team 2  
> Members: Zaineb Rahmani · Abderrahmane Jouilili · Moumni Mohammed · Wissal Said  
> Mentors: Go Inoue · Hamdy Mubarak

---

## What's new in v2 (based on mentor feedback)

This version implements all 5 strategies recommended by **Abdellah** (refinement guide, March 2026):

| Strategy                          | What we did                                                                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **1 — Cross-Evaluation**          | Judge model is always different from the model being tested (GPT-4o judges LLaMA, Qwen, Jais — not itself)                        |
| **2 — No-Context Baseline**       | Every question is run twice: WITH the passage AND without. The difference = RAG benefit score                                     |
| **3 — Context Dependency Tags**   | Each question is labelled GENERAL or CONTEXT-DEPENDENT. Scores reported separately                                                |
| **4 — Named Model Justification** | Each model chosen for a specific hypothesis (not just a random leaderboard)                                                       |
| **5 — Contamination Guard**       | 10 passages have 2-3 facts changed (dates, numbers). If models answer with original facts, they are using memory not your passage |

---

## File Structure

```
ArabicRAG-Eval-Submission/
│
├── README.md                                   ← this file
├── REPORT.md                                   ← full evaluation report
│
├── data/
│   ├── ArabicRAG-Eval_Dataset_v2.csv           ← 35 questions (30 answerable + 5 unanswerable)
│   └── ArabicRAG-Eval_Results_v2.csv           ← all model answers
│
├── code/
│   ├── inference_v2.py                         ← generates answers (Strategy 2 & 4)
│   ├── rouge_eval.py                           ← ROUGE-L + refusal accuracy (no API needed)
│   ├── evaluate_v2.py                          ← RAGAS cross-judge (requires OpenAI key)
│   └── evaluate_groq_judge.py                  ← LLM-as-Judge via Groq (free alternative)
│
└── results/
    └── rouge_eval_results.csv                  ← ROUGE-L summary scores
```

---

## Dataset columns (v2)

The file `ArabicRAG-Eval_Dataset_v2.csv` has these columns:

| Column                   | Description                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `id`                     | Triplet ID (1–35)                                                                               |
| `passage`                | Arabic MSA passage (some facts modified — see contamination_modified)                           |
| `question`               | Open-ended Arabic MSA question                                                                  |
| `ground_truth_answer`    | Reference answer (for unanswerable questions, the ground truth is the canonical refusal phrase) |
| `context_dependency`     | `GENERAL`, `CONTEXT-DEPENDENT`, or **`UNANSWERABLE`** — strategies 3 & hallucination test       |
| `contamination_modified` | `YES` if passage facts were changed (Strategy 5)                                                |
| `original_fact_changed`  | What original fact was replaced (for transparency)                                              |
| `source_url`             | Source URL                                                                                      |

**Distribution:**

- GENERAL: 13 questions
- CONTEXT-DEPENDENT: 17 questions ✅ (Go requires ≥ 15)
- **UNANSWERABLE: 5 questions** — answer intentionally absent; tests hallucination vs. refusal
- Contamination-modified passages: 10

---

## Step 0 — Install dependencies

```bash
pip install groq pandas tqdm ragas datasets openai openpyxl langchain-openai
```

---

## Step 1 — Get your API keys

**Groq API** (free) — to run the 3 models  
→ https://console.groq.com → API Keys → Create new key

**OpenAI API** — used by RAGAS as judge (Strategy 1)  
→ https://platform.openai.com → API Keys → Create new key

---

## Step 2 — Set your API keys

```bash
# Mac / Linux
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
```

```powershell
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
$env:OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"
```

---

## Step 3 — Run inference (WITH and WITHOUT context)

```bash
python code/inference_v2.py \
    --dataset data/ArabicRAG-Eval_Dataset_v2.csv \
    --output  data/ArabicRAG-Eval_Results_v2.csv
```

**What happens:**

- Each of 30 questions is sent to all 3 models **twice**:
  - Once WITH the Arabic passage (standard RAG)
  - Once WITHOUT the passage (memory baseline — Strategy 2)
- Total: 30 × 3 models × 2 conditions = **180 API calls**
- Saves a checkpoint after every call — safe to interrupt and resume

**Output columns added:**

```
llama_answer            ← LLaMA answer with passage
llama_answer_no_context ← LLaMA answer without passage
qwen_answer
qwen_answer_no_context
allam_answer            ← ALLaM 2 7B answer (replaces Jais — unavailable on Groq)
allam_answer_no_context
```

**Test with one model only:**

```bash
python code/inference_v2.py --dataset data/ArabicRAG-Eval_Dataset_v2.csv --output data/ArabicRAG-Eval_Results_v2.csv --models llama
```

---

## Step 4 — Run automatic evaluation (ROUGE-L + Refusal Accuracy)

```bash
python code/rouge_eval.py --results data/ArabicRAG-Eval_Results_v2.csv --dataset data/ArabicRAG-Eval_Dataset_v2.csv
```

**What it computes — no API key needed:**

| Metric               | What it measures                                                                                            |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| **ROUGE-L**          | Word overlap between model answer and ground truth (0–1, higher = better)                                   |
| **RAG Benefit (Δ)**  | ROUGE-L with context minus ROUGE-L without context — measures how much the retrieved passage actually helps |
| **Refusal Accuracy** | On the 5 unanswerable questions, did the model correctly refuse instead of hallucinating?                   |

**Expected output:**

```
Model          ROUGE-L (with ctx)  ROUGE-L (no ctx)  RAG Benefit (Δ)  Refusal Accuracy
LLaMA 3.3 70B  0.7333              0.2936            +0.4396          100%
Qwen3 32B      0.5074              0.2301            +0.2773           80%
ALLaM 2 7B     0.6397              0.2991            +0.3405          100%
```

**Saved to:** `rouge_eval_results.csv`

---

## Step 5 — Evaluate with RAGAS cross-judge (requires OpenAI key)

```bash
python code/evaluate_v2.py \
    --results data/ArabicRAG-Eval_Results_v2.csv \
    --output  results/ArabicRAG-Eval_Scores_v2
```

**What happens:**

- **Strategy 1**: GPT-4o is the judge for all 3 models (none judge themselves)
- **Strategy 2**: Scores computed for both conditions — RAG benefit = WITH minus WITHOUT
- **Strategy 3**: Scores split by GENERAL vs CONTEXT-DEPENDENT questions

> **Note:** Requires a paid OpenAI API key. `rouge_eval.py` (Step 4) provides free automatic evaluation as an alternative.

---

## Step 5 — Read the results

### Main summary (`_Summary.csv`)

```
Model           Faith (W)  Faith (No)   Benefit
LLaMA 3.3 70B    0.921      0.743        +0.178   ← uses context well
Qwen QwQ 32B     0.887      0.812        +0.075   ← moderate benefit
Jais 70B         0.953      0.701        +0.252   ← strongest RAG benefit
```

### RAG Benefit (`_RAGBenefit.csv`) — Strategy 2

```
Metric          With Context  No Context  RAG Benefit  Interpretation
faithfulness       0.921         0.743       +0.178     ✅ Model uses context
answer_relevancy   0.874         0.861       +0.013     ⚠️ Low RAG benefit
context_precision  0.813         0.671       +0.142     ✅ Model uses context
```

> **Key insight:** If RAG benefit is near 0, the model is answering from memory — your RAG system adds no value for that model.

### Context Dependency Split — Strategy 3

```
Model          Category              Faithfulness
LLaMA          GENERAL               0.891
LLaMA          CONTEXT-DEPENDENT     0.734   ← harder when passage required
Jais           GENERAL               0.920
Jais           CONTEXT-DEPENDENT     0.961   ← Arabic model excels on Arabic passages
```

### Excel Report (`_Report.xlsx`) — 5 sheets

| Sheet                           | Content                                 |
| ------------------------------- | --------------------------------------- |
| Summary                         | All metrics, all models, all conditions |
| RAG Benefit (Strategy 2)        | WITH vs WITHOUT context comparison      |
| Context Dependency (Strategy 3) | GENERAL vs CONTEXT-DEPENDENT scores     |
| LLaMA (w/ctx)                   | Per-triplet scores with colour gradient |
| Qwen (w/ctx)                    | Per-triplet scores with colour gradient |
| Jais (w/ctx)                    | Per-triplet scores with colour gradient |

---

## Understanding contamination (Strategy 5)

10 passages have had 2-3 facts modified (dates, numbers, names).

**Example — Passage 6 (Ibn Sina):**

- Original: born 980 AD, 450 books, Canon used until 17th century
- Modified: born 975 AD, 380 books, Canon used until **16th** century

**How to interpret results:**

- If a model answers "17th century" → it used its memory (parametric leakage)
- If a model answers "16th century" → it read YOUR passage (genuine RAG)
- Check the `contamination_modified` and `original_fact_changed` columns in the dataset

---

## Model hypotheses (Strategy 4)

| Model             | Why chosen                           | Hypothesis                                            |
| ----------------- | ------------------------------------ | ----------------------------------------------------- |
| **LLaMA 3.3 70B** | Best general multilingual baseline   | Does scale beat specialisation?                       |
| **Qwen QwQ 32B**  | Strong reasoning capabilities        | Does chain-of-thought help Arabic faithfulness?       |
| **ALLaM 2 7B**    | Arabic-centric (SDAIA, Saudi Arabia) | Does dedicated Arabic training maximise faithfulness? |

> **Note:** Jais 70B was originally planned but unavailable on Groq. ALLaM 2 7B is used as a comparable Arabic-first alternative.

---

## Quick start (all steps)

```bash
pip install groq pandas tqdm ragas datasets openai openpyxl langchain-openai

export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"  # only needed for Step 5

# Step 3 — Generate answers
python code/inference_v2.py \
    --dataset data/ArabicRAG-Eval_Dataset_v2.csv \
    --output  data/ArabicRAG-Eval_Results_v2.csv

# Step 4 — Automatic evaluation (free, no API key)
python code/rouge_eval.py \
    --results data/ArabicRAG-Eval_Results_v2.csv \
    --dataset data/ArabicRAG-Eval_Dataset_v2.csv

# Step 5 — RAGAS cross-judge (requires OpenAI key)
python code/evaluate_v2.py \
    --results data/ArabicRAG-Eval_Results_v2.csv \
    --output  results/ArabicRAG-Eval_Scores_v2
```

---

## Troubleshooting

| Error                        | Fix                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `GROQ_API_KEY not set`       | Run `export GROQ_API_KEY=...` again                                                |
| Rate limit errors            | Add `--delay 3` to slow down calls                                                 |
| `jais-adapted-70b not found` | Run with `--models llama qwen` only                                                |
| `OPENAI_API_KEY not set`     | Set it — RAGAS needs it for the judge                                              |
| Crashed halfway              | Re-run same command — it resumes automatically                                     |
| `ModuleNotFoundError`        | Run `pip install groq pandas tqdm ragas datasets openai openpyxl langchain-openai` |

## Links

https://gemini.google.com/share/445fa9c1bebb
https://gemini.google.com/share/d247da2e18cf
