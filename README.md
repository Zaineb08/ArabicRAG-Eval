# ArabicRAG-Eval

**A Faithfulness Benchmark for Retrieval-Augmented Generation in Arabic**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

> **Arabic NLP School 2026 · EACL 2026 · Rabat, Morocco**  
> **Team:** Zaineb Rahmani · Abderrahmane Jouilili · Moumni Mohammed · Wissal Said · Abdellah Hasnaoui · Soukaina Assam  
> **Mentors:** Go Inoue · Hamdy Mubarak

---

## What is ArabicRAG-Eval?

**ArabicRAG-Eval** is the first standardized benchmark for evaluating Retrieval-Augmented Generation (RAG) systems in Arabic. It measures three critical properties:

| Property                     | Question We Answer                                                    |
| ---------------------------- | --------------------------------------------------------------------- |
| **Faithfulness**             | Does the model actually read and use the retrieved passage?           |
| **Hallucination Resistance** | Does the model refuse to answer when the answer isn't in the passage? |
| **RAG Benefit**              | How much does providing a passage improve the answer quality?         |

### Why This Matters

RAG is everywhere — legal assistants, medical chatbots, customer service tools. But for Arabic, there was no way to systematically test if these systems actually work. A model that ignores the document and makes up answers is dangerous. **ArabicRAG-Eval lets you catch that.**

---

## Key Results

We tested **3 models** on **35 Arabic questions**:

| Model             | ROUGE-L   | RAG Benefit | Refusal Accuracy |
| ----------------- | --------- | ----------- | ---------------- |
| **LLaMA 3.3 70B** | **0.733** | **+0.440**  | **100%** ✓       |
| ALLaM 2 7B        | 0.640     | +0.341      | **100%** ✓       |
| Qwen3 32B         | 0.507     | +0.277      | 80% ✗            |

### 3 Key Findings

1. **Chain-of-Thought hurts Arabic** — Qwen produced English reasoning blocks in 100% of responses, lowering quality
2. **Specialization beats scale** — ALLaM 7B (Arabic-specialized) achieved 87% of LLaMA 70B's score with 10× fewer parameters
3. **Size ≠ Safety** — Only Qwen hallucinated on unanswerable questions; LLaMA & ALLaM refused correctly

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Zaineb08/ArabicRAG-Eval.git
cd ArabicRAG-Eval
```

### 2. Install dependencies

```bash
pip install groq pandas tqdm openpyxl
```

### 3. Get your API key (free)

Get a **Groq API key** (free tier, no credit card):
→ https://console.groq.com → API Keys → Create new key

### 4. Set your API key

**Mac/Linux:**

```bash
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"
```

**Windows PowerShell:**

```powershell
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

### 5. Run inference

```bash
python code/inference_v2.py \
    --dataset data/ArabicRAG-Eval_Dataset_v2.csv \
    --output data/ArabicRAG-Eval_Results_v2.csv
```

This runs each question through all 3 models, with and without context (180 API calls total).

### 6. Run evaluation (no API needed)

```bash
python code/rouge_eval.py \
    --results data/ArabicRAG-Eval_Results_v2.csv \
    --dataset data/ArabicRAG-Eval_Dataset_v2.csv
```

**Output:** ROUGE-L scores, RAG benefit, and refusal accuracy for each model.

---

## Project Structure

```
ArabicRAG-Eval/
│
├── README.md                           ← You are here
├── REPORT.md                           ← Full evaluation report with analysis
├── RESEARCH_PROPOSAL.md                ← Academic proposal (English + French)
│
├── data/
│   ├── ArabicRAG-Eval_Dataset_v2.csv   ← 35 Arabic Q&A triplets
│   └── ArabicRAG-Eval_Results_v2.csv   ← Model predictions (180 answers)
│
├── code/
│   ├── inference_v2.py                 ← Run models via Groq API
│   ├── rouge_eval.py                   ← ROUGE-L evaluation (free, no API)
│   ├── evaluate_v2.py                  ← RAGAS evaluation (requires OpenAI)
│   └── evaluate_groq_judge.py          ← LLM-as-Judge via Groq (free)
│
└── results/
    └── rouge_eval_results.csv          ← Summary scores
```

---

## Dataset

The dataset contains **35 Arabic question-passage-answer triplets**:

| Category          | Count | Description                                         |
| ----------------- | ----- | --------------------------------------------------- |
| Context-Dependent | 17    | Answer exists only in the passage                   |
| General Knowledge | 13    | Answer may exist in model's memory                  |
| **Unanswerable**  | **5** | Answer is deliberately absent — tests hallucination |

### Dataset Columns

| Column                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `id`                     | Question ID (1–35)                                |
| `passage`                | Arabic Wikipedia passage                          |
| `question`               | Arabic question                                   |
| `ground_truth_answer`    | Reference answer                                  |
| `context_dependency`     | `GENERAL`, `CONTEXT-DEPENDENT`, or `UNANSWERABLE` |
| `contamination_modified` | `YES` if facts were changed (anti-memorization)   |

---

## Evaluation Methods

### Option A: ROUGE-L (Free, Automatic)

```bash
python code/rouge_eval.py --results data/ArabicRAG-Eval_Results_v2.csv --dataset data/ArabicRAG-Eval_Dataset_v2.csv
```

| Metric               | What It Measures                              |
| -------------------- | --------------------------------------------- |
| **ROUGE-L**          | Word overlap with ground truth (0–1)          |
| **RAG Benefit**      | ROUGE with context − ROUGE without context    |
| **Refusal Accuracy** | % of unanswerable questions correctly refused |

### Option B: RAGAS (Requires OpenAI Key)

```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
python code/evaluate_v2.py --results data/ArabicRAG-Eval_Results_v2.csv --output results/scores
```

Uses GPT-4o as a judge to score faithfulness, relevancy, and precision.

### Option C: LLM-as-Judge via Groq (Free)

```bash
python code/evaluate_groq_judge.py --results data/ArabicRAG-Eval_Results_v2.csv
```

Uses a different Groq model to judge answers (cross-evaluation).

---

## Models Tested

| Model             | Parameters | Why We Chose It                                             |
| ----------------- | ---------- | ----------------------------------------------------------- |
| **LLaMA 3.3 70B** | 70B        | Multilingual baseline — does scale dominate?                |
| **Qwen3 32B**     | 32B        | Reasoning-focused — does Chain-of-Thought help Arabic?      |
| **ALLaM 2 7B**    | 7B         | Arabic-specialized (SDAIA) — does pretraining offset scale? |

---

## Methodology (5 Strategies)

Our evaluation implements research-grade methodology:

| Strategy                         | Implementation                                         |
| -------------------------------- | ------------------------------------------------------ |
| **1. Cross-Evaluation**          | Judge model ≠ tested model (prevents self-bias)        |
| **2. No-Context Baseline**       | Each question tested WITH and WITHOUT passage          |
| **3. Context Dependency Tags**   | Questions labeled by type for stratified analysis      |
| **4. Named Model Justification** | Each model tests a specific hypothesis                 |
| **5. Contamination Guard**       | 10 passages have modified facts to detect memorization |

---

## Reproducing Our Results

**Full reproduction (all 180 predictions):**

```bash
# 1. Run inference (~15 min with Groq free tier)
python code/inference_v2.py --dataset data/ArabicRAG-Eval_Dataset_v2.csv --output my_results.csv

# 2. Evaluate
python code/rouge_eval.py --results my_results.csv --dataset data/ArabicRAG-Eval_Dataset_v2.csv
```

**Quick test (one model only):**

```bash
python code/inference_v2.py --dataset data/ArabicRAG-Eval_Dataset_v2.csv --output test.csv --models llama
```

**Resume interrupted runs:**
The inference script saves checkpoints after every API call. Just re-run the same command to resume.

---

## Troubleshooting

| Problem                     | Solution                                                                 |
| --------------------------- | ------------------------------------------------------------------------ |
| `GROQ_API_KEY not set`      | Run `export GROQ_API_KEY="your_key"` or `$env:GROQ_API_KEY = "your_key"` |
| Rate limit errors           | Script auto-retries with exponential backoff; just wait                  |
| Model unavailable           | ALLaM replaces Jais (Jais unavailable on Groq)                           |
| Qwen outputs `<think>` tags | Normal — script strips them before scoring                               |

---

## Citation

If you use ArabicRAG-Eval in your research, please cite:

```bibtex
@misc{arabicrag-eval-2026,
  title={ArabicRAG-Eval: A Faithfulness Benchmark for Retrieval-Augmented Generation in Arabic},
  author={Rahmani, Zaineb and Jouilili, Abderrahmane and Mohammed, Moumni and Said, Wissal and Hasnaoui, Abdellah and Assam, Soukaina},
  year={2026},
  howpublished={Arabic NLP School 2026, EACL},
  url={https://github.com/Zaineb08/ArabicRAG-Eval}
}
```

---

## License

This project is open source under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Mentors:** Go Inoue and Hamdy Mubarak for guidance and feedback
- **Arabic NLP School 2026** for the opportunity to develop this research
- **Groq** for providing free API access to run large language models
- **SDAIA** for developing ALLaM, the Arabic-specialized model

---

## Contact

For questions or collaboration:

- **GitHub Issues:** https://github.com/Zaineb08/ArabicRAG-Eval/issues
- **Team Lead:** Zaineb Rahmani
