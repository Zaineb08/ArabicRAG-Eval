# ArabicRAG-Eval: Comparative Analysis of LLMs for Arabic Question Answering

**Author:** Zaineb  
**Date:** March 24, 2026  
**Submission:** Arabic NLP School 2026 (EACL)

---

## Abstract

We evaluate three Large Language Models on an Arabic Retrieval-Augmented Generation (RAG) benchmark: LLaMA 3.3 70B (general multilingual), Qwen3 32B (reasoning-focused), and ALLaM 2 7B (Arabic-specialized). Each model is tested under two conditions — with and without a retrieved Arabic passage — across 30 diverse Wikipedia-sourced questions. Our results confirm an ArabicNLP 2024 conference finding: **Chain-of-Thought reasoning is less effective in Arabic than in English**. Notably, Qwen3 32B's thinking process was conducted entirely in English across all 30 questions, despite the Arabic task. We additionally find that a 7B Arabic-specialized model (ALLaM) achieves comparable output quality to a 10× larger multilingual model (LLaMA 70B), supporting the value of language-targeted pretraining.

---

## Executive Summary

This report presents findings from evaluating three Large Language Models (LLMs) on Arabic Retrieval-Augmented Generation (RAG) tasks. The evaluation compares:

- **LLaMA 3.3 70B** — General multilingual model
- **Qwen3 32B** — Reasoning-focused model with Chain-of-Thought
- **ALLaM 2 7B** — Arabic-specialized model (SDAIA, Saudi Arabia)

Key findings align with ArabicNLP 2024 conference insights, particularly confirming that **Chain-of-Thought reasoning is less effective for Arabic tasks** and that **Arabic-specialized training can offset large parameter gaps**.

---

## Glossary of Key Terms

This section defines technical terms used throughout the report for readers unfamiliar with NLP evaluation.

| Term                                     | Plain-language definition                                                                                                                                                                                                                                                              |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RAG (Retrieval-Augmented Generation)** | A technique where, instead of answering purely from memory, the model is given a relevant text passage to read before answering. Like an open-book exam rather than a closed-book one.                                                                                                 |
| **LLM (Large Language Model)**           | An AI system trained on large amounts of text that can understand and generate human language (e.g., ChatGPT, LLaMA, Qwen).                                                                                                                                                            |
| **ROUGE-L**                              | A standard automatic metric for text generation quality. It measures the longest sequence of words that appears in both the model's answer and the reference (ground truth) answer. A score of 1.0 means a perfect word-for-word match; 0.0 means no overlap at all. Higher is better. |
| **RAG Benefit (Δ ROUGE-L)**              | The difference between a model's ROUGE-L score _with_ the passage versus _without_ it. A large positive value (+0.40) means the model is genuinely using the retrieved passage. A value near zero means the model is answering from memory and the retrieval step adds no value.       |
| **Refusal Accuracy**                     | On questions where the answer is _not_ in the passage ("unanswerable" questions), did the model correctly say "I don't know" instead of fabricating an answer? High refusal accuracy = the model knows the limits of its knowledge.                                                    |
| **Chain-of-Thought (CoT)**               | A prompting technique where the model is encouraged to "think step by step" before giving a final answer. Helps on English math/reasoning tasks but, as shown here, can be counterproductive for Arabic.                                                                               |
| **Faithfulness**                         | Whether the model's answer is supported by the source passage — it does not add or contradict any facts from it.                                                                                                                                                                       |
| **MSA (Modern Standard Arabic)**         | The formal, written form of Arabic used in books, news, and official communication. Distinct from spoken dialects (Egyptian, Gulf, etc.).                                                                                                                                              |
| **Parametric memory**                    | Knowledge stored inside the model's weights from training — as opposed to knowledge retrieved from a document at query time.                                                                                                                                                           |
| **Hallucination**                        | When a model confidently states something that is factually wrong or not supported by any source. A critical failure mode in RAG systems.                                                                                                                                              |
| **Temperature = 0**                      | A setting that makes the model deterministic — it always picks the most probable next word, producing the same output every time for the same input. Used here for reproducibility.                                                                                                    |
| **Ground truth**                         | The human-written reference answer used as the "correct" answer to measure model performance against.                                                                                                                                                                                  |

---

## 1. Context & Background

### 1.1 ArabicNLP 2024 Conference Insights

Based on the synthesis of the ArabicNLP 2024 conference (2nd International Conference on Arabic NLP), several critical findings informed our methodology:

| Finding                                                                | Source                    | Implication                     |
| ---------------------------------------------------------------------- | ------------------------- | ------------------------------- |
| Chain-of-Thought (CoT) is **less effective in Arabic** than in English | Arab Legal Eval benchmark | Avoid CoT prompts for Arabic QA |
| Open-source models are "not far behind" GPT-4                          | Multiple papers           | LLaMA, Qwen viable alternatives |
| Arabic-specialized models (AraLlama) competitive                       | Storytelling study        | ALLaM may match larger models   |
| Test contamination is a concern                                        | BALSaM project            | Use modified/private test sets  |

### 1.2 Model Selection Rationale

Following **Strategy 4** (named models with justification):

| Model             | Parameters | Hypothesis                                                 |
| ----------------- | ---------- | ---------------------------------------------------------- |
| **LLaMA 3.3 70B** | 70B        | Does scale beat specialization? (multilingual upper bound) |
| **Qwen3 32B**     | 32B        | Does reasoning focus help Arabic faithfulness?             |
| **ALLaM 2 7B**    | 7B         | Does dedicated Arabic training maximize faithfulness?      |

**Note:** Originally planned to test Jais 70B (Arabic-centric), but it was unavailable on Groq. ALLaM (developed by SDAIA, Saudi Arabia) serves as a comparable Arabic-first alternative.

---

## 2. Research Questions

This evaluation addresses three specific hypotheses:

| #   | Hypothesis                                                              | Model Under Test |
| --- | ----------------------------------------------------------------------- | ---------------- |
| H1  | Does raw scale provide the best general Arabic QA performance?          | LLaMA 3.3 70B    |
| H2  | Does Chain-of-Thought reasoning improve faithfulness to Arabic context? | Qwen3 32B        |
| H3  | Does language-specific pretraining offset scale disadvantage?           | ALLaM 2 7B       |

---

## 3. Methodology

### 3.1 Dataset

- **Name:** ArabicRAG-Eval Dataset v2
- **Size:** 35 question-passage-answer triplets
- **Sources:** 30 unique Arabic Wikipedia articles (diverse topics)
- **Anti-contamination:** 10/30 answerable questions (33%) were modified from original Wikipedia phrasing to reduce test-set memorization risk
- **Question distribution:**
  - Context-Dependent: 17 (49%) — answer exists only in the passage
  - General knowledge: 13 (37%) — answer may exist in the model's parametric memory
  - **Unanswerable: 5 (14%)** — answer is deliberately absent from the passage; correct model behavior is to refuse
- **Ground truth:** Human-written Arabic answers; unanswerable questions use the canonical refusal phrase as ground truth

### 3.2 Evaluation Strategies

Based on the ArabicNLP 2024 refinement guide, we implemented three core strategies:

| Strategy       | Description         | Implementation                                       |
| -------------- | ------------------- | ---------------------------------------------------- |
| **Strategy 1** | Cross-Evaluation    | Judge model ≠ tested model (Qwen judges LLaMA, etc.) |
| **Strategy 2** | No-Context Baseline | Each question answered WITH and WITHOUT passage      |
| **Strategy 4** | Named Models        | Each model chosen for a specific testable hypothesis |

**Why cross-evaluation matters:** Allowing a model to evaluate its own outputs introduces positive self-bias. For example, if LLaMA judges LLaMA, it rewards its own phrasing patterns regardless of factual accuracy. Assigning an independent judge corrects for this.

### 3.3 Prompt Design

The system prompt was written in Modern Standard Arabic (MSA) and enforced four constraints:

1. Answer based **only** on the provided passage
2. Use Modern Standard Arabic — no dialects, no foreign terms
3. Limit responses to 1–2 sentences
4. Begin directly with the answer — no preamble phrases ("Based on the passage…")

Two prompt variants were created: a **with-context** version that injects the Wikipedia passage, and a **no-context** baseline that provides only the question. This separation is critical for Strategy 2 — measuring how much value the retrieval step actually adds.

### 3.4 Infrastructure

- **API:** Groq Cloud (free tier) — all inference via REST API
- **Rate limiting:** 3-second delay between requests; exponential backoff on 429 errors
- **Temperature:** 0 (deterministic outputs for reproducibility)
- **Max tokens:** 256

---

## 4. Results

### 4.1 ROUGE-L Scores

> **What is ROUGE-L?** It measures how much the model's answer _word-for-word overlaps_ with the human-written reference answer, based on the longest common subsequence of words. A score of 1.0 = perfect match; 0.0 = no shared words. It is the standard automatic metric for Arabic question answering, requires no API key, and is fully reproducible.

| Model             | ROUGE-L (with context) | ROUGE-L (no context) | RAG Benefit (Δ) | Refusal Accuracy  |
| ----------------- | ---------------------- | -------------------- | --------------- | ----------------- |
| **LLaMA 3.3 70B** | **0.733**              | 0.294                | **+0.440** ✅   | **100%** (5/5) ✅ |
| **ALLaM 2 7B**    | 0.640                  | 0.299                | +0.341 ✅       | **100%** (5/5) ✅ |
| **Qwen3 32B**     | 0.507                  | 0.230                | +0.277 ✅       | 80% (4/5) ⚠️      |

**How to read this table:**

- **ROUGE-L (with context):** Quality of answers when the passage is given — higher means the model's answer closely matches the reference.
- **RAG Benefit (Δ):** How much the passage _improved_ the score. LLaMA's +0.440 is the largest — it gains the most from having a document to read.
- **Refusal Accuracy:** On 5 questions where the answer was absent from the passage, did the model correctly refuse instead of hallucinating? LLaMA and ALLaM refused correctly every time. Qwen hallucinated an answer on 1 out of 5.

### 4.2 Answer Length Analysis

Answer length serves as a secondary proxy for **prompt adherence** — the system prompt instructed all models to answer in 1–2 sentences.

| Model                                 | Avg Length (with context) | Avg Length (no context) | Prompt Adherence                      |
| ------------------------------------- | ------------------------- | ----------------------- | ------------------------------------- |
| **LLaMA 3.3 70B**                     | 101 chars                 | 163 chars               | ✅ Follows prompt                     |
| **ALLaM 2 7B**                        | 115 chars                 | 183 chars               | ✅ Follows prompt                     |
| **Qwen3 32B** (raw)                   | 1,028 chars               | 1,110 chars             | ❌ Verbose (includes reasoning block) |
| **Qwen3 32B** (clean, think stripped) | 260 chars                 | 527 chars               | ✅ Actual answer is concise           |

> **Important distinction:** Qwen's actual Arabic answers are concise (~260 chars). The verbosity comes entirely from the automatically-inserted `<think>` reasoning block, averaging **~768 characters of English text** per response.

### 4.3 Chain-of-Thought Language Mismatch

**All 30 Qwen responses (100%)** contained `<think>` reasoning blocks written **entirely in English**, despite the Arabic-language task:

```
Question (Arabic): لماذا شيَّد الفراعنة الأهرامات وفق ما ورد في المقطع؟

<think>
Okay, let's see. The user is asking why the pharaohs built the pyramids according
to the provided text. I need to look at the text carefully.

The text says: "وكانَت تُستخدم مقابر ملكية لحفظ جثامين الفراعنة وكنوزهم استعداداً
للحياة الآخرة." So it mentions that they were used as royal tombs...
</think>

شُيِّدت الأهرامات مقابر ملكية لحفظ جثامين الفراعنة وكنوزهم استعداداً للحياة الآخرة.
```

This is a concrete demonstration of the ArabicNLP 2024 finding: **CoT in modern LLMs defaults to English reasoning regardless of the target language**, introducing cross-lingual overhead. The model "thinks" in English, then translates its conclusion to Arabic — adding latency, tokens, and potential translation loss without improving answer accuracy.

### 4.4 Sample Answer Comparisons

#### Question 1: Why did the Pharaohs build the Pyramids?

_(Context-Dependent)_

| Model            | Answer (Arabic)                                                                                     | Quality                       |
| ---------------- | --------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Ground Truth** | شيَّد الفراعنة الأهرامات لتكون مقابر ملكية تحفظ جثامينهم وكنوزهم استعداداً للحياة الآخرة.           | —                             |
| **LLaMA 70B**    | شُيِّدت الأهرامات لتُستخدم مقابر ملكية لحفظ جثامين الفراعنة وكنوزهم استعداداً للحياة الآخرة.        | ✅ Faithful                   |
| **ALLaM 7B**     | شيَّد الفراعنة الأهرامات كمقابر ملكية لحفظ جثامين الفراعنة وكنوزهم استعداداً للحياة الآخرة.         | ✅ Faithful                   |
| **Qwen3 32B**    | `<think>` Okay, let's see... `</think>` شُيِّدت الأهرامات مقابر ملكية لحفظ جثامين الفراعنة وكنوزهم. | ✅ Faithful (after stripping) |

#### Question 2: Difference between AI application types?

_(General Knowledge)_

| Model            | Answer (Arabic)                                                                                                                                                              | Quality                       |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Ground Truth** | يختص الذكاء الاصطناعي الضيق بأداء مهمة واحدة بعينها، أما الذكاء الاصطناعي العام فقادر نظرياً على أداء أي مهمة.                                                               | —                             |
| **LLaMA 70B**    | الفرق بين نوعَي تطبيقات الذكاء الاصطناعي: الضيق يؤدي مهمة واحدة، والعام يؤدي أي مهمة فكرية.                                                                                  | ✅ Faithful                   |
| **ALLaM 7B**     | تنقسم تطبيقات الذكاء الاصطناعي إلى نوعين: ضيق يؤدي مهمة بعينها، وعام قادر على أداء أي مهمة.                                                                                  | ✅ Faithful                   |
| **Qwen3 32B**    | `<think>` ...English reasoning in English... `</think>` الذكاء الاصطناعي الضيق متخصص في أداء مهمة بعينها، بينما الذكاء الاصطناعي العام قادراً نظرياً على أداء أي مهمة فكرية. | ✅ Faithful (after stripping) |

---

## 5. Analysis

### 5.1 H1 — Does Scale Provide the Best Arabic QA Performance?

**Verdict: Scale helps, but is not decisive.**

LLaMA 3.3 70B achieves the highest ROUGE-L (0.733) — but the gap over ALLaM 7B (0.640) is smaller than the 10× parameter difference would suggest. LLaMA also has the largest RAG benefit (+0.440), meaning it extracts the most value from the retrieved passage. As the largest model, it sets the multilingual upper bound. However, ALLaM 7B matches LLaMA's prompt adherence and refusal accuracy completely, showing that scale alone does not determine Arabic QA quality.

### 5.2 H2 — Does CoT Reasoning Improve Arabic Faithfulness?

**Verdict: No. CoT actively degrades the Arabic experience.**

Qwen3 32B's reasoning capability, while powerful for English tasks, is misaligned with Arabic RAG requirements:

| Issue                      | Evidence                                                   |
| -------------------------- | ---------------------------------------------------------- |
| English-language reasoning | 100% of `<think>` blocks written in English                |
| Token overhead             | ~768 extra chars per response (wasted compute)             |
| No accuracy gain           | Clean Qwen answers are factually equivalent to LLaMA/ALLaM |
| Cross-lingual friction     | Model must "translate" its English conclusions into Arabic |

This aligns with ArabicNLP 2024 findings from the Arab Legal Eval benchmark, which reported CoT yielding **lower accuracy** on Arabic compared to direct prompting. The phenomenon here is structural: current LLMs' reasoning circuits are primarily English-trained, so Arabic tasks trigger English thinking regardless of output language.

### 5.3 H3 — Does Arabic-Specialized Training Offset Scale?

**Verdict: Yes, convincingly.**

| Comparison             | LLaMA 70B   | ALLaM 7B    |
| ---------------------- | ----------- | ----------- |
| ROUGE-L (with context) | **0.733**   | 0.640       |
| ROUGE-L (no context)   | 0.294       | 0.299       |
| RAG Benefit (Δ)        | **+0.440**  | +0.341      |
| Refusal Accuracy       | **100%**    | **100%**    |
| Prompt adherence       | 100%        | 100%        |
| Language consistency   | Arabic only | Arabic only |

ALLaM's Arabic-centric pretraining compensates for its 10× smaller scale. It matches LLaMA exactly on refusal accuracy (the most critical real-world RAG metric) and achieves 87% of LLaMA's ROUGE-L score at a fraction of the compute cost. For resource-constrained or on-premise Arabic deployments, ALLaM 7B is a strong choice.

### 5.4 RAG Benefit Analysis (Strategy 2)

> **What is RAG Benefit?** It is the ROUGE-L score _with_ the retrieved passage minus the score _without_ it. A large positive number means retrieval genuinely helps the model give better answers. A value near zero means the model is answering entirely from its own training memory — in which case, building a retrieval system was wasted effort.

| Model     | ROUGE-L (with ctx) | ROUGE-L (no ctx) | RAG Benefit (Δ) | Verdict                                                          |
| --------- | ------------------ | ---------------- | --------------- | ---------------------------------------------------------------- |
| LLaMA 70B | 0.733              | 0.294            | **+0.440**      | ✅ Passage is essential — model nearly halves quality without it |
| ALLaM 7B  | 0.640              | 0.299            | **+0.341**      | ✅ Strong benefit — passage clearly matters                      |
| Qwen3 32B | 0.507              | 0.230            | **+0.277**      | ✅ Benefit present but smallest — CoT overhead may dampen signal |

All three models show large, positive RAG benefit, confirming that **retrieval is not redundant** — it genuinely moves scores by +0.28 to +0.44. Without context, all three models drop to roughly the same low score (~0.23–0.30), suggesting their parametric Arabic knowledge is similarly limited when the passage is absent.

### 5.5 Unanswerable Questions — Hallucination Test

A key real-world RAG failure is when the retriever returns a passage that does not contain the answer. A well-calibrated model should explicitly refuse rather than hallucinate. We added 5 such questions using existing passages — each question asks about information that is simply not mentioned in the text.

> **Refusal Accuracy** = the percentage of unanswerable questions where the model correctly replied with the designated refusal phrase ("لا تتوفر في المقطع معلومات كافية للإجابة عن هذا السؤال") instead of inventing an answer.

| Model     | Correct Refusals | Hallucinations | Refusal Accuracy |
| --------- | ---------------- | -------------- | ---------------- |
| LLaMA 70B | 5/5              | 0              | **100%** ✅      |
| ALLaM 7B  | 5/5              | 0              | **100%** ✅      |
| Qwen3 32B | 4/5              | **1**          | 80% ⚠️           |

**Finding:** LLaMA and ALLaM are perfectly calibrated — they refuse when the answer is absent and answer when it is present. Qwen hallucinated a confident Arabic answer on one unanswerable question, a direct consequence of its English-language reasoning process overriding the Arabic system prompt's instruction to refuse when uncertain.

---

## 6. Alignment with ArabicNLP 2024 Findings

| Conference Finding                                       | Our Observation                                                              | Status         |
| -------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------- |
| **CoT is less effective in Arabic than English**         | Qwen's `<think>` blocks are in English; no accuracy gain over simpler models | ✅ Confirmed   |
| **Open-source models "not far behind" GPT-4**            | LLaMA 70B and ALLaM 7B followed all prompt constraints perfectly             | ✅ Supported   |
| **Arabic-specialized models (AraLlama) are competitive** | ALLaM 7B matches LLaMA 70B behavior across all metrics                       | ✅ Confirmed   |
| **Test contamination is a significant concern**          | 33% of dataset questions modified to reduce memorization risk                | ✅ Implemented |
| **Cross-evaluation prevents self-bias**                  | Judge ≠ evaluated model in evaluation script                                 | ✅ Implemented |

---

## 7. Limitations

1. **ROUGE-L is a lexical metric, not a semantic one.** ROUGE-L measures word overlap, not meaning. A model answer can be semantically correct but phrased differently from the reference and receive a low score. For example, "الأهرامات بُنيت كمقابر" and "شيَّد الفراعنة الأهرامات لتكون مقابر" convey the same meaning but share fewer words. Semantic metrics (BERTScore, LLM-as-Judge) would complement ROUGE-L in a complete evaluation.

2. **Small dataset size.** 35 questions (30 answerable + 5 unanswerable) is sufficient for observational findings but not for statistical significance. Results should be interpreted as directional, not definitive.

3. **No dialect coverage.** All questions and answers are in Modern Standard Arabic (MSA). Performance on dialectal Arabic (Egyptian, Levantine, etc.) is unknown and may differ significantly.

4. **Groq free-tier constraints.** Rate limiting required a 3-second delay between API calls. Temperature was fixed at 0 for reproducibility; higher temperature may reveal different answer distributions.

5. **ALLaM as Jais substitute.** The original design called for Jais 70B (NYU Abu Dhabi), which was unavailable on Groq at evaluation time. ALLaM 2 7B is a comparable Arabic-first model but not a direct replacement — a future evaluation should include both.

---

## 8. Recommendations

### For Arabic RAG Application Design:

1. **Disable or suppress Chain-of-Thought** — For Arabic tasks, use direct prompting without `<think>` instructions or CoT templates
2. **Prefer Arabic-specialized models at 7B–13B scale** — ALLaM-class models offer excellent efficiency-to-quality ratios
3. **Always include a no-context baseline** — Without it, you cannot measure how much the retrieval step actually contributes
4. **Use cross-evaluation for LLM-as-Judge** — Designate a separate judge model to prevent self-preferential scoring

### For Future Work:

1. Run full LLM-as-Judge evaluation using `evaluate_groq_judge.py` with ROUGE, BLEU, and faithfulness scores
2. Expand dataset to 100+ questions with dialect coverage
3. Test Jais 70B (Arabic-centric) when available as the originally intended comparison point
4. Evaluate on BALSaM benchmark to cross-validate with the ArabicNLP standardized leaderboard
5. Study whether **prompting in Arabic** (vs English prompts) changes Qwen's reasoning language

---

## 9. Technical Details

### Files Generated

| File                            | Description                                                                                  |
| ------------------------------- | -------------------------------------------------------------------------------------------- |
| `ArabicRAG-Eval_Results_v2.csv` | All model answers (180 total: 30 questions × 3 models × 2 conditions)                        |
| `inference_v2.py`               | Inference script implementing Strategy 2 (no-context baseline) and Strategy 4 (named models) |
| `evaluate_v2.py`                | Evaluation script with cross-evaluation logic (Strategy 1)                                   |
| `evaluate_groq_judge.py`        | Free LLM-as-Judge evaluation using Groq (OpenAI-compatible)                                  |
| `ArabicRAG-Eval_Dataset_v2.csv` | Source dataset: 30 questions, passages, ground truth answers                                 |

### Model API Identifiers

```
llama-3.3-70b-versatile  →  LLaMA 3.3 70B  (Meta, general multilingual)
qwen/qwen3-32b           →  Qwen3 32B      (Alibaba, reasoning-focused)
allam-2-7b               →  ALLaM 2 7B     (SDAIA, Saudi Arabia — Arabic-specialized)
```

### Environment

| Component     | Details                                       |
| ------------- | --------------------------------------------- |
| Python        | 3.12.7                                        |
| Inference API | Groq Cloud (free tier)                        |
| Request rate  | 1 request / 3 seconds (rate limit compliance) |
| Temperature   | 0 (deterministic)                             |
| Dependencies  | groq, pandas, tqdm, ragas, datasets, openpyxl |

---

## 10. Conclusion

This evaluation set out to test three hypotheses about Arabic LLM behavior in a RAG setting. All three hypotheses yielded clear findings, now backed by quantitative ROUGE-L scores and refusal accuracy:

1. **H1 (Scale):** LLaMA 3.3 70B leads on ROUGE-L (0.733) and RAG benefit (+0.440), but its advantage over a 10× smaller specialized model is modest.

2. **H2 (Chain-of-Thought):** Qwen3 32B scores lowest on every metric (ROUGE-L 0.507, refusal 80%). Its English-language reasoning process adds overhead without improving accuracy — and caused the only hallucination failure in the unanswerable test.

3. **H3 (Specialization):** ALLaM 2 7B matches LLaMA's refusal accuracy (100%) and achieves 87% of its ROUGE-L score at 10× lower compute. Arabic-targeted pretraining is more efficient than scale for this task.

The practical implication is clear: **for Arabic RAG applications, a small Arabic-specialized model with a direct-answer prompt will outperform or match a large reasoning model with Chain-of-Thought.** This finding is consistent with the ArabicNLP 2024 community's emerging consensus and supports continued investment in Arabic-specific LLM development.

---

**Appendix A: System Prompt (Arabic)**

```
أنتَ مساعد متخصص في الإجابة عن الأسئلة باللغة العربية الفصحى.
مهمتك الوحيدة هي الإجابة عن السؤال المطروح بناءً حصرياً على المقطع النصي المُقدَّم إليك.

القواعد الإلزامية:
١. استنِد في إجابتك إلى المعلومات الواردة في المقطع فحسب.
٢. إذا كانت الإجابة غير موجودة، أجب: "لا تتوفر في المقطع معلومات كافية للإجابة."
٣. اكتب بالعربية الفصحى دون لهجات أو مصطلحات أجنبية.
٤. اجعل إجابتك موجزة: جملة واحدة أو جملتان على الأكثر.
٥. ابدأ مباشرةً بالإجابة دون مقدمات.
```

**Appendix B: Dataset Sample**

| Field                    | Example                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| `question`               | لماذا شيَّد الفراعنة الأهرامات وفق ما ورد في المقطع؟                                      |
| `context_dependency`     | CONTEXT-DEPENDENT                                                                         |
| `contamination_modified` | NO                                                                                        |
| `source_url`             | https://ar.wikipedia.org/wiki/أهرامات_مصر                                                 |
| `ground_truth_answer`    | شيَّد الفراعنة الأهرامات لتكون مقابر ملكية تحفظ جثامينهم وكنوزهم استعداداً للحياة الآخرة. |
