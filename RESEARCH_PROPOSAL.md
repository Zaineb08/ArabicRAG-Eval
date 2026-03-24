# Faithfulness, Hallucination, and Specialization:

# Evaluating Large Language Models for Arabic Retrieval-Augmented Generation

---

**Team:** Data Resources & Benchmarking — Team 2  
**Members:** Zaineb Rahmani · Abderrahmane Jouilili · Moumni Mohammed · Wissal Said · Abdellah Hasnaoui · Soukaina Assam  
**Mentors:** Go Inoue · Hamdy Mubarak  
**Venue:** Arabic NLP School 2026 · EACL 2026 · Rabat, Morocco  
**Date:** March 24, 2026

---

## Abstract

Retrieval-Augmented Generation (RAG) is rapidly becoming the dominant architecture for deploying Large Language Models (LLMs) in real-world applications — from legal research tools to medical question answering systems. Yet rigorous evaluation of RAG systems in Arabic remains scarce. This proposal presents **ArabicRAG-Eval**, a benchmark and evaluation framework designed to measure three critical properties of Arabic RAG systems: _faithfulness_ to retrieved passages, _hallucination resistance_ when the answer is absent, and the _genuine benefit_ provided by the retrieval step. Through a systematic comparison of three models — LLaMA 3.3 70B, Qwen3 32B, and ALLaM 2 7B — our preliminary results reveal that Arabic-specialized models can match much larger multilingual models, and that Chain-of-Thought reasoning, despite its success in English, is counterproductive for Arabic generation tasks. We propose extending this framework to a larger dataset, additional models, and dialectal Arabic coverage.

---

## 1. Problem Statement

Arabic is spoken by over 400 million people across 27 countries, yet it remains severely under-resourced in the context of modern NLP benchmarking. The rise of Retrieval-Augmented Generation — where a model receives a retrieved document alongside a query before generating a response — has created a new and critical evaluation challenge: **how do we know if an Arabic LLM is actually reading the document, or just answering from memory?**

This distinction matters enormously in high-stakes applications. A legal assistant that ignores the retrieved contract, a medical chatbot that fabricates a drug dosage, or a customer service tool that confabulates a policy — all of these are RAG failure modes that no existing Arabic benchmark adequately measures.

Three specific gaps motivate this research:

1. **No Arabic RAG benchmark** tests _faithfulness_ (does the answer come from the passage?) alongside _refusal_ (does the model say "I don't know" when it should?)
2. **No systematic comparison** of scale vs. specialization for Arabic RAG — i.e., does a 70B multilingual model outperform a 7B Arabic-dedicated model?
3. **No empirical testing** of whether Chain-of-Thought prompting helps or hurts Arabic generation quality, despite contradictory claims in the literature.

---

## 2. Background and Motivation

### 2.1 The RAG Paradigm

Standard LLMs answer questions purely from _parametric memory_ — knowledge encoded in model weights during training. RAG augments this by providing a retrieved text passage at inference time, enabling the model to answer questions about documents it has never seen. This makes RAG essential for up-to-date, domain-specific, and private-document applications.

The core evaluation challenge in RAG is twofold:

- **Does the model use the passage?** (faithfulness + RAG benefit)
- **Does the model know when the passage is insufficient?** (refusal accuracy / hallucination resistance)

### 2.2 The Arabic NLP Gap

Arabic presents unique challenges for LLM evaluation:

- **Morphological richness:** A single Arabic root can produce hundreds of surface forms, making exact-match metrics unreliable
- **Diglossia:** Modern Standard Arabic (MSA) and spoken dialects (Egyptian, Gulf, Levantine) are linguistically distinct — model behavior differs significantly between registers
- **Training data imbalance:** Most LLMs are predominantly trained on English text, raising questions about whether their reasoning capabilities transfer to Arabic

The ArabicNLP 2024 conference (2nd International Conference on Arabic NLP, December 2024) highlighted several findings directly relevant to this work: Chain-of-Thought prompting is _less effective_ in Arabic than in English; Arabic-specialized models are competitive with larger multilingual ones; and test contamination is a pervasive concern that demands modified evaluation sets.

### 2.3 Why This Matters Now

The rapid deployment of Arabic-language AI systems across the Arab world — in government, education, healthcare, and commerce — makes rigorous RAG evaluation urgent. Without a standardized benchmark, practitioners cannot make informed model selection decisions, and researchers cannot track progress.

---

## 3. Preliminary Work: ArabicRAG-Eval v2

We have constructed and evaluated an initial version of the ArabicRAG-Eval benchmark.

### 3.1 Dataset

**ArabicRAG-Eval Dataset v2** consists of 35 Arabic question-passage-answer triplets, all sourced from Arabic Wikipedia:

| Category               | Count | Description                                                   |
| ---------------------- | ----- | ------------------------------------------------------------- |
| Context-Dependent      | 17    | Answer exists only in the retrieved passage                   |
| General Knowledge      | 13    | Answer may exist in the model's parametric memory             |
| **Unanswerable**       | **5** | Answer is deliberately absent; correct behavior is to refuse  |
| Contamination-Modified | 10    | Facts changed (dates, numbers) to detect memory vs. retrieval |

### 3.2 Models Evaluated

| Model         | Parameters | Role in Study                                          |
| ------------- | ---------- | ------------------------------------------------------ |
| LLaMA 3.3 70B | 70B        | Multilingual upper bound — does scale dominate?        |
| Qwen3 32B     | 32B        | Reasoning-focused — does Chain-of-Thought help Arabic? |
| ALLaM 2 7B    | 7B         | Arabic-specialized — does pretraining offset scale?    |

### 3.3 Evaluation Metrics

- **ROUGE-L**: Longest common subsequence word overlap with ground truth (0–1 scale)
- **RAG Benefit (Δ)**: ROUGE-L _with context_ minus ROUGE-L _without context_
- **Refusal Accuracy**: On unanswerable questions, did the model correctly decline to answer?

### 3.4 Key Results

| Model             | ROUGE-L (with ctx) | RAG Benefit (Δ) | Refusal Accuracy |
| ----------------- | ------------------ | --------------- | ---------------- |
| **LLaMA 3.3 70B** | **0.733**          | **+0.440**      | **100%** (5/5)   |
| **ALLaM 2 7B**    | 0.640              | +0.341          | **100%** (5/5)   |
| **Qwen3 32B**     | 0.507              | +0.277          | 80% (4/5)        |

### 3.5 Preliminary Findings

**Finding 1 — Chain-of-Thought is counterproductive for Arabic RAG.**  
Qwen3 32B produced `<think>` reasoning blocks in _all 30 answerable responses_, every one written in English despite the Arabic task. This cross-lingual thinking adds ~768 characters of overhead per response and yields the _lowest_ ROUGE-L and refusal accuracy of the three models. This directly confirms the ArabicNLP 2024 finding on CoT ineffectiveness for Arabic.

**Finding 2 — Arabic specialization compensates for scale.**  
ALLaM 2 7B, despite having 10× fewer parameters than LLaMA 70B, achieves 87% of LLaMA's ROUGE-L score and identical refusal accuracy (100%). Language-targeted pretraining is demonstrably more efficient than raw scale for this task.

**Finding 3 — RAG is genuinely beneficial, but the margin depends on the model.**  
All three models show large positive RAG benefit (+0.28 to +0.44). Without context, all drop to approximately the same low score (~0.23–0.30), confirming that parametric Arabic knowledge is similarly limited across models and that the retrieval step provides real, measurable value.

**Finding 4 — Unanswerable questions expose hallucination.**  
Qwen hallucinated a confident Arabic answer to one unanswerable question while LLaMA and ALLaM refused correctly every time. Hallucination resistance is therefore not guaranteed by model size and must be explicitly tested.

---

## 4. Proposed Research

Building on these preliminary results, we propose three extensions:

### 4.1 Expand the Benchmark (ArabicRAG-Eval v3)

The current 35-question dataset is sufficient for directional findings but not for statistical claims. We propose:

- **Scale to 200 questions** across 10 topic domains (law, medicine, history, science, religion, economics, literature, technology, environment, politics)
- **Add unanswerable questions** for each domain (target: 20% of total, ~40 questions)
- **Include dialectal passages** — Egyptian Arabic, Gulf Arabic, Levantine Arabic — to measure MSA vs. dialect performance gaps
- **Add multi-passage retrieval** — provide 3 passages per question, only one containing the answer, to test distractor resistance

### 4.2 Broaden Model Coverage

| Planned Model            | Rationale                                                    |
| ------------------------ | ------------------------------------------------------------ |
| Jais 70B (NYU Abu Dhabi) | Original target; unavailable at evaluation time — retry      |
| AraLlama                 | Confirmed competitive in storytelling tasks (ArabicNLP 2024) |
| GPT-4o (mini)            | Establishes closed-source ceiling                            |
| Command R+ (Arabic)      | Document-focused model; ideal RAG comparison                 |

### 4.3 Develop Semantic Evaluation

ROUGE-L measures word overlap, not meaning. An answer can be semantically correct but phrased differently and score low. We propose implementing:

- **BERTScore** using Arabic BERT (CAMeL-BERT or AraBERT) for semantic similarity
- **LLM-as-Judge** using the cross-evaluation paradigm (Strategy 1) — judge model is always different from the tested model
- **Human evaluation** on a 50-question gold set, with 3 annotators rating faithfulness on a 1–3 scale, to validate the automatic metrics

---

## 5. Expected Contributions

| Contribution                                             | Impact                                                      |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| First Arabic RAG benchmark with unanswerable questions   | Fills a critical gap in Arabic NLP evaluation tooling       |
| Empirical confirmation of CoT ineffectiveness for Arabic | Actionable guidance for Arabic LLM deployment practitioners |
| Scale vs. specialization analysis across 5+ models       | Informs model selection for Arabic RAG products             |
| Open evaluation suite (dataset + code + results)         | Enables reproducible research and community benchmarking    |

All code, data, and results will be released publicly under a permissive open-source license.

---

## 6. Relevance to the Arabic NLP Community

This work directly addresses priorities identified at ArabicNLP 2024:

- The BALSaM project's call for contamination-resistant Arabic benchmarks
- The community's need for standardized RAG evaluation to track progress across years
- The practical need for model selection guidance as Arabic AI deployments accelerate

We position ArabicRAG-Eval as a living benchmark — one that can be extended by the community, integrated into leaderboards, and versioned annually alongside the ArabicNLP conference.

---

## 7. References

1. Antoun, W. et al. (2020). _AraBERT: Transformer-based Model for Arabic Language Understanding_. LREC 2020.
2. Inoue, G. et al. (2021). _Interplay of Dialect and Standard Language in Arabic NLP_. ACL-IJCNLP 2021.
3. Touvron, H. et al. (2023). _LLaMA: Open and Efficient Foundation Language Models_. arXiv.
4. Qwen Team, Alibaba (2025). _Qwen3 Technical Report_. arXiv.
5. SDAIA (2025). _ALLaM: Large Arabic Language Model_. Technical Report.
6. Lewis, P. et al. (2020). _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_. NeurIPS 2020.
7. ArabicNLP 2024 Proceedings. _2nd International Conference on Arabic NLP_. ACL Anthology.
8. Es, S. et al. (2023). _RAGAS: Automated Evaluation of Retrieval Augmented Generation_. arXiv.

---

---

# Fidélité, Hallucination et Spécialisation :

# Évaluation des Grands Modèles de Langage pour la Génération Augmentée par Récupération en Arabe

---

**Équipe :** Ressources de Données et Évaluation — Équipe 2  
**Membres :** Zaineb Rahmani · Abderrahmane Jouilili · Moumni Mohammed · Wissal Said · Abdellah Hasnaoui · Soukaina Assam  
**Encadrants :** Go Inoue · Hamdy Mubarak  
**Événement :** École Arabe de TAL 2026 · EACL 2026 · Rabat, Maroc  
**Date :** 24 mars 2026

---

## Résumé

La Génération Augmentée par Récupération (RAG — _Retrieval-Augmented Generation_) s'est imposée comme l'architecture dominante pour le déploiement des Grands Modèles de Langage (LLM) dans des applications réelles — des outils de recherche juridique aux systèmes de questions-réponses médicales. Pourtant, l'évaluation rigoureuse des systèmes RAG en arabe reste insuffisante. Cette proposition présente **ArabicRAG-Eval**, un référentiel d'évaluation (_benchmark_) conçu pour mesurer trois propriétés essentielles des systèmes RAG arabes : la _fidélité_ aux passages récupérés, la _résistance aux hallucinations_ lorsque la réponse est absente, et le _bénéfice réel_ apporté par l'étape de récupération. À travers une comparaison systématique de trois modèles — LLaMA 3.3 70B, Qwen3 32B et ALLaM 2 7B — nos résultats préliminaires montrent que les modèles spécialisés en arabe peuvent rivaliser avec des modèles multilingues bien plus grands, et que le raisonnement par Chaîne de Pensée (_Chain-of-Thought_), malgré son efficacité en anglais, est contre-productif pour les tâches de génération en arabe.

---

## 1. Problématique

L'arabe est parlé par plus de 400 millions de personnes dans 27 pays, mais il reste une langue sous-représentée dans l'évaluation des systèmes NLP modernes. L'essor de la Génération Augmentée par Récupération a créé un nouveau défi d'évaluation critique : **comment savoir si un LLM arabe lit réellement le document fourni, ou s'il répond simplement de mémoire ?**

Cette distinction est fondamentale dans les applications à enjeux élevés. Un assistant juridique qui ignore le contrat récupéré, un chatbot médical qui invente une posologie, ou un service client qui fabrique une politique d'entreprise — tous constituent des échecs du RAG qu'aucun référentiel arabe existant ne mesure adéquatement.

Trois lacunes spécifiques motivent cette recherche :

1. **Aucun référentiel RAG arabe** ne teste simultanément la _fidélité_ (la réponse provient-elle du passage ?) et le _refus_ (le modèle dit-il « je ne sais pas » quand il le devrait ?)
2. **Aucune comparaison systématique** de la taille versus la spécialisation pour le RAG arabe
3. **Aucun test empirique** de l'effet du raisonnement par Chaîne de Pensée sur la qualité de génération en arabe

---

## 2. Contexte et Motivation

### 2.1 Le Paradigme RAG

Les LLM standard répondent aux questions uniquement depuis leur _mémoire paramétrique_ — les connaissances encodées dans les poids du modèle lors de l'entraînement. Le RAG enrichit cette capacité en fournissant un passage texte récupéré au moment de l'inférence. Le défi central d'évaluation est double :

- **Le modèle utilise-t-il le passage ?** (fidélité + bénéfice RAG)
- **Sait-il reconnaître quand le passage est insuffisant ?** (résistance aux hallucinations)

### 2.2 Les Défis Spécifiques de l'Arabe

L'arabe présente des obstacles particuliers pour l'évaluation :

- **Richesse morphologique :** une même racine peut générer des centaines de formes de surface, rendant les métriques de correspondance exacte peu fiables
- **Diglossie :** l'arabe littéral (MSA) et les dialectes parlés (égyptien, du Golfe, levantin) sont linguistiquement distincts — le comportement des modèles diffère significativement selon le registre
- **Déséquilibre des données d'entraînement :** la plupart des LLM sont entraînés majoritairement sur des textes anglais

La conférence ArabicNLP 2024 a mis en évidence des résultats directement pertinents : le raisonnement par Chaîne de Pensée est _moins efficace_ en arabe qu'en anglais ; les modèles spécialisés en arabe sont compétitifs face aux modèles multilingues plus grands ; et la contamination des données de test est une préoccupation majeure exigeant des jeux d'évaluation modifiés.

---

## 3. Travaux Préliminaires : ArabicRAG-Eval v2

Nous avons construit et évalué une première version du référentiel ArabicRAG-Eval.

### 3.1 Jeu de Données

**ArabicRAG-Eval Dataset v2** comprend 35 triplets question-passage-réponse en arabe, issus de Wikipédia arabe :

| Catégorie                              | Nombre | Description                                                                  |
| -------------------------------------- | ------ | ---------------------------------------------------------------------------- |
| Dépendant du contexte                  | 17     | La réponse n'existe que dans le passage récupéré                             |
| Connaissance générale                  | 13     | La réponse peut exister dans la mémoire paramétrique                         |
| **Sans réponse (_Unanswerable_)**      | **5**  | La réponse est délibérément absente — le comportement correct est de refuser |
| Passages modifiés (anti-contamination) | 10     | Faits modifiés (dates, chiffres) pour détecter mémoire vs. récupération      |

### 3.2 Modèles Évalués

| Modèle        | Paramètres | Rôle dans l'étude                                                                         |
| ------------- | ---------- | ----------------------------------------------------------------------------------------- |
| LLaMA 3.3 70B | 70B        | Référence multilingue — la taille domine-t-elle ?                                         |
| Qwen3 32B     | 32B        | Orienté raisonnement — la Chaîne de Pensée aide-t-elle l'arabe ?                          |
| ALLaM 2 7B    | 7B         | Spécialisé arabe (SDAIA, Arabie Saoudite) — le pré-entraînement compense-t-il la taille ? |

### 3.3 Métriques d'Évaluation

- **ROUGE-L** : chevauchement de mots par plus longue sous-séquence commune avec la réponse de référence (échelle 0–1)
- **Bénéfice RAG (Δ)** : ROUGE-L _avec contexte_ moins ROUGE-L _sans contexte_
- **Précision de refus** : sur les questions sans réponse, le modèle a-t-il correctement refusé de répondre ?

### 3.4 Résultats Clés

| Modèle            | ROUGE-L (avec contexte) | Bénéfice RAG (Δ) | Précision de refus |
| ----------------- | ----------------------- | ---------------- | ------------------ |
| **LLaMA 3.3 70B** | **0,733**               | **+0,440**       | **100 %** (5/5)    |
| **ALLaM 2 7B**    | 0,640                   | +0,341           | **100 %** (5/5)    |
| **Qwen3 32B**     | 0,507                   | +0,277           | 80 % (4/5)         |

### 3.5 Principales Conclusions

**Conclusion 1 — La Chaîne de Pensée est contre-productive pour le RAG arabe.**  
Qwen3 32B a produit des blocs de raisonnement `<think>` dans _toutes les 30 réponses_, chacun rédigé intégralement en anglais malgré la tâche en arabe. Ce raisonnement en langue croisée ajoute en moyenne 768 caractères par réponse sans améliorer la précision — et produit le ROUGE-L _le plus faible_ et la précision de refus la plus basse des trois modèles.

**Conclusion 2 — La spécialisation arabe compense la taille.**  
ALLaM 2 7B, avec 10× moins de paramètres que LLaMA 70B, atteint 87 % de son score ROUGE-L et une précision de refus identique (100 %). Le pré-entraînement ciblé sur l'arabe est plus efficient que la simple augmentation de la taille du modèle.

**Conclusion 3 — Le RAG apporte un bénéfice réel et mesurable.**  
Les trois modèles affichent un bénéfice RAG fortement positif (+0,28 à +0,44). Sans contexte, tous tombent à un score faible similaire (~0,23–0,30), confirmant que les connaissances arabes paramétriques sont insuffisantes et que l'étape de récupération a une valeur réelle.

**Conclusion 4 — Les questions sans réponse révèlent les hallucinations.**  
Qwen a inventé une réponse arabe confiante à une question sans réponse, tandis que LLaMA et ALLaM ont correctement refusé dans tous les cas. La résistance aux hallucinations doit être testée explicitement — elle n'est pas garantie par la taille du modèle.

---

## 4. Recherches Proposées

### 4.1 Élargir le Référentiel (ArabicRAG-Eval v3)

Nous proposons de passer de 35 à **200 questions** couvrant 10 domaines thématiques (droit, médecine, histoire, science, religion, économie, littérature, technologie, environnement, politique), avec :

- 20 % de questions sans réponse (~40 questions)
- Des passages en dialectes arabes (égyptien, du Golfe, levantin)
- Des scénarios multi-passages (3 passages par question, un seul contenant la réponse)

### 4.2 Élargir la Couverture des Modèles

| Modèle prévu             | Justification                                                |
| ------------------------ | ------------------------------------------------------------ |
| Jais 70B (NYU Abu Dhabi) | Cible originale, indisponible lors de l'évaluation initiale  |
| AraLlama                 | Compétitif sur les tâches narratives arabes (ArabicNLP 2024) |
| GPT-4o (mini)            | Établit le plafond des modèles propriétaires                 |
| Command R+ (arabe)       | Modèle orienté documents, idéal pour la comparaison RAG      |

### 4.3 Développer une Évaluation Sémantique

ROUGE-L mesure le chevauchement lexical, pas le sens. Nous proposons :

- **BERTScore** utilisant un BERT arabe (CAMeL-BERT ou AraBERT) pour la similarité sémantique
- **LLM comme juge** (_LLM-as-Judge_) avec évaluation croisée — le modèle juge est toujours différent du modèle évalué
- **Évaluation humaine** sur 50 questions avec 3 annotateurs évaluant la fidélité sur une échelle de 1 à 3

---

## 5. Contributions Attendues

| Contribution                                                | Impact                                                               |
| ----------------------------------------------------------- | -------------------------------------------------------------------- |
| Premier référentiel RAG arabe avec questions sans réponse   | Comble une lacune critique dans les outils d'évaluation du TAL arabe |
| Confirmation empirique de l'inefficacité de la CoT en arabe | Recommandation pratique pour le déploiement de LLM arabes            |
| Analyse taille vs. spécialisation sur 5+ modèles            | Aide au choix de modèles pour les produits RAG arabes                |
| Suite d'évaluation ouverte (données + code + résultats)     | Recherche reproductible et évaluation comparative par la communauté  |

Tout le code, les données et les résultats seront publiés sous licence open source permissive.

---

## 6. Pertinence pour la Communauté du TAL Arabe

Ces travaux répondent directement aux priorités identifiées lors d'ArabicNLP 2024 :

- L'appel du projet BALSaM pour des référentiels arabes résistants à la contamination
- Le besoin de la communauté en évaluation RAG standardisée pour suivre les progrès
- Le besoin pratique de critères de sélection de modèles face à l'accélération des déploiements d'IA arabes

Nous positionnons ArabicRAG-Eval comme un référentiel _vivant_ — extensible par la communauté, intégrable dans des classements (_leaderboards_), et mis à jour annuellement avec la conférence ArabicNLP.

---

## 7. Références

1. Antoun, W. et al. (2020). _AraBERT: Transformer-based Model for Arabic Language Understanding_. LREC 2020.
2. Inoue, G. et al. (2021). _Interplay of Dialect and Standard Language in Arabic NLP_. ACL-IJCNLP 2021.
3. Touvron, H. et al. (2023). _LLaMA: Open and Efficient Foundation Language Models_. arXiv.
4. Qwen Team, Alibaba (2025). _Qwen3 Technical Report_. arXiv.
5. SDAIA (2025). _ALLaM: Large Arabic Language Model_. Rapport Technique.
6. Lewis, P. et al. (2020). _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks_. NeurIPS 2020.
7. ArabicNLP 2024 Proceedings. _2nd International Conference on Arabic NLP_. ACL Anthology.
8. Es, S. et al. (2023). _RAGAS: Automated Evaluation of Retrieval Augmented Generation_. arXiv.
