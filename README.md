# Section-Scoped RAG System for SEC 10-K Filings (2024)

A Retrieval-Augmented Generation (RAG) pipeline that queries SEC 10-K annual reports to surface AI strategy and investment insights from companies' financial disclosures.

---

## Overview

This project builds an end-to-end RAG system scoped to three SEC 10-K items most relevant to AI strategy:

| SEC Item | Section | Why It Was Chosen |
|---|---|---|
| **Item 1** | Business | Describes the company's core operations and strategic direction |
| **Item 1A** | Risk Factors | Captures AI-related risks, regulatory concerns, and uncertainties |
| **Item 7** | MD&A | Management's Discussion & Analysis — strategic investments and financial outlook |

Focusing on these items keeps the vector index dense with strategic and risk-oriented language, preventing noise from boilerplate legal text, financial tables, and exhibits from polluting retrieval results.

---

## Pipeline Architecture

```
SEC 10-K Raw HTML Files (7,754 filings)
        │
        ▼
 Step 2: Section Extraction (Items 1, 1A, 7)
        │  regex-based boundary detection
        ▼
 Step 3: HTML Cleaning + Chunking
        │  BeautifulSoup → plain text
        │  400-word chunks, 50-word overlap
        ▼
 Step 4: Corpus Inspection
        │  39,271 chunks across 725 companies
        ▼
 Step 5: Embedding
        │  all-MiniLM-L6-v2 (384-dim, local, free)
        ▼
 Step 6: FAISS Vector Index
        │  IndexFlatIP (inner product = cosine similarity)
        ▼
 Step 7: Retrieval
        │  top_k = 5 chunks per query
        ▼
 Step 8: Reranking + Generation
        │  Cross-encoder reranker (ms-marco-TinyBERT-L-2)
        │  gpt-4o-mini via GitHub Models API
        ▼
 Step 9: Structured Query Evaluation (10 questions)
        │
        ▼
 Step 10: RAGAS Evaluation
        │  Faithfulness + Answer Relevancy
        ▼
 Step 11: Manual Evaluation & Failure Analysis
```

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, 384-dim — fast on CPU/GPU with strong English semantic similarity |
| **Vector Store** | FAISS `IndexFlatIP` | Exact inner product search; cosine similarity via L2-normalized embeddings |
| **Reranker** | `cross-encoder/ms-marco-TinyBERT-L-2` | Lightweight cross-encoder that re-scores retrieved chunks for precision |
| **Generation LLM** | `gpt-4o-mini` via GitHub Models API | Fast inference, strict instruction-following, cost-efficient |
| **Evaluation** | RAGAS | Automated faithfulness and answer relevancy scoring using LLM-as-a-judge |
| **Environment** | Google Colab + Google Drive | Persistent storage for embeddings, chunks, and results |

---

## Corpus Statistics

```
Total 10-K filings processed : 7,754
Extracted item files          : 1,400
Total chunks                  : 39,271
Unique companies              : 725
Average chunk size            : 393 tokens
Chunk size (min / max)        : 51 / 400 tokens

Section distribution:
  Item 1A  (Risk Factors) : 18,922 chunks
  Item 7   (MD&A)         : 11,724 chunks
  Item 1   (Business)     : 8,625  chunks
```

---

## Setup & Usage

### Prerequisites

```bash
pip install sentence-transformers faiss-cpu anthropic langchain langchain-community \
            pandas tqdm beautifulsoup4 openai langchain-openai ragas datasets
```

### Environment

This notebook is designed to run in **Google Colab** with **Google Drive** mounted for persistent storage.

1. Mount Google Drive and set your input/output paths:

```python
from google.colab import drive
drive.mount('/content/drive')

INPUT_DIR  = Path("/content/drive/MyDrive/Gen/SEC-10K-2024-TXT")
OUTPUT_DIR = Path("/content/drive/MyDrive/SEC-10K-2024-ITEMS")
```

2. Add your GitHub Models API key to Colab Secrets as `GITHUB_TOKEN`.

3. Run all cells in order (Steps 0–11).

---

## Retrieval Design Decisions

**Why `top_k = 5`?**
Retrieving 5 chunks provides approximately 2,000 words of context — enough for the LLM to synthesize a comprehensive answer without overflowing the prompt window or introducing irrelevant noise.

**Why filtered corpus (Items 1, 1A, 7 only)?**
Embedding entire 10-K documents would pollute the vector space with boilerplate legal text, financial tables, and exhibits. A narrow corpus ensures FAISS surfaces semantically relevant, AI-strategy-focused chunks.

**Why a cross-encoder reranker?**
The bi-encoder (MiniLM) retrieves candidates quickly via approximate similarity. The cross-encoder then re-scores the top-5 using full query-chunk interaction, improving final ranking precision before passing to the LLM.

---

## Evaluation Results

### RAGAS Scores (averaged across 10 questions)

| Metric | Score | Evaluator |
|---|---|---|
| **Faithfulness** | **0.85** | LLM only (gpt-4o-mini) |
| **Answer Relevancy** | **0.48** | LLM + Embeddings (text-embedding-3-small) |

**How each metric works:**

- **Faithfulness** — The LLM extracts all claims from the generated answer and cross-checks each against the retrieved context chunks. A score of 1.0 means every claim is supported; 0.0 means none are.
- **Answer Relevancy** — The LLM reverse-engineers potential questions from the generated answer; embeddings then score the semantic similarity between those and the original question. High similarity = the answer directly addressed the prompt.

### Per-Question Breakdown

| # | Question (abbreviated) | Faithfulness | Answer Relevancy |
|---|---|---|---|
| Q1 | AI as a strategic investment area? | 1.00 | 0.00 |
| Q2 | Operational efficiency vs. product innovation? | 0.50 | 0.69 |
| Q3 | AI-related expenditures / R&D quantified? | 1.00 | 0.51 |
| Q4 | Risks & uncertainties in AI adoption? | 1.00 | 0.98 |
| Q5 | Regulatory / data privacy concerns for AI? | 1.00 | 0.97 |
| Q6 | Does SIFCO Industries describe AI as strategic? | 0.00 | 0.00 |
| Q7 | AI as competitive advantage or threat? | 1.00 | ~0.87 |
| Q8 | Talent challenges for AI expertise? | 1.00 | 0.00 |
| Q9 | Ethical AI / responsible AI references? | 1.00 | ~0.79 |
| Q10 | Generative AI for customer experience? | 1.00 | ~0.72 |

### Key Findings

**Strengths:**
- Faithfulness scored 1.0 on 8 of 10 queries — generated answers are tightly grounded in retrieved SEC 10-K context with minimal hallucination.
- The system prompt correctly instructs the LLM to say "I cannot find a direct answer" when context is weak, preventing hallucination at the cost of answer relevancy.

**Limitations:**
- Answer relevancy drops to 0.0 on Q1, Q6, and Q8 — these are cases where retrieval failed to surface relevant context, causing the LLM to return a refusal rather than an answer.
- RAGAS penalizes broad questions answered with company-specific details, because the reverse-engineered question becomes too specific to match the original prompt.
- "I cannot find an answer" responses score inconsistently on faithfulness — the evaluator LLM sometimes treats the refusal itself as a factual claim and flags it as unsupported, returning 0.0 instead of 1.0.

---

## Failure Analysis

| Failure Case | What Happened | Future Fix |
|---|---|---|
| **Weak retrieval** | Top chunks had low similarity scores; LLM correctly refused to answer | Query expansion, hybrid dense/sparse search, stricter AI-domain pre-filtering |
| **Company-specific queries (e.g., Q6 — SIFCO)** | Pipeline does not route company metadata; semantic search returns irrelevant chunks | Metadata injection: prepend company name, ticker, and section to each chunk before embedding |
| **Low answer relevancy despite grounding** | Answer was faithful but non-responsive when retrieved context was off-topic | Tune `top_k` by question type; add keyword or metadata filtering layer |
| **Domain noise** | Some Items 1/1A/7 chunks only tangentially related to AI strategy | Add an AI-relevance pre-filter before embedding (e.g., keyword or classifier gate) |

---

## Design Trade-offs

### Local vs. API Models

| Dimension | Local Models (Llama-3, Mistral) | API Models (gpt-4o-mini) |
|---|---|---|
| **Data Privacy** | Stays on-premise — preferred for sensitive financial data | Sends data to external servers — potential compliance risk |
| **Compute** | Requires GPU (T4/A100); risk of OOM errors | Runs on CPU-only Colab instance |
| **Instruction Following** | Smaller models may ignore strict grounding rules | Highly aligned; reliably follows system prompt constraints |
| **Rate Limits** | No limits; continuous processing | Free-tier introduces latency, timeouts, `time.sleep()` workarounds |

---

## Output Files

All outputs are saved to `OUTPUT_DIR` on Google Drive:

| File | Description |
|---|---|
| `*.html` | Extracted Item sections per filing (1,400 files) |
| `processed_chunks.csv` | Cleaned, chunked corpus with metadata |
| `embeddings.npy` | Precomputed embedding matrix (39,271 × 384) |
| `domain_qa.csv` | RAG answers for 10 evaluation questions |
| `df_ragas_results.csv` | RAGAS faithfulness and answer relevancy scores |

---

## Future Improvements

- **Metadata injection** — Prepend company name, ticker, CIK, and section ID into each chunk before embedding to enable reliable company-specific queries
- **Hybrid retrieval** — Combine dense (FAISS) and sparse (BM25) search for better coverage on both semantic and keyword-heavy queries
- **Query expansion** — Automatically rephrase or expand queries before retrieval to improve recall on weak or ambiguous questions
- **AI-relevance pre-filtering** — Add a keyword or classifier gate before embedding to ensure only AI-relevant passages enter the index
- **Ground truth evaluation** — Build a labeled test set to unlock RAGAS context precision and context recall metrics

---

## Project Context

Built as part of **AD698** coursework by Afreen Alam, Bhavya, Veena and Kriti Singh. Domain focus: AI strategy and investment signals in SEC 10-K filings (2024 fiscal year).
