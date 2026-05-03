# Section-Scoped RAG System for SEC 10-K Filings (2024)

**AD698 | Course Project**  
Afreen Alam, Bhavya, Veena, Kriti Singh

---

## Overview

This project builds a Retrieval-Augmented Generation (RAG) pipeline over a domain-scoped subset of SEC 10-K filings. Rather than indexing full filings, we extract and focus on three sections most likely to contain AI-related disclosures — Item 1 (Business), Item 1A (Risk Factors), and Item 7 (Management's Discussion & Analysis) — and use them to answer structured financial analysis questions.

---

## Repository Contents

```
├── AD698_rag_notebook.ipynb     # Main project notebook (full pipeline)
├── AD698_Final_Project_Report.pdf  # Written report with evaluation and analysis
└── README.md
```

---

## Pipeline

The notebook walks through the following steps end-to-end:

1. **Mount Drive & Extract SEC Sections** — reads raw 10-K HTML files and extracts Items 1, 1A, and 7 using regex-based boundary detection
2. **Clean & Chunk** — strips HTML with BeautifulSoup, splits text into ~400-word chunks with 50-word overlap, and saves metadata (chunk_id, company_name, section_id)
3. **Generate Embeddings** — encodes all 39,271 chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized)
4. **Build FAISS Index** — stores embeddings in a `IndexFlatIP` index for fast cosine similarity search
5. **Retrieve** — encodes a user query with the same model and retrieves top_k=5 most similar chunks
6. **Re-rank** — re-scores retrieved chunks using `cross-encoder/ms-marco-TinyBERT-L-2` for more precise relevance ordering
7. **Generate** — passes re-ranked chunks to `gpt-4o-mini` with a grounding-enforced system prompt via the GitHub Models API
8. **Evaluate** — uses RAGAS (faithfulness + answer relevancy) to quantitatively assess pipeline quality


