# Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system for legal documents. Upload any legal PDF and ask questions in plain English — answers come directly from the document, not from the AI's general knowledge.

## Demo

Upload a legal PDF → Ask a question → Get a grounded answer with source citations.

## How It Works

```
Upload PDF → Extract text → Split into chunks → Embed with sentence-transformers
→ Store in ChromaDB → Ask question → Retrieve relevant chunks → Send to LLM → Answer
```

1. **Chunk** — Document split into 800-character overlapping pieces
2. **Embed** — Each chunk converted to a vector using `all-MiniLM-L6-v2` (runs locally)
3. **Store** — Vectors saved in ChromaDB for fast similarity search
4. **Query** — Question matched against chunks; top 5 sent to Llama via Groq
5. **Guard** — If no chunk is relevant enough, warns instead of hallucinating

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (in-memory) |
| LLM | Llama 3.1 8B via Groq API |
| UI | Streamlit |
| PDF parsing | pypdf |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires a free [Groq API key](https://console.groq.com) — enter it in the sidebar.

## Features

- Hallucination guard — warns when the document lacks relevant information
- Source citations — shows which chunk each answer came from, with similarity score
- Multi-PDF support — index multiple documents in one session
- No GPU required — embeddings run on CPU
