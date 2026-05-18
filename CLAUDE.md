# Legal RAG Project — Portfolio Project

## Project Summary
A Legal RAG (Retrieval-Augmented Generation) system that lets you upload any legal PDF
and ask questions in plain English. The AI answers ONLY from the uploaded document —
no hallucination, no outside knowledge.

Built as a portfolio/interview project. Single file, easy to explain to freshers.

---

## Project Location
```
C:\Users\kisho\legal RAG project\
├── app.py               ← entire project in one file
├── requirements.txt
├── start_app.bat        ← double click to run
├── .env.example
└── CLAUDE.md
```

---

## How to Run

**Easy way — double click `start_app.bat`**

**Terminal way:**
```powershell
cd "C:\Users\kisho\legal RAG project"
streamlit run app.py
```

If you get errors about old database, delete it first:
```powershell
Remove-Item -Recurse -Force "C:\Users\kisho\legal RAG project\legal_chroma_db"
```

---

## Tech Stack (No API Key Needed — 100% Free & Offline)

| Part | Tool | Why |
|------|------|-----|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API |
| Vector DB | ChromaDB | Local, saves to disk |
| AI Generation | Ollama + llama3.2:1b | Local, no API |
| UI | Streamlit | Simple web interface |
| PDF Reading | pypdf | Extracts text from PDFs |

---

## How It Works (5 Steps)

```
1. Upload PDF  → extract text page by page
2. Chunk       → split into 800-character pieces (100 overlap)
3. Embed       → convert chunks to vectors (numbers) using local model
4. Store       → save vectors in ChromaDB on disk
5. Query       → question → find matching chunks → send to Llama → answer
```

---

## Key Features
- Hallucination Guard: if best chunk distance > 0.6, shows low confidence warning
- Source citations: shows which part of document the answer came from
- Batch size: 50 chunks per embedding batch
- No API key required — fully offline

---

## Current Status (as of May 18, 2026)
- ✅ PDF upload works
- ✅ Embeddings work (local sentence-transformers)
- ✅ ChromaDB storage works
- ✅ Groq API (llama-3.1-8b-instant) — tested and working
- ✅ Answer generation confirmed working — tested with Adani Hindenburg PDF
- ✅ Hallucination guard working — correctly says "not found" for unknown info
- ✅ LOW_CONFIDENCE_THRESHOLD tuned to 0.8 (was 0.6, caused false positives)

---

## Sample PDF for Testing
```
C:\Users\kisho\Downloads\Adani Hindenburg Supreme Court 2024 for legal RAG project.pdf
```

## Sample Questions to Ask
- "What are the main allegations against Adani?"
- "What did the Supreme Court decide?"
- "What is SEBI's role in this case?"
- "Who are the judges in this case?"

---

## Interview Explanation (30 seconds)
"I built a Legal RAG system where you upload any legal PDF and ask questions.
It splits the document into chunks, converts them to vectors using sentence-transformers,
stores them in ChromaDB, and when you ask a question it finds the most relevant chunks
and sends them to a local Llama model to generate a grounded answer.
It has a hallucination guard — if no chunk is relevant enough, it warns the user
instead of making up an answer. Everything runs locally — no API key needed."

---

## Problems We Solved
- Google Gemini embedding API gave 404 errors → switched to local sentence-transformers
- Google Gemini generation API gave 429 quota errors → switched to Ollama (local)
- st.rerun() was wiping error messages → fixed by saving to session state first
- Old ChromaDB incompatible after model change → delete legal_chroma_db folder and restart
