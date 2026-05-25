---
title: RAG Document Assistant
emoji: 📄
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# RAG Document Q&A Assistant

An AI-powered application...

# RAG Document Q&A Assistant

An AI-powered application that answers questions from any uploaded PDF document using RAG (Retrieval-Augmented Generation) architecture.

## Tech Stack
- LangChain — RAG pipeline and chain orchestration
- HuggingFace all-MiniLM-L6-v2 — free local embeddings
- FAISS — vector store for semantic search
- Groq + Llama 3.3 — free LLM for answer generation
- Gradio — web interface

## How it works
1. Upload any PDF document
2. Document is split into 500-character chunks with 50-character overlap
3. Each chunk is converted to a vector using HuggingFace embeddings
4. Your question is also converted to a vector
5. FAISS finds the top 3 most semantically similar chunks
6. Those chunks are passed as context to Llama 3.3 via Groq
7. LLM generates an accurate, grounded answer

## Project Structure
```
rag-document-assistant/
├── app.py          # Gradio UI and user interaction
├── rag_engine.py   # RAG pipeline logic
├── .env            # API keys (not uploaded)
└── .gitignore      # Ignored files
```