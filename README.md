# HR Resource Query Chatbot

**Submission deadline:** August 10, 2025

## Overview
This project builds a small RAG-based HR assistant that finds employees from a sample dataset using semantic search (SentenceTransformers + FAISS) and generates human-readable answers using OpenAI.

## Features
- Semantic retrieval with SentenceTransformers + FAISS
- Filtering: min experience, availability, required skills
- RAG: retrieved context + OpenAI generation (chat completion)
- Streamlit frontend with filters
- Persistent embeddings for fast startup

## Setup
1. Clone repo
2. Create virtual env & install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt

3. start backend 
    uvicorn backend.main:app --reload --port 8000

4. start frontend
    streamlit run frontend/app.py