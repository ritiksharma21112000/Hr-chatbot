# backend/main.py
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dataclasses import asdict

from .rag_engine import RAGEngine, Employee

app = FastAPI(title="HR Resource Query Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine (loads data, embeddings/index)
rag = RAGEngine(data_path="backend/employees.json", index_path="backend/embeddings.faiss")

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    min_experience: Optional[int] = 0
    availability: Optional[str] = None  # "available" or "busy" or None
    required_skills: Optional[List[str]] = None  # list of skills to filter on

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Retrieval + augmentation
    matches = rag.search(
        payload.query,
        top_k=payload.top_k,
        min_experience=payload.min_experience,
        availability=payload.availability,
        required_skills=payload.required_skills,
    )

    # Generation
    try:
        answer = rag.generate_answer(payload.query, matches)
    except Exception as e:
        # fallback: return raw matches if generation fails
        answer = f"(Generation failed) {str(e)}"

    # Return structured response
    return {
        "generated_answer": answer,
        "matches": [asdict(m) for m in matches]
    }

@app.get("/employees/search")
async def simple_search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    min_experience: int = Query(0),
    availability: Optional[str] = Query(None),
):
    results = rag.search(q, top_k=top_k, min_experience=min_experience, availability=availability)
    return {"results": [r.dict() for r in results]}

@app.get("/health")
async def health():
    return {"status": "ok"}
