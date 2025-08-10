# backend/rag_engine.py
import os
import json
from typing import List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

load_dotenv()  # reads .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    # It's allowed to run retrieval-only without an OpenAI key; generation will fail without it
    print("Warning: OPENAI_API_KEY not found. Generation will fail until you set it in .env.")

openai.api_key = OPENAI_API_KEY

@dataclass
class Employee:
    id: int
    name: str
    skills: List[str]
    experience_years: int
    projects: List[str]
    availability: str

    @staticmethod
    def from_dict(d):
        return Employee(
            id=d.get("id"),
            name=d.get("name"),
            skills=d.get("skills", []),
            experience_years=int(d.get("experience_years", 0)),
            projects=d.get("projects", []),
            availability=d.get("availability", "available"),
        )

    def to_text(self) -> str:
        skills = ", ".join(self.skills)
        projects = "; ".join(self.projects)
        return f"{self.name} ({self.experience_years} yrs) — skills: {skills}. Projects: {projects}. Availability: {self.availability}."

class RAGEngine:
    def __init__(self, data_path="backend/employees.json", index_path="backend/embeddings.faiss", model_name="all-MiniLM-L6-v2"):
        self.data_path = data_path
        self.index_path = index_path
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.employees: List[Employee] = []
        self.documents: List[str] = []
        self.embeddings = None
        self.index = None
        self._load_data()
        self._ensure_index()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            j = json.load(f)
        self.employees = [Employee.from_dict(e) for e in j.get("employees", [])]
        # Build flat documents for each employee (used as retrieval payload)
        self.documents = [emp.to_text() for emp in self.employees]

    def _ensure_index(self):
        # If an index file exists, load it + embeddings cache if present
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".meta.npz"):
            meta = np.load(self.index_path + ".meta.npz", allow_pickle=True)
            self.embeddings = meta["embeddings"]
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings.astype("float32"))
            print("Loaded FAISS index and embeddings from disk.")
        else:
            # create embeddings and index
            self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings.astype("float32"))
            # save embeddings for subsequent runs
            np.savez_compressed(self.index_path + ".meta.npz", embeddings=self.embeddings)
            print("Created new embeddings and FAISS index (saved to disk).")

    def search(self, query: str, top_k: int = 3, min_experience: int = 0, availability: Optional[str] = None, required_skills: Optional[List[str]] = None) -> List[Employee]:
        # Query embedding
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_emb, top_k * 5)  # get more and filter afterwards
        candidates = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.employees):
                continue
            emp = self.employees[idx]
            # apply simple filters
            if emp.experience_years < min_experience:
                continue
            if availability and emp.availability.lower() != availability.lower():
                continue
            if required_skills:
                # require that all required_skills are present (case-insensitive)
                emp_skills_lower = [s.lower() for s in emp.skills]
                if not all(req.lower() in emp_skills_lower for req in required_skills):
                    continue
            candidates.append(emp)
            if len(candidates) >= top_k:
                break
        return candidates

    def generate_answer(self, query: str, matches: List[Employee]) -> str:
        """
        Build a prompt using the matches and call OpenAI to create a well-formed response.
        If OPENAI_API_KEY is missing this will raise an exception.
        """
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set. Cannot call generation API.")

        context = "\n\n".join([m.to_text() for m in matches]) if matches else "No matching employees found."

        prompt = f"""
You are a helpful HR assistant. Use the following employee context to answer the user's query.

Context:
{context}

User Query:
{query}

Produce:
- A short summary of best matches (2-4 sentences).
- A bullet list of the matching employees with one-line reasons why they fit.
- If none match, suggest next steps (skills to relax, hire, or filter changes).

Answer:
"""
        # Chat completion call (OpenAI python package)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change as needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=350,
        )
        # Newer API returns choices with message content
        text = response["choices"][0]["message"]["content"]
        return text
