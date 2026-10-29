"""
search_api.py — Hybrid search API for Phase 7 (Person 3).
Run with: uvicorn search_api:app --host 0.0.0.0 --port 8002
NOTE: this file assumes hybrid_search() and the underlying indices are already built in the

same Python process (see the notebook cells run before this). For a standalone deploy outside

Colab, the full setup (data loading, filtering, embeddings, BM25, reranker, hybrid_search)

would need to be moved into this file directly.

"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Phase 7 Medical Reference Search Service")

class SearchRequest(BaseModel):

    query: str
    source_filter: Optional[str] = None   # "ESI_Handbook" or "Tintinalli_Manual"
    top_k: int = 5
class SearchResult(BaseModel):

    score: float

    source: str

    chapter: Optional[str]

    section_title: Optional[str]

    page_number: int

    type: str

    matched_snippet: str

    context_text: str


@app.get("/health")

def health():
    return {"status": "ok"}
@app.post("/search", response_model=List[SearchResult])

def search_endpoint(request: SearchRequest):

    return hybrid_search(request.query, source_filter=request.source_filter, top_k=request.top_k)
