from fastapi import FastAPI
from pydantic import BaseModel
from hybrid import hybrid_search
from datasets import load_dataset

app = FastAPI()
dataset = load_dataset("simecek/czech_news")

class SearchRequest(BaseModel):
    query: str
    k: int = 10

@app.get("/")
def root():
    return {"status": "ok", "service": "NLP Retrieval Pipeline"}

@app.post("/search")
def search(request: SearchRequest):
    indices = [int(i) for i in hybrid_search(request.query, k=request.k)]
    results = []
    for rank, idx in enumerate(indices):
        article = dataset["train"][idx]
        results.append({
            "rank": rank + 1,
            "article_id": idx,
            "headline": article["headline"],
            "brief": article["brief"],
        })
    return {"query": request.query, "results": results}