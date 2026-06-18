# NLP Retrieval Pipeline — DataSentics Assignment

Full-text search over 1,000 Czech news articles using sparse, dense, and hybrid retrieval, served via a containerized FastAPI.

## Approach

Three retrieval systems evaluated against three ground truth protocols:

| System | Method |
|---|---|
| Sparse | BM25 with field weighting (headline×2 + brief×2 + content) |
| Dense | Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2), chunking 2,000 chars / 200 overlap |
| Hybrid | Reciprocal Rank Fusion (RRF, k=60) over top-100 from each system |

## Results

| Metric | Sparse | Dense | Hybrid |
|---|---|---|---|
| Keyword GT · P@1 | 0.62 | 0.25 | 0.43 |
| Keyword GT · MRR@10 | 0.69 | 0.32 | 0.53 |
| Synthetic GT · P@1 | 0.74 | 0.64 | 0.74 |
| Synthetic GT · MRR@10 | 0.80 | 0.69 | 0.82 |
| Semantic GT · P@1 | 0.44 | 0.48 | 0.48 |
| Semantic GT · MRR@10 | 0.53 | 0.59 | 0.58 |

Hybrid never regresses and leads on MRR across synthetic and semantic protocols.

## Setup

```bash
conda create -n datasentics python=3.10
conda activate datasentics
pip install -r requirements.txt
```

## Run with Docker (recommended)

```bash
docker-compose up --build
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

## Run locally

```bash
uvicorn main:app --reload
```

## Query example

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Jaké jsou důsledky pandemie na českou ekonomiku?", "k": 5}'
```

## Evaluation

```bash
python evaluate.py                        # keyword ground truth (sparse + dense)
python evaluate_hybrid.py                 # hybrid on keyword GT
python synthetic_eval.py                  # LLM-generated queries
python evaluate_hybrid_synthetic.py       # hybrid on synthetic GT
python synthetic_eval_semantic.py         # lexically-independent queries
python evaluate_hybrid_semantic.py        # hybrid on semantic GT
```

## Dataset

`simecek/czech_news` on Hugging Face — 1,000 Czech news articles with headline, brief, content, and keywords fields.

## Project structure

```
sparse.py                       # BM25 retrieval
dense.py                        # dense retrieval
hybrid.py                       # RRF fusion
main.py                         # FastAPI server
ground_truth.py                 # keyword-based ground truth
evaluate.py                     # evaluation (sparse + dense)
evaluate_hybrid.py              # hybrid on keyword GT
synthetic_eval.py               # synthetic query generation + eval
synthetic_eval_semantic.py      # semantic query generation + eval
evaluate_hybrid_synthetic.py    # hybrid on synthetic GT
evaluate_hybrid_semantic.py     # hybrid on semantic GT
embeddings.npy                  # precomputed chunk embeddings
Dockerfile                      # container definition
docker-compose.yml              # orchestration
```
