# NLP Retrieval Pipeline — DataSentics Assignment

A retrieval pipeline for searching in a corpus of Czech news articles.

## Approach
- Sparse retrieval: BM25
- Dense retrieval: Sentence Transformers + vector search
- Evaluation: precision, recall, MRR on keyword-based ground truth

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Structure
```
├── data/           # dataset
├── src/            # pipeline code
├── notebooks/      # exploration
├── evaluation/     # ground truth + metrics
└── README.md
```
