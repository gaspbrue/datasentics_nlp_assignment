from sparse import sparse_search
from dense import dense_search

K_RRF = 60

def hybrid_search(query, k=10):
    _, sparse_indices, _ = sparse_search(query, k=100)
    dense_indices = dense_search(query, k=100)
    
    sparse_indices = list(sparse_indices)
    dense_indices = list(dense_indices)
    
    scores = {}
    
    for rank, idx in enumerate(sparse_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (K_RRF + rank + 1)
    
    for rank, idx in enumerate(dense_indices):
        scores[idx] = scores.get(idx, 0) + 1 / (K_RRF + rank + 1)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [idx for idx, _ in ranked[:k]]