import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

dataset = load_dataset("simecek/czech_news")

# 0 - Texte pondéré
corpus = []
for i in range(len(dataset["train"])):
    article = dataset['train'][i]
    full = article["headline"] + " " + article["headline"] + " " + article["brief"] + " " + article["brief"] + " " + article["content"]
    corpus.append(full)

# 1 - Indexation
tokenized_docs = [doc.split() for doc in corpus] #je sépare en mots
bm25 = BM25Okapi(tokenized_docs)

# 2 - Fonction de recherche
def sparse_search(query, k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]
    results = dataset["train"][top_indices]
    return results["headline"], top_indices, scores[top_indices]

""""
# 3 - Test
headlines, indices, scores = sparse_search("volby Praha", k=5)    ###elections prague
for i, (h, s) in enumerate(zip(headlines, scores)):
    print(f"{i+1}. [{s:.2f}] {h}")
"""