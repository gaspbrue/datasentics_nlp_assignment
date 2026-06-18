def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks



import numpy as np
from sentence_transformers import util

from datasets import load_dataset
dataset = load_dataset("simecek/czech_news")

all_chunks = []
for i in range(len(dataset["train"])):
    chunks=chunk_text(dataset["train"][i]["content"])
    for j in range(len(chunks)):
        all_chunks.append({"text" : chunks[j], "article_index" : i})



print(len(all_chunks))
print(all_chunks[0])


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)
print(embeddings.shape)




def dense_search(query, k=5):
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores.numpy())[::-1][:k]
    seen = set()
    result_indices = []
    for idx in top_indices:
        article_idx = all_chunks[idx]["article_index"]
        if article_idx not in seen:
            seen.add(article_idx)
            result_indices.append(article_idx)
        if len(result_indices) == k:
            break
    return result_indices


print(dense_search("volby Praha", k=5))



results = dense_search("volby Praha", k=5)
for idx in results:
    print(dataset["train"][idx]["headline"])