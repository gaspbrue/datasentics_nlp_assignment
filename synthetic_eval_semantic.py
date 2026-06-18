import os
import random
import anthropic
import numpy as np
from datasets import load_dataset
from sparse import sparse_search
from dense import dense_search, model, all_chunks, chunk_text

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
dataset = load_dataset("simecek/czech_news")

def generate_query(article_content):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                "Based on this news article, write one natural language search query in Czech "
                "that someone might use to find this article.\n\n"
                "CRITICAL RULES:\n"
                "- Do NOT use any words that appear in the article\n"
                "- Express the concept using synonyms, paraphrases, or related concepts only\n"
                "- Write a genuine question a user would ask, not a keyword list\n"
                "- Return only the query, nothing else\n\n"
                f"Article: {article_content[:500]}"
            )
        }]
    )
    return response.content[0].text.strip()

# 50 articles aléatoires
random.seed(42)
sample_indices = random.sample(range(len(dataset["train"])), 50)

# Génération des requêtes synthétiques
print("Generating synthetic queries...")
synthetic_ground_truth = []
for idx in sample_indices:
    article = dataset["train"][idx]
    query = generate_query(article["content"])
    synthetic_ground_truth.append({
        "query": query,
        "expected_index": idx
    })
    print(f"  [{idx}] {query}")

# Évaluation sparse
print("\n=== SPARSE (synthetic queries) ===")
hits_at_1 = 0
hits_at_10 = 0
mrr = 0

for entry in synthetic_ground_truth:
    _, top_indices, _ = sparse_search(entry["query"], k=10)
    top_indices = list(top_indices)
    expected = entry["expected_index"]
    if expected in top_indices:
        hits_at_10 += 1
        rank = top_indices.index(expected) + 1
        mrr += 1 / rank
        if rank == 1:
            hits_at_1 += 1

n = len(synthetic_ground_truth)
print(f"Precision@1:  {hits_at_1/n:.2f}")
print(f"Precision@10: {hits_at_10/n:.2f}")
print(f"MRR@10:       {mrr/n:.2f}")

# Évaluation dense
print("\n=== DENSE (synthetic queries) ===")
hits_at_1 = 0
hits_at_10 = 0
mrr = 0

for entry in synthetic_ground_truth:
    result_indices = dense_search(entry["query"], k=10)
    expected = entry["expected_index"]
    if expected in result_indices:
        hits_at_10 += 1
        rank = result_indices.index(expected) + 1
        mrr += 1 / rank
        if rank == 1:
            hits_at_1 += 1

print(f"Precision@1:  {hits_at_1/n:.2f}")
print(f"Precision@10: {hits_at_10/n:.2f}")
print(f"MRR@10:       {mrr/n:.2f}")