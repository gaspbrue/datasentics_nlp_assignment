from datasets import load_dataset
from hybrid import hybrid_search

dataset = load_dataset("simecek/czech_news")

ground_truth = []
for i in range(len(dataset["train"])):
    article = dataset["train"][i]
    if article["keywords"] != []:
        ground_truth.append({
            "query": " ".join(article["keywords"]),
            "expected_index": i
        })

hits_at_1 = 0
hits_at_10 = 0
mrr = 0

for entry in ground_truth:
    result_indices = hybrid_search(entry["query"], k=10)
    expected = entry["expected_index"]
    if expected in result_indices:
        hits_at_10 += 1
        rank = result_indices.index(expected) + 1
        mrr += 1 / rank
        if rank == 1:
            hits_at_1 += 1

n = len(ground_truth)
print("=== HYBRID (keyword ground truth) ===")
print(f"Precision@1:  {hits_at_1/n:.2f}")
print(f"Precision@10: {hits_at_10/n:.2f}")
print(f"MRR@10:       {mrr/n:.2f}")