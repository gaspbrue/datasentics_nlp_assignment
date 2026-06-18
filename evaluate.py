from datasets import load_dataset
from sparse import sparse_search
from dense import dense_search

def evaluate(mode="sparse"):
    if mode == "sparse":
        search_fn = sparse_search
        def get_indices(query, k):
            _, top_indices, _ = search_fn(query, k)
            return list(top_indices)
    else:
        search_fn = dense_search
        def get_indices(query, k):
            return search_fn(query, k)

    dataset = load_dataset("simecek/czech_news")

    # Ground truth précision
    ground_truth = []
    for i in range(len(dataset["train"])):
        article = dataset["train"][i]
        if article["keywords"] != []:
            ground_truth.append({
                "query": " ".join(article["keywords"]),
                "expected_index": i
            })

    # Précision + MRR
    hits_at_1 = 0
    hits_at_10 = 0
    mrr = 0

    for entry in ground_truth:
        result_indices = get_indices(entry["query"], k=10)
        expected = entry["expected_index"]
        if expected in result_indices:
            hits_at_10 += 1
            rank = result_indices.index(expected) + 1
            mrr += 1 / rank
            if rank == 1:
                hits_at_1 += 1

    print(f"\n=== {mode.upper()} ===")
    print(f"Precision@1:  {hits_at_1/len(ground_truth):.2f}")
    print(f"Precision@10: {hits_at_10/len(ground_truth):.2f}")
    print(f"MRR@10:       {mrr/len(ground_truth):.2f}")

    # Recall
    recall_ground_truth = {}
    for i in range(len(dataset["train"])):
        for keyword in dataset["train"][i]["keywords"]:
            if keyword not in recall_ground_truth:
                recall_ground_truth[keyword] = []
            recall_ground_truth[keyword].append(i)

    total_recall = 0
    for keyword, expected_indices in recall_ground_truth.items():
        result_indices = get_indices(keyword, k=10)
        found = sum(1 for idx in expected_indices if idx in result_indices)
        total_recall += found / len(expected_indices)

    print(f"Recall@10:    {total_recall/len(recall_ground_truth):.2f}")


evaluate(mode="sparse")
evaluate(mode="dense")