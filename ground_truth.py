from datasets import load_dataset

dataset = load_dataset("simecek/czech_news")

ground_truth = []
for i in range(len(dataset["train"])):
    article = dataset["train"][i]
    if article["keywords"] != []:
        ground_truth.append({
            "query": " ".join(article["keywords"]),
            "expected_index": i
        })

print(len(ground_truth))
print(ground_truth[800])