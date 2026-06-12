import math

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def canonical_ranking_metrics(actual, predicted, cutoffs):
    if len(actual) != len(predicted):
        raise ValueError(
            "actual and predicted must have the same number of canonical examples."
        )
    if not actual:
        raise ValueError("Canonical metrics require at least one example.")

    result = {}
    for cutoff_value in cutoffs:
        cutoff = int(cutoff_value)
        if cutoff <= 0:
            raise ValueError("Metric cutoffs must be positive.")
        hit_sum = 0.0
        reciprocal_rank_sum = 0.0
        ndcg_sum = 0.0
        for label, ranking in zip(actual, predicted):
            try:
                rank = list(ranking[:cutoff]).index(int(label)) + 1
            except ValueError:
                rank = 0
            if rank > 0:
                hit_sum += 1.0
                reciprocal_rank_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log(rank + 1, 2)
        count = float(len(actual))
        result[f"hr@{cutoff}"] = hit_sum / count
        result[f"mrr@{cutoff}"] = reciprocal_rank_sum / count
        result[f"ndcg@{cutoff}"] = ndcg_sum / count
    return result
