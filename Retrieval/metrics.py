import numpy as np

def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / len(relevant) if relevant else 0


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def dcg_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    dcg = 0
    for i, doc in enumerate(retrieved_at_k):
        if doc in relevant:
            rel = 1  # binary relevance
            dcg += rel / np.log2(i + 2)  # log2(i+2) since i starts from 0
    return dcg

def ndcg_at_k(retrieved, relevant, k):
    ideal_relevant = sorted(relevant, reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_relevant, relevant, k)
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

def Mean_Average_Precision(retrieved, relevant, k, max_k):
    precisions = []
    for i in range(1, max_k + 1):
        precision_at_i = precision_at_k(retrieved, relevant, i)
        precisions.append(precision_at_i)
    
    total_precision = 0
    for j in range(0, len(precisions)):
        total_precision = total_precision + precisions[j]
        
    mean_average_precision = total_precision/len(precisions)
    
    return mean_average_precision


def evaluate_retrieval(retrieved_docs, relevant_docs, k, max_k):
    #precision = precision_at_k(retrieved_docs, relevant_docs, k)
    recall300 = recall_at_k(retrieved_docs, relevant_docs, 300)
    ndcg300 = ndcg_at_k(retrieved_docs, relevant_docs, 300)
    
    recall200 = recall_at_k(retrieved_docs, relevant_docs, 200)
    ndcg200 = ndcg_at_k(retrieved_docs, relevant_docs, 200)
    
    recall100 = recall_at_k(retrieved_docs, relevant_docs, 100)
    ndcg100 = ndcg_at_k(retrieved_docs, relevant_docs, 100)
    
    
    #f1 = f1_score(precision, recall)
    #mean_average_precision = Mean_Average_Precision(retrieved_docs, relevant_docs, k, max_k)
    return {
        "Recall@300": recall300,
        "nDCG@300": ndcg300,
        "Recall@200": recall200,
        "nDCG@200": ndcg200,
        "Recall@100": recall100,
        "nDCG@100": ndcg100,
    }
    
def average_evaluation_metrics(total_results: list[dict]):
    total_Recall300 = 0
    total_nDCG300 = 0
    total_Recall200 = 0
    total_nDCG200 = 0
    total_Recall100 = 0
    total_nDCG100 = 0
    for result in total_results:
        Recall300 = result["Recall@300"]
        nDCG300 = result["nDCG@300"]
        
        total_Recall300 = total_Recall300 + Recall300
        total_nDCG300 = total_nDCG300 + nDCG300
        
        Recall200 = result["Recall@200"]
        nDCG200 = result["nDCG@200"]
        
        total_Recall200 = total_Recall200 + Recall200
        total_nDCG200 = total_nDCG200 + nDCG200
        
        Recall100 = result["Recall@100"]
        nDCG100 = result["nDCG@100"]
        
        total_Recall100 = total_Recall100 + Recall100
        total_nDCG100 = total_nDCG100 + nDCG100
        
    
    average_Recall300 = total_Recall300 / len(total_results)
    average_nDCG300 = total_nDCG300 / len(total_results)
    
    average_Recall200 = total_Recall200 / len(total_results)
    average_nDCG200 = total_nDCG200 / len(total_results)
    
    average_Recall100 = total_Recall100 / len(total_results)
    average_nDCG100 = total_nDCG100 / len(total_results)
    
    return {
        "Average_Recall@300": average_Recall300,
        "Average_nDCG@300": average_nDCG300,
        "Average_Recall@200": average_Recall200,
        "Average_nDCG@200": average_nDCG200,
        "Average_Recall@100": average_Recall100,
        "Average_nDCG@100": average_nDCG100,
    }
    
