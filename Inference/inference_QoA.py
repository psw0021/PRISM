import argparse
import os
import sys
import json
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Retrieval.metrics import evaluate_retrieval, average_evaluation_metrics
from Retrieval.retriever import Retriever
from QueryOptimizer.agents import QueryOptimizer


def config():
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    parser.add_argument("--benchmark_directory", default="")
    parser.add_argument("--iteration_num", default=0, type=int, choices=[1, 2, 3])
    parser.add_argument("--remove_citations", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--remove_mentions", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_query_optimizer", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--multi_agent", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--query_optimizer_model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--use_gpt", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--embedding_model", default="", choices=["SPECTER", "text-embedding-3-small", \
        "jina-embeddings-v2-base-en", "SPECTER2", "SPECTER2_Base", "SciNCL", "text-embedding-ada-002"])
    parser.add_argument("--embedding_fusion", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--embedding_fuse_method", default="naive_aggregation", choices=["naive_aggregation", "maximum_similarity"])
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=500, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--query_detailedness", default=3, type=int)
    parser.add_argument("--max_top_k", default=100, type=int)
    parser.add_argument("--include_original_retrieval", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_base_agent", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_method_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_experiment_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_research_question_agent", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--corpus_directory", default="")
    parser.add_argument("--use_multi_source", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_chunked", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_full_paper_as_corpus", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--chunk_unit", default=3000, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--use_full_paper_as_query", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_trained_model", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--method_agent_model_path", default="", type=str)
    parser.add_argument("--experiment_agent_model_path", default="", type=str)
    parser.add_argument("--research_question_agent_model_path", default="", type=str)
    parser.add_argument("--temperature", default=0, type=str)
    parser.add_argument("--max_tokens", default=2000, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.7, type=float)
    parser.add_argument("--repetition_penalty", default=1.2, type=float)
    
    args = parser.parse_args()
    
    return args

def save_config(config, result_path):
    """Save the configuration dictionary as JSON."""
    config_to_save = vars(config)
    config_path = os.path.join(result_path, "config.json")

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4)

    print(f"Config saved at {config_path}")

    
def evaluate(args):
    ## Bencharmk directory
    path = Path(args.benchmark_directory)
    last_two_dirs = "/".join(path.parts[-3:])
    result_directory = "Results"

    if args.use_chunked == True and args.use_full_paper_as_corpus == True:
        raise TypeError("Target Corpus cannot be both chunked and not chunked")
    
    if args.embedding_model ==  "SPECTER2" or args.embedding_model == "SPECTER2_Base" or args.embedding_model == "SciNCL":
        if args.use_query_optimizer == True:
            raise TypeError("When Embedding models from ASPIRE and SPECTER2 are used, you should not experiment with query optimizer")

    if args.embedding_fusion == True:
        if args.use_query_optimizer == False or args.multi_agent == False:
            raise TypeError("When experimenting with embedding fusion, you must always generate multiple queries with query optimizers")

    if args.use_query_optimizer == True:
        if args.embedding_fusion == True:
            result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}_{args.query_detailedness}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_fuse_embedding_{args.embedding_fusion}_{args.embedding_fuse_method}/iteration_{args.iteration_num}"
        elif args.embedding_fusion == False:
            if args.use_trained_model == False:
                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}_{args.query_detailedness}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
            elif args.use_trained_model == True:
                try:
                    path = Path(args.method_agent_model_path)
                    train_dataset_name = path.parents[3].name
                except:
                    pass

                try:
                    path = Path(args.experiment_agent_model_path)
                    train_dataset_name = path.parents[3].name
                except:
                    pass

                try:
                    path = Path(args.research_question_agent_model_path)
                    train_dataset_name = path.parents[3].name
                except:
                    pass

                result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/Use_QueryOptimizers/use_trained_model_{args.use_trained_model}/{train_dataset_name}/{args.query_optimizer_model}/use_multi_source_{args.use_multi_source}/multi_agent_{args.multi_agent}_INCLUDE_ORIGINAL_RETRIEVAL_{args.include_original_retrieval}_BASE_{args.use_base_agent}_METHOD_{args.use_method_agent}_RESEARCH_QUESTION_{args.use_research_question_agent}_EXPERIMENT_{args.use_experiment_agent}_{args.query_detailedness}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
            
    else:
        if args.use_full_paper_as_query == False:
            result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
        elif args.use_full_paper_as_query == True: 
            result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/use_query_optimizer_{args.use_query_optimizer}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"
    
    result_folder_path = os.path.join(result_directory, result_folder_name)
    os.makedirs(result_folder_path, exist_ok=True)
    
    save_config(args, result_folder_path)

    test_file_directory = []
    # Walk through the directory tree
    for root, _, files in os.walk(args.benchmark_directory):
        for file in files:
            if file.endswith('.json'):
                test_file_directory.append(os.path.join(root, file))
        
    test_file_directory = test_file_directory[args.start_idx:args.end_idx]
    
    ## Initialize Retrieval Module
    retriever = Retriever(args, result_folder_path)
    retriever.initialize()
    retriever.evaluate(args, test_file_directory, result_folder_path)

# Example usage
if __name__ == "__main__":
    # Abstract to compare
    args = config()
    evaluate(args)
