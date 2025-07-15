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
from Retrieval.retriever_for_scimult import Retriever


def config():
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    parser.add_argument("--benchmark_directory", default="")
    parser.add_argument("--iteration_num", default=0, type=int, choices=[1, 2, 3])
    parser.add_argument("--remove_citations", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--remove_mentions", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--embedding_model", default="SciMult-MoE", choices=["SciMult-MoE"])
    parser.add_argument("--MoE_task", default='multi-hop-retrieval')
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=500, type=int)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--max_top_k", default=100, type=int)
    parser.add_argument("--corpus_directory", default="")
    parser.add_argument("--use_chunked", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--use_full_paper_as_corpus", default=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--chunk_unit", default=3000, type=int)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--use_full_paper_as_query", default=False, type=lambda x: x.lower() == "true")
    
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
    result_directory = "SciMult_Results"

    if args.use_chunked == True and args.use_full_paper_as_corpus == True:
        raise TypeError("Target Corpus cannot be both chunked and not chunked")
    
    if args.use_full_paper_as_query == False:
        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/{args.MoE_task}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}/iteration_{args.iteration_num}"
    elif args.use_full_paper_as_query == True: 
        result_folder_name = f"{last_two_dirs}/remove_citations_{args.remove_citations}_remove_mentions_{args.remove_mentions}/No_QueryOptimizers/{args.MoE_task}/{args.embedding_model}_top_{args.top_k}_use_chunked_{args.chunk_unit}_{args.use_chunked}_use_full_paper_as_corpus_{args.use_full_paper_as_corpus}_use_full_paper_as_query_{args.use_full_paper_as_query}/iteration_{args.iteration_num}"
    
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
