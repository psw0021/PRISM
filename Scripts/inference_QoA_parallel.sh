#!/bin/bash
echo $CUDA_VISIBLE_DEVICES

# conda environment
CONDA_ENV_NAME="paper_retrieval"
### Environmental variables

use_query_optimizer=True
use_full_paper_as_query=False
multi_agent=True
query_optimizer_model="gpt-4o-mini-2024-07-18"
gpu_memory_utilization=0.6
use_gpt=True
benchmark_directory="Final_Dataset_w_citations_mentions_removed/Benchmark/ACL/Cited_Papers"
embedding_model="text-embedding-3-small"
top_k=300
max_top_k=300
corpus_directory="Final_Dataset_w_citations_mentions_removed/Target_Corpus/target_corpus_citations_removed_True_mentions_removed_True/corpus.json"
batch_size=4
include_original_retrieval=False
use_base_agent=False
use_method_agent=True
use_experiment_agent=True
use_research_question_agent=True
query_detailedness=3
use_multi_source=False
use_chunked=True
chunk_unit=3000
use_full_paper_as_corpus=False
embedding_fusion=True
embedding_fuse_method="naive_aggregation"
use_trained_model=False
remove_citations=True 
remove_mentions=True



fi

if [[ ${use_query_optimizer} == False && ${multi_agent} == False ]]; then
    use_multi_source=False
fi

function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    local iteration=$4
    echo "Running inference_QoA.py"
    rm logs/output_${job_num}_GPT.log 
    python Inference/inference_QoA.py \
    --iteration_num ${iteration}    \
    --batch_size ${batch_size} \
    --use_query_optimizer ${use_query_optimizer} \
    --use_full_paper_as_query ${use_full_paper_as_query} \
    --multi_agent ${multi_agent}    \
    --query_optimizer_model ${query_optimizer_model} \
    --gpu_memory_utilization ${gpu_memory_utilization}  \
    --use_gpt ${use_gpt} \
    --corpus_directory ${corpus_directory} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --include_original_retrieval ${include_original_retrieval} \
    --use_base_agent ${use_base_agent}  \
    --use_method_agent ${use_method_agent} \
    --use_experiment_agent ${use_experiment_agent} \
    --use_research_question_agent ${use_research_question_agent} \
    --benchmark_directory ${benchmark_directory} \
    --remove_citations ${remove_citations}  \
    --remove_mentions ${remove_mentions}    \
    --embedding_model ${embedding_model} \
    --query_detailedness ${query_detailedness} \
    --top_k ${top_k} \
    --max_top_k ${max_top_k} \
    --use_multi_source ${use_multi_source} \
    --use_chunked ${use_chunked} \
    --chunk_unit ${chunk_unit}  \
    --use_full_paper_as_corpus ${use_full_paper_as_corpus}  \
    --embedding_fusion ${embedding_fusion}  \
    --embedding_fuse_method ${embedding_fuse_method}    \
    --use_trained_model ${use_trained_model} \
    --method_agent_model_path ${method_agent_model_path} \
    --experiment_agent_model_path ${experiment_agent_model_path} \
    --research_question_agent_model_path ${research_question_agent_model_path} > logs/GPT/output_${job_num}_GPT.log 2>&1
}




for i in {1..1}; do
    echo "Iteration $i"
    iteration=$i
    current_start=0  # Reset for each iteration
        
    total_indices=400
    indices_per_thread=400

    job_count=0
    while [ "$current_start" -lt "$total_indices" ]; do
        export CUDA_VISIBLE_DEVICES=1

        current_end=$((current_start + indices_per_thread))
        if [ "$current_end" -gt "$total_indices" ]; then
            current_end=$total_indices
        fi

        # Run the job in background
        run_job $current_start $current_end $job_count $iteration &

        ((job_count++))
        current_start=$current_end
    done

    # Wait for all background jobs in this iteration to finish
    wait
done
