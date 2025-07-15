import torch
import faiss
import os
import sys
from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn
import json
import torch
from pathlib import Path
import logging
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Retrieval.metrics import evaluate_retrieval, average_evaluation_metrics
from MoE_Model.get_embedding import SciMult

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

logging.basicConfig(level=logging.INFO)

class Retriever:
    def __init__(self, args, result_folder_path):
        """
        Our overall retriever that retrieves input paper from massive target paper corpus.
        """
        self.benchmark_directory = args.benchmark_directory
        self.embedding_model = args.embedding_model
        self.use_gpu = torch.cuda.is_available()
        self.embedding_model_max_length = None

        self.use_chunked = args.use_chunked
        self.use_full_paper_as_corpus = args.use_full_paper_as_corpus
        self.chunk_unit = args.chunk_unit

        self.use_full_paper_as_query = args.use_full_paper_as_query

        self.chunked_corpus_to_paper_dict = {}
        self.full_paper_corpus_to_paper_dict = {}
        
        self.remove_citations = args.remove_citations
        self.remove_mentions = args.remove_mentions
        
        root_dir = os.path.dirname(args.corpus_directory)
        self.root_dir = os.path.join(root_dir, self.embedding_model)
        os.makedirs(self.root_dir, exist_ok=True)
        
        
        if self.use_chunked == False and self.use_full_paper_as_corpus == False:
            self.faiss_index_path = os.path.join(self.root_dir, f"DB_{args.MoE_task}.faiss")
        elif args.use_chunked == True and self.use_full_paper_as_corpus == False:
            self.faiss_index_path = os.path.join(self.root_dir, f"DB_chunked_{self.chunk_unit}_{args.MoE_task}.faiss")
        elif args.use_chunked == False and self.use_full_paper_as_corpus == True:
            self.faiss_index_path = os.path.join(self.root_dir, f"DB_full_paper_{args.MoE_task}.faiss")
        
        self.embedding_dimension = None
        self.batch_size = args.batch_size

        self.faiss_index = None
        self.chunked_index = None
        self.full_paper_index = None
        self.MoE_task = args.MoE_task

        num_devices = torch.cuda.device_count()
        
        device_list = []
        for device_number in range(num_devices):
            device_list.append(f"cuda:{device_number}")

        assert len(device_list) > 0, f"Expected at least 1 devices, but found {len(device_list)}"

        with open(args.corpus_directory, "r") as json_file:
            self.corpus = json.load(json_file)
        
        self.device = device_list[-1]
        if self.embedding_model == "SciMult-MoE":
            model_fn = f"MoE_Model/scimult_moe.ckpt"

            logging.info(f"Current expert is {self.MoE_task}")
            scimult = SciMult(model_fn=model_fn, device=self.device, task=self.MoE_task)
            
            self.embedding_dimension = 768
            self.question_encoder = scimult
            
            self.context_encoder = scimult
            
    
    def initialize(self):
        self.format_corpus()
        os.makedirs(self.root_dir, exist_ok=True)
        if self.use_chunked == False and self.use_full_paper_as_corpus == True:
            self.faiss_index, self.full_paper_index = self.build_faiss_index()
        elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
            self.faiss_index, self.chunked_index = self.build_faiss_index()
        elif self.use_chunked == False and self.use_full_paper_as_corpus == False:
            self.faiss_index = self.build_faiss_index()
    
    
    def format_corpus(self):        
        if self.use_chunked == True and self.use_full_paper_as_corpus == False:
            total_chunked_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                chunked_sections = paper[f"chunked_sections_{self.chunk_unit}"]
                for chunked_section in chunked_sections:
                    try:
                        self.chunked_corpus_to_paper_dict[chunked_section]
                    except KeyError:
                        total_chunked_corpus.append(chunked_section)
                        self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"

            self.corpus = total_chunked_corpus
        
        elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
            print("Formatting corpus for full paper for single source")
            total_full_paper_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper['abstract']
                arxiv_full_paper_directory = paper["full_paper_directory"]
                with open(arxiv_full_paper_directory, "r") as file:
                    full_paper_content = file.read()
                    
                try:
                    self.full_paper_corpus_to_paper_dict[full_paper_content]
                except KeyError:
                    total_full_paper_corpus.append(full_paper_content)
                    self.full_paper_corpus_to_paper_dict[full_paper_content] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
            
            self.corpus = total_full_paper_corpus

        elif self.use_chunked == False and self.use_full_paper_as_corpus == False:
            logging.info("Formatting corpus for abstracts without chunked or full content")
            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                    
                formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                try:
                    formatted_total_corpus_dict[formatted_total_candidate]
                except KeyError:   
                    formatted_total_corpus.append(formatted_total_candidate)
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                    
            self.corpus = formatted_total_corpus
    
        
    def format_query_candidates(self, paper):
        if self.use_full_paper_as_query == False:
            query_paper = paper["Query"]
            query_title = query_paper["title"]
            query_abstract = query_paper["abstract"]
            
            formatted_query = f"Title: {query_title}\nAbstract: {query_abstract}"

        elif self.use_full_paper_as_query == True:
            paper_id = paper["id"]
            paper_filename = f"{paper_id}.mmd"
                
            parent_dir = os.path.dirname(self.benchmark_directory)
            benchmark_root_dir = os.path.dirname(parent_dir) 
            if self.remove_citations == False and self.remove_mentions == False:
                    root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers")
                
            if self.remove_citations == True and self.remove_mentions == True:
                    root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers_remove_citations_True_remove_mentions_True")

            full_paper_path = os.path.join(root_folder_name, paper_filename)

            logging.info(f"{full_paper_path}")

            with open(full_paper_path, "r") as file:
                paper_content = file.read()

            formatted_query = paper_content
        
        formatted_candidates = []
        candidate_papers = paper["Candidate"]
        for candidate in candidate_papers:
            candidate_title = candidate["title"]
            candidate_abstract = candidate["abstract"]
            
            formatted_candidate = f"Title: {candidate_title}\nAbstract: {candidate_abstract}"

            formatted_candidates.append(formatted_candidate)
        
        return formatted_query, formatted_candidates  
        
    # Function to encode passages in batches
    def encode_passages(self, passages):
        if self.embedding_model == "SciMult-MoE":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.get_embedding_for_SciMult(paper_texts=passages, batch_size=self.batch_size)

            logging.info(f"Vector shape for passages is {embeddings.shape}")
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)


    # Function to encode a query
    def encode_query(self, query):
        if self.embedding_model == "SciMult-MoE":
            with torch.no_grad():
                embedding = self.question_encoder.get_embedding_for_SciMult(paper_texts=[query], batch_size=1)
            
            logging.info(f"Vector shape for query is {embedding.shape}")
            return embedding
        
      
    
    # Step 1: Preprocess and index the corpus
    def build_faiss_index(self):
        if os.path.exists(self.faiss_index_path):
            logging.info(f"Loading existing FAISS index...{self.faiss_index_path}")
            index = faiss.read_index(self.faiss_index_path)
            
            return index

        logging.info("Building FAISS index...")
            
        # Initialize a FAISS index for Maximum Inner Product
        index = faiss.IndexFlatIP(self.embedding_dimension)


        # Process passages in chunks to avoid memory issues
        for i in range(0, len(self.corpus), self.batch_size):
            batch = self.corpus[i:i + self.batch_size]
            logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.corpus) // self.batch_size + 1}")
            embeddings = self.encode_passages(batch)
            index.add(embeddings)  

        # Save the index to disk
        faiss.write_index(index, self.faiss_index_path)

        logging.info("FAISS index built and saved.")

        return index
    
    # Calculate similarities and retrieve top-k abstracts
    def retrieve_top_k(self, query, top_k):        
        query_embedding = self.encode_query(query)  # Encode the query
        if self.use_chunked == False and self.use_full_paper_as_corpus == False:
            logging.info(f"Retrieving without using full papers")
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            top_k_corpus = [(self.corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])]    
        elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]
        elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            top_k_corpus = [(self.full_paper_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]
        
        return top_k_corpus
    
    def evaluate(self, args, test_file_directory, result_folder_path):
        """
        Evaluate Retrieval Performance on scientific papers submitted to recent venues, such as ICLR 2024, ICLR 2025.
        You can either use query optimizer or not. 
        """
                     
        total_results = []
        for files in test_file_directory:
            with open(files, "r") as json_file:
                evaluation_data = json.load(json_file)


            formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                    
            parsed_retrieved_top_corpus = []
            top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k)
                    
            if self.use_chunked == True or self.use_full_paper_as_corpus == True:
                organized_retrieved_top_corpus = {}    
                for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                    organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                    parsed_retrieved_top_corpus.append(corpus)
                    
            elif self.use_chunked == False and self.use_full_paper_as_corpus == False:
                organized_retrieved_top_corpus = {}
                for idx, (corpus, score) in enumerate(top_k_corpus):
                    organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                    parsed_retrieved_top_corpus.append(corpus)

            current_result = evaluate_retrieval(parsed_retrieved_top_corpus, formatted_correct_candidates, args.top_k, args.max_top_k)
            total_results.append(current_result)

            organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": f"{formatted_query}"}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
                
                    
            current_result_folder_path = f"{result_folder_path}/{evaluation_data['id']}.json"
            with open(current_result_folder_path, "w") as json_file:
                json.dump(organized_results, json_file, indent=4)
                