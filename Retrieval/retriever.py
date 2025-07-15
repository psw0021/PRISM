import torch
import faiss
import os
import sys
from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, AutoModelForCausalLM, AutoTokenizer,DPRContextEncoderTokenizer, \
DPRQuestionEncoder, DPRQuestionEncoderTokenizer, AutoModel, AutoConfig
from adapters import AutoAdapterModel
import json
import torch
from torch import nn as nn
from pathlib import Path
import logging
from openai import OpenAI
import tiktoken
import re
from vllm import LLM, SamplingParams

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from QueryOptimizer.agents import QueryOptimizer
from Retrieval.metrics import evaluate_retrieval, average_evaluation_metrics

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

        self.embedding_fusion = args.embedding_fusion
        self.embedding_fuse_method = args.embedding_fuse_method
        
        self.use_multi_source = args.use_multi_source

        self.use_chunked = args.use_chunked
        self.use_full_paper_as_corpus = args.use_full_paper_as_corpus
        self.chunk_unit = args.chunk_unit
    
        self.use_gpt = args.use_gpt
        self.use_trained_model = args.use_trained_model

        self.include_original_retrieval = args.include_original_retrieval

        self.use_full_paper_as_query = args.use_full_paper_as_query

        self.chunked_corpus_to_paper_dict = {}
        self.full_paper_corpus_to_paper_dict = {}
        
        self.remove_citations = args.remove_citations
        self.remove_mentions = args.remove_mentions
        
        root_dir = os.path.dirname(args.corpus_directory)
        self.root_dir = os.path.join(root_dir, self.embedding_model)
        os.makedirs(self.root_dir, exist_ok=True)
        
        if self.use_multi_source == False:
            if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == True and self.use_full_paper_as_corpus == False:
                self.faiss_index_path = os.path.join(self.root_dir, f"DB_chunked_{self.chunk_unit}.faiss")
            elif args.use_chunked == False and self.use_full_paper_as_corpus == True:
                self.faiss_index_path = os.path.join(self.root_dir, f"DB_full_paper.faiss")
            
        elif self.use_multi_source == True:
            if args.use_chunked == False and self.use_full_paper_as_corpus == False:
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == True and self.use_full_paper_as_corpus == False:
                self.chunked_faiss_index_path = os.path.join(self.root_dir, f"DB_chunked_{self.chunk_unit}.faiss")
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
            elif args.use_chunked == False and self.use_full_paper_as_corpus == True:
                self.full_paper_faiss_index_path = os.path.join(self.root_dir, f"DB_full_paper.faiss")
                self.faiss_index_path = os.path.join(self.root_dir, "DB.faiss")
        
        self.candidate_embeddings = None
        
        self.embedding_dimension = None
        self.use_query_optimizer = args.use_query_optimizer
        self.batch_size = args.batch_size

        self.faiss_index = None
        self.chunked_index = None
        self.full_paper_index = None
        
        self.vllm_dict = {}
        num_devices = torch.cuda.device_count()
        
        device_list = []
        for device_number in range(num_devices):
            device_list.append(f"cuda:{device_number}")

        assert len(device_list) > 0, f"Expected at least 1 devices, but found {len(device_list)}"
        if self.use_gpt == False: 
            if self.use_query_optimizer == True and self.use_trained_model == True:
                print(device_list)
                
                self.vllm_dict["METHOD"] = {"agent": args.method_agent_model_path, "device": None}
                self.vllm_dict["EXPERIMENT"] = {"agent": args.experiment_agent_model_path, "device": None}
                self.vllm_dict["RESEARCH_QUESTION"] = {"agent": args.research_question_agent_model_path, "device": None}
                
            elif self.use_query_optimizer == True and self.use_trained_model == False:
                assert len(device_list) >= 1, f"Expected at least 1 device, but found {len(device_list)}"
                base_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
                
                self.vllm_dict["METHOD"] = {"agent": base_model, "device": device_list[0]}
                self.vllm_dict["EXPERIMENT"] = {"agent": base_model, "device": device_list[0]}
                self.vllm_dict["RESEARCH_QUESTION"] = {"agent": base_model, "device": device_list[0]}
        
        self.cuda_embedding_model_device = device_list[-1]

        if self.use_query_optimizer == True:
            self.QueryOptimizer = QueryOptimizer(args, result_folder_path, self.vllm_dict)

        with open(args.corpus_directory, "r") as json_file:
            self.corpus = json.load(json_file)
        
        if self.embedding_model == "jina-embeddings-v2-base-en":
            self.question_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.cuda_embedding_model_device)
            self.context_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=8192
            self.embedding_dimension = 768
            
        elif self.embedding_model == "text-embedding-3-small":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise EnvironmentError("VARIABLE_NAME is not set")

            client = OpenAI(api_key=api_key)
            self.question_encoder = client.embeddings
            self.question_tokenizer = None
            
            self.context_encoder = client.embeddings
            self.embedding_model_max_length = 8191
            self.embedding_dimension = 1536

        elif self.embedding_model == "text-embedding-ada-002":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise EnvironmentError("VARIABLE_NAME is not set")

            client = OpenAI(api_key=api_key)
            self.question_encoder = client.embeddings
            self.question_tokenizer = None
            
            self.context_encoder = client.embeddings
            self.embedding_model_max_length = 8191
            self.embedding_dimension = 1536
            
        elif self.embedding_model == "SPECTER":
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
            model = AutoModel.from_pretrained("allenai/specter", device_map="auto").to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            self.question_encoder = model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = model
            self.context_tokenizer = tokenizer
            
        elif self.embedding_model == "SPECTER2_Base":
            tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')  
            model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            specter2_model = model.to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            
            self.question_encoder = specter2_model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = specter2_model
            self.context_tokenizer = tokenizer
            
            
        elif self.embedding_model == "SPECTER2":
            tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')  
            model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            specter2_model = model.to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            
            self.question_encoder = specter2_model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = specter2_model
            self.context_tokenizer = tokenizer

        elif self.embedding_model == "SciNCL":
            tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
            model = AutoModel.from_pretrained('malteos/scincl')
            model = model.to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            
            self.question_encoder = model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = model
            self.context_tokenizer = tokenizer
            
            
    def initialize(self):
        """
        Initialize Vector Index or create embeddings for retrieving from target corpus
        """
        if self.embedding_fusion == False:
            self.format_corpus()
            os.makedirs(self.root_dir, exist_ok=True)
            if self.use_multi_source == False:
                self.faiss_index = self.build_faiss_index()
            elif self.use_multi_source == True:
                if self.use_chunked == False and self.use_full_paper_as_corpus == True:
                    self.faiss_index, self.full_paper_index = self.build_faiss_index()
                elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
                    self.faiss_index, self.chunked_index = self.build_faiss_index()
                

        elif self.embedding_fusion == True:
            self.format_corpus_for_embedding_fusion()
            os.makedirs(self.root_dir, exist_ok=True)
            self.candidate_embeddings_for_embedding_fusion = self.build_candidate_embeddings_for_embedding_fusion()

    def format_corpus_for_embedding_fusion(self):
        """
        Format corpus for experimenting late interaction and naive similarity aggregation.
        """
        def clean_text(text):
            """Remove special tokens and non-printable characters."""
            text = re.sub(r"[^\x20-\x7E]", "", text)  # Keep only printable characters
            text = text.strip()  # Remove leading/trailing spaces
            return text
        
        def truncate(text, number=8192):
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(clean_text(text), disallowed_special=())[:number]

            truncated_text = encoding.decode(tokens)

            return truncated_text
        
        if self.use_multi_source == False and self.use_chunked == True and self.use_full_paper_as_corpus == False:
            total_chunked_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                formatted_paper = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                chunked_sections = paper[f"chunked_sections_{self.chunk_unit}"]

                current_chunked_sections_list = []
                for chunked_section in chunked_sections:
                    if self.embedding_model == "text-embedding-3-small" or self.embedding_model == "text-embedding-ada-002":
                        truncated_chunked_section = truncate(chunked_section)
                        current_chunked_sections_list.append(truncated_chunked_section)
                    else:
                        current_chunked_sections_list.append(chunked_section)
                
                total_chunked_corpus.append({"Paper": formatted_paper, "chunked_sections_list": current_chunked_sections_list})


            self.corpus = total_chunked_corpus

        else:
            raise ValueError("Embedding Fusion experiment can only be conducted with single source and chunked corpus, and when not using full paper as corpus")
    
    
    
    def format_corpus(self):
        def clean_text(text):
            """Remove special tokens and non-printable characters."""
            text = re.sub(r"[^\x20-\x7E]", "", text)  # Keep only printable characters
            text = text.strip()  # Remove leading/trailing spaces
            return text
        
        def truncate(text, number=8192):
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(clean_text(text), disallowed_special=())[:number]

            truncated_text = encoding.decode(tokens)

            return truncated_text
        
        if self.use_multi_source == True:
            total_chunked_corpus = []
            total_full_paper_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                if self.use_chunked == False and self.use_full_paper_as_corpus == True:
                    arxiv_full_paper_directory = paper["full_paper_directory"]
                    with open(arxiv_full_paper_directory, "r") as file:
                        full_paper_content = file.read()

                    if self.embedding_model == "text-embedding-3-small" or self.embedding_model == "text-embedding-ada-002":
                        truncated_full_paper = truncate(full_paper_content)
                        try:
                            self.full_paper_corpus_to_paper_dict[truncated_full_paper]
                        except KeyError:
                            total_full_paper_corpus.append(truncated_full_paper)
                            self.full_paper_corpus_to_paper_dict[truncated_full_paper] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                    else:
                        try:
                            self.full_paper_corpus_to_paper_dict[full_paper_content]
                        except KeyError:
                            total_full_paper_corpus.append(full_paper_content)
                            self.full_paper_corpus_to_paper_dict[full_paper_content] = f"Title: {paper_title}\nAbstract: {paper_abstract}"

                        
                
                elif self.use_chunked == True and self.use_full_paper_as_corpus == False:     
                    chunked_sections = paper[f"chunked_sections_{self.chunk_unit}"]
                    for chunked_section in chunked_sections:
                        if self.embedding_model == "text-embedding-3-small" or self.embedding_model == "text-embedding-ada-002":
                            truncated_chunked_section = truncate(chunked_section)
                            try:
                                self.chunked_corpus_to_paper_dict[truncated_chunked_section]
                            except KeyError:
                                total_chunked_corpus.append(truncated_chunked_section)
                                self.chunked_corpus_to_paper_dict[truncated_chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                        else:
                            try:
                                self.chunked_corpus_to_paper_dict[chunked_section]
                            except KeyError:
                                total_chunked_corpus.append(chunked_section)
                                self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                    

            self.chunked_corpus = total_chunked_corpus
            self.full_paper_corpus = total_full_paper_corpus

            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                
                try:
                    formatted_total_corpus_dict[formatted_total_candidate]
                except:
                    formatted_total_corpus.append(formatted_total_candidate)
                    formatted_total_corpus_dict[formatted_total_candidate] = formatted_total_candidate
                    
            self.corpus = formatted_total_corpus
        
        elif self.use_multi_source == False and self.use_chunked == True and self.use_full_paper_as_corpus == False:
            total_chunked_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                
                chunked_sections = paper[f"chunked_sections_{self.chunk_unit}"]
                for chunked_section in chunked_sections:
                    if self.embedding_model == "text-embedding-3-small" or self.embedding_model == "text-embedding-ada-002":
                        truncated_chunked_section = truncate(chunked_section)
                        try:
                            self.chunked_corpus_to_paper_dict[truncated_chunked_section]
                        except KeyError:
                            total_chunked_corpus.append(truncated_chunked_section)
                            self.chunked_corpus_to_paper_dict[truncated_chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                    else:
                        try:
                            self.chunked_corpus_to_paper_dict[chunked_section]
                        except KeyError:
                            total_chunked_corpus.append(chunked_section)
                            self.chunked_corpus_to_paper_dict[chunked_section] = f"Title: {paper_title}\nAbstract: {paper_abstract}"

            self.corpus = total_chunked_corpus
        
        elif self.use_multi_source == False and self.use_chunked == False and self.use_full_paper_as_corpus == True:
            print("Formatting corpus for full paper for single source")
            total_full_paper_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper['abstract']
                arxiv_full_paper_directory = paper["full_paper_directory"]
                with open(arxiv_full_paper_directory, "r") as file:
                    full_paper_content = file.read()
                    
                if self.embedding_model == "text-embedding-3-small" or self.embedding_model == "text-embedding-ada-002":
                    truncated_full_paper = truncate(full_paper_content)
                    try:
                        self.full_paper_corpus_to_paper_dict[truncated_full_paper]
                    except KeyError:
                        total_full_paper_corpus.append(truncated_full_paper)
                        self.full_paper_corpus_to_paper_dict[truncated_full_paper] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
                else:
                    try:
                        self.full_paper_corpus_to_paper_dict[full_paper_content]
                    except KeyError:
                        total_full_paper_corpus.append(full_paper_content)
                        self.full_paper_corpus_to_paper_dict[full_paper_content] = f"Title: {paper_title}\nAbstract: {paper_abstract}"
            
            self.corpus = total_full_paper_corpus

        elif self.use_multi_source == False and self.use_chunked == False and self.use_full_paper_as_corpus == False:
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
        if self.use_query_optimizer == False:
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
            
        elif self.use_query_optimizer == True:
            query_paper = paper["Query"]
            query_title = query_paper["title"]
            optimized_queries = query_paper["optimized_queries"]
            formatted_query = []

            if self.include_original_retrieval == True:
                query_abstract = query_paper["abstract"]
                formatted_query.append(("Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = f"{query}"
                formatted_query.append((agent_name, formatted_optimized_query))
        
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
        if self.embedding_model == "jina-embeddings-v2-base-en":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "text-embedding-3-small":
            encoded_embeddings = []
            #for passage in passages:
            response = self.context_encoder.create(input=passages, model=self.embedding_model)
            for i in range(0, len(response.data)):
                embeddings = response.data[i].embedding
                encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "text-embedding-ada-002":
            encoded_embeddings = []
            #for passage in passages:
            response = self.context_encoder.create(input=passages, model=self.embedding_model)
            for i in range(0, len(response.data)):
                embeddings = response.data[i].embedding
                encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "SPECTER":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                    embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        elif self.embedding_model == "SPECTER2_Base":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)

        
        elif self.embedding_model == "SPECTER2":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)

        elif self.embedding_model == "SciNCL":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)


    # Function to encode a query
    def encode_query(self, query):
        def clean_text(text):
            """Remove special tokens and non-printable characters."""
            text = re.sub(r"[^\x20-\x7E]", "", text)  # Keep only printable characters
            text = text.strip()  # Remove leading/trailing spaces
            return text
        
        def truncate(text, number=8192):
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(clean_text(text), disallowed_special=())[:number]

            truncated_text = encoding.decode(tokens)

            return truncated_text
        
        if self.embedding_model == "jina-embeddings-v2-base-en":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)  # Encode on GPU
            
            embedding = embedding.reshape(1,-1)
            return embedding
        
        elif self.embedding_model == "text-embedding-3-small":
            query = truncate(query)
            response = self.question_encoder.create(input=query, model=self.embedding_model)
            embedding = response.data[0].embedding
            embedding =  np.array(embedding)
            embedding = embedding.reshape(1, -1)
            return embedding
        
        elif self.embedding_model == "text-embedding-ada-002":
            query = truncate(query)
            response = self.question_encoder.create(input=query, model=self.embedding_model)
            embedding = response.data[0].embedding
            embedding =  np.array(embedding)
            embedding = embedding.reshape(1, -1)
            return embedding
        
        elif self.embedding_model == "SPECTER":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
                
            return embedding
        
        elif self.embedding_model == "SPECTER2_Base":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()

            return embedding
        
        elif self.embedding_model == "SPECTER2":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()

            return embedding
        
        elif self.embedding_model == "SciNCL":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()

            return embedding
        

      
    
    # Step 1: Preprocess and index the corpus
    def build_faiss_index(self):
        if self.use_multi_source == False:
            if os.path.exists(self.faiss_index_path):
                logging.info(f"Loading existing FAISS index...{self.faiss_index_path}")
                index = faiss.read_index(self.faiss_index_path)
                return index

            logging.info("Building FAISS index...")
            
            # Initialize a FAISS index for L2 distance
            index = faiss.IndexFlatL2(self.embedding_dimension)

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
        
        elif self.use_multi_source == True:
            if self.use_chunked == False and self.use_full_paper_as_corpus == True:
                if os.path.exists(self.faiss_index_path):
                    logging.info("Loading existing FAISS index...")
                    index = faiss.read_index(self.faiss_index_path)

                else:
                    logging.info("Creating base FAISS corpus index...")
                    # Initialize a FAISS index for L2 distance
                    index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.corpus), self.batch_size):
                        batch = self.corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.corpus) // self.batch_size + 1}")
                        embeddings = self.encode_passages(batch)
                        index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(index, self.faiss_index_path)
                    logging.info("FAISS index built and saved.")

                if os.path.exists(self.full_paper_faiss_index_path):
                    logging.info("Loading existing FAISS corpus index with full papers...")
                    full_paper_index = faiss.read_index(self.full_paper_faiss_index_path)
                else:
                    logging.info("Creating full paper FAISS corpus index...")
                    full_paper_index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.full_paper_corpus), self.batch_size):
                        batch = self.full_paper_corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.full_paper_corpus) // self.batch_size + 1}")

                        embeddings = self.encode_passages(batch)
                        full_paper_index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(full_paper_index, self.full_paper_faiss_index_path)
                    logging.info("FAISS index built and saved.")
                    
                return index, full_paper_index
            
            elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
                if os.path.exists(self.faiss_index_path):
                    logging.info("Loading existing FAISS index...")
                    index = faiss.read_index(self.faiss_index_path)

                else:
                    logging.info("Creating base corpus index...")
                    # Initialize a FAISS index for L2 distance
                    index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.corpus), self.batch_size):
                        batch = self.corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.corpus) // self.batch_size + 1}")
                        embeddings = self.encode_passages(batch)
                        index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(index, self.faiss_index_path)
                    logging.info("FAISS index built and saved.")

                if os.path.exists(self.chunked_faiss_index_path):
                    logging.info("Loading chunked existing FAISS corpus index...")
                    chunked_index = faiss.read_index(self.chunked_faiss_index_path)
                else:
                    logging.info("Creating chunked FAISS corpus index...")
                    chunked_index = faiss.IndexFlatL2(self.embedding_dimension)

                    # Process passages in chunks to avoid memory issues
                    for i in range(0, len(self.chunked_corpus), self.batch_size):
                        batch = self.chunked_corpus[i:i + self.batch_size]
                        logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.chunked_corpus) // self.batch_size + 1}")

                        embeddings = self.encode_passages(batch)
                        chunked_index.add(embeddings)  

                    # Save the index to disk
                    faiss.write_index(chunked_index, self.chunked_faiss_index_path)
                    logging.info("FAISS index built and saved.")
                    
                return index, chunked_index
                
    
    def build_candidate_embeddings_for_embedding_fusion(self):
        logging.info("Building candidate embeddings for embedding fusion method...")

        total_embeddings = []
        # Process passages in chunks to avoid memory issues
        ## Batch size always hard coded to 1
       
        for i in range(0, len(self.corpus)):
            logging.info(f"Processing vectors for {i}th paper")
            embeddings = self.encode_passages(self.corpus[i]["chunked_sections_list"])

            current_entry_dictionary = self.corpus[i]
            current_entry_dictionary["embeddings"] = embeddings

            total_embeddings.append(current_entry_dictionary)

        total_number_of_document_vectors = 0
        for element in total_embeddings:
            total_number_of_document_vectors = len(element['embeddings'])

        logging.info(f"Number of document subvectors is {total_number_of_document_vectors}")
        logging.info(f"Number of total documents is {len(total_embeddings)}")

        return total_embeddings
    
    # Calculate similarities and retrieve top-k abstracts
    def retrieve_top_k(self, query, top_k, agent_name):     
        if self.embedding_fusion == True:
            subquery_list = []
            for subquery in query:
                subquery_embedding = self.encode_query(subquery)
                subquery_list.append(subquery_embedding)
            if self.embedding_fuse_method == "naive_aggregation":
                logging.info(f"Currently retrieving through {self.embedding_fuse_method}")
                document_score_list = []
                for index in range(0, len(self.candidate_embeddings_for_embedding_fusion)):
                    document_vectors = self.candidate_embeddings_for_embedding_fusion[index]['embeddings']
                    query_document_alignment_scores = 0
                    for subquery_embedding in subquery_list:
                        for document_subvector in document_vectors:
                            if not isinstance(document_subvector, torch.Tensor):
                                document_subvector = torch.from_numpy(document_subvector)
                                document_subvector = document_subvector.unsqueeze(0)
                            if not isinstance(subquery_embedding, torch.Tensor):
                                subquery_embedding = torch.from_numpy(subquery_embedding)
                            l2_distances = torch.cdist(subquery_embedding, document_subvector, p=2).item()
                            query_document_alignment_scores = query_document_alignment_scores + l2_distances
                    
                    document_score_list.append((self.candidate_embeddings_for_embedding_fusion[index]["Paper"], query_document_alignment_scores))

                sorted_document = sorted(document_score_list, key=lambda x: x[1])

                top_k_corpus = sorted_document[:top_k]
                
                return top_k_corpus
            
            elif self.embedding_fuse_method == "maximum_similarity":
                logging.info(f"Currently retrieving through {self.embedding_fuse_method}")
                document_score_list = []
                for index in range(0, len(self.candidate_embeddings_for_embedding_fusion)):
                    document_vectors = self.candidate_embeddings_for_embedding_fusion[index]['embeddings']
                    query_document_alignment_scores = 0
                    for subquery_embedding in subquery_list:
                        subquery_subvector_alignment_score_list = []
                        for document_subvector in document_vectors:
                            if not isinstance(document_subvector, torch.Tensor):
                                document_subvector = torch.from_numpy(document_subvector)
                                document_subvector = document_subvector.unsqueeze(0)
                            if not isinstance(subquery_embedding, torch.Tensor):
                                subquery_embedding = torch.from_numpy(subquery_embedding)
                            l2_distances = torch.cdist(subquery_embedding, document_subvector, p=2).item()
                            subquery_subvector_alignment_score_list.append(l2_distances)
                        
                        best_subquery_subvector_alignment_score = max(subquery_subvector_alignment_score_list)
                        query_document_alignment_scores = query_document_alignment_scores + best_subquery_subvector_alignment_score
                    
                    document_score_list.append((self.candidate_embeddings_for_embedding_fusion[index]["Paper"], query_document_alignment_scores))

                sorted_document = sorted(document_score_list, key=lambda x: x[1])

                top_k_corpus = sorted_document[:top_k]
                
                return top_k_corpus


        elif self.embedding_fusion == False:   
                query_embedding = self.encode_query(query)  # Encode the query
                if self.use_multi_source == False:
                    if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                        distances, indices = self.faiss_index.search(query_embedding, top_k)
                        top_k_corpus = [(self.corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
                    elif self.use_chunked == True and self.use_full_paper_as_corpus == False:
                        distances, indices = self.faiss_index.search(query_embedding, top_k)
                        top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]
                        #total_index = self.faiss_index.ntotal
                        #distances, indices = self.faiss_index.search(query_embedding, total_index) 
                        #top_k_corpus = []
                        #existing_corpus_to_paper_mapped = []
                        #for i, idx in enumerate(indices[0]):
                            #found_corpus = self.corpus[idx]
                            #found_corpus_distance = distances[0][i]

                            #corpus_to_paper_mapped = self.chunked_corpus_to_paper_dict[found_corpus]
                            #if corpus_to_paper_mapped not in existing_corpus_to_paper_mapped:
                                #top_k_corpus.append((corpus_to_paper_mapped, found_corpus_distance, found_corpus))

                            #if len(top_k_corpus) == top_k:
                                #break
                    elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
                        distances, indices = self.faiss_index.search(query_embedding, top_k)
                        top_k_corpus = [(self.full_paper_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]

                    
                elif self.use_multi_source == True:                        
                    if agent_name == "Original Retrieval":
                        distances, indices = self.faiss_index.search(query_embedding, top_k)
                        top_k_corpus = [(self.corpus[idx], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]
                    else:
                        if self.use_chunked == True and self.use_full_paper_as_corpus == False:
                            distances, indices = self.chunked_index.search(query_embedding, top_k)
                            top_k_corpus = [(self.chunked_corpus_to_paper_dict[self.chunked_corpus[idx]], distances[0][i], self.chunked_corpus[idx]) for i, idx in enumerate(indices[0])]
                        elif self.use_chunked == False and self.use_full_paper_as_corpus == True:
                            distances, indices = self.full_paper_index.search(query_embedding, top_k)
                            top_k_corpus = [(self.full_paper_corpus_to_paper_dict[self.full_paper_corpus[idx]], distances[0][i], self.full_paper_corpus[idx]) for i, idx in enumerate(indices[0])]


                        
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

            if self.use_query_optimizer == False:
                formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                    
            
            elif self.use_query_optimizer == True:
                filename = os.path.basename(files)
                paper_id = filename.removesuffix(".json")
                full_paper_filename = paper_id + ".mmd"
                
                parent_dir = os.path.dirname(args.benchmark_directory)
                benchmark_root_dir = os.path.dirname(parent_dir) 
                if args.remove_citations == False and args.remove_mentions == False:
                    root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers")
                
                if args.remove_citations == True and args.remove_mentions == True:
                    root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers_remove_citations_True_remove_mentions_True")
                    
                full_paper_path = os.path.join(root_folder_name, full_paper_filename)

                print(full_paper_path)
                evaluation_data, optimized_queries = self.QueryOptimizer.forward(full_paper_path, evaluation_data)
                
                
                formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                    

            if self.use_query_optimizer == False:
                parsed_retrieved_top_corpus = []
                agent_name = None
                top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name)
                    
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
                        
            elif self.use_query_optimizer == True:
                if self.embedding_fusion == True:
                    queries = []
                    for agent_name, current_query in formatted_query:
                        queries.append(current_query)
                    top_k_corpus = self.retrieve_top_k(queries, args.top_k, "")
                    
                    organized_retrieved_top_corpus={}
                    parsed_retrieved_top_corpus = []
                    for idx, (corpus, score) in enumerate(top_k_corpus):
                        organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": score, "Content": f"{corpus}"}
                        parsed_retrieved_top_corpus.append(corpus)

                elif self.embedding_fusion == False:  
                    organized_retrieved_top_corpus = {}
                    organized_retrieved_top_corpus_list = []
                    for agent_name, current_query in formatted_query:
                        organized_retrieved_top_corpus[agent_name] = {}
                        top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name)
                        organized_retrieved_top_corpus_list.append(top_k_corpus)
                        if self.use_multi_source == True:
                            for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                                organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                                    
                        elif self.use_multi_source == False:
                            if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                                for idx, (corpus, score) in enumerate(top_k_corpus):
                                    organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                            elif self.use_chunked == True or self.use_full_paper_as_corpus == True:
                                for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                                    organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                        
                    if self.use_multi_source == False:
                        if self.use_chunked == False and self.use_full_paper_as_corpus == False:
                            ### Fuse rankings using RRF
                            fused_scores = {}
                            for top_k_corpus in organized_retrieved_top_corpus_list:
                                for rank, (corpus, score) in enumerate(top_k_corpus):
                                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (60 + rank + 1)
                                
                            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                            
                        elif self.use_chunked == True or self.use_full_paper_as_corpus == True:
                                ### Fuse rankings using RRF
                            fused_scores = {}
                            for top_k_corpus in organized_retrieved_top_corpus_list:
                                for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (60 + rank + 1)
                                
                            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
                    elif self.use_multi_source == True:
                        ### Fuse rankings using RRF
                        fused_scores = {}
                        for top_k_corpus in organized_retrieved_top_corpus_list:
                            for rank, (corpus, score, original_content) in enumerate(top_k_corpus):
                                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (60 + rank + 1)
                            
                        sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                        parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                            
                    ############################


            
            current_result = evaluate_retrieval(parsed_retrieved_top_corpus, formatted_correct_candidates, args.top_k, args.max_top_k)
            total_results.append(current_result)
                
            if self.use_query_optimizer == False:
                organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": f"{formatted_query}"}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
                
            elif self.use_query_optimizer == True:
                queries = []
                for current_query in formatted_query:
                    query = {'agent': current_query[0], "content": current_query[1]}
                    queries.append(query)
                        
                organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": queries}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Final_Ranked_Results": parsed_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
                    
            current_result_folder_path = f"{result_folder_path}/{evaluation_data['id']}.json"
            with open(current_result_folder_path, "w") as json_file:
                json.dump(organized_results, json_file, indent=4)
                