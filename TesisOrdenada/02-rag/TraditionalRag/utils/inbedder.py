"""
Custom InBedder implementation using Hugging Face models with parallel processing
"""
import torch
from torch import nn
from torch.nn.functional import gelu, cosine_similarity
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from typing import List
from loguru import logger
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import math

class InBedder:
    """
    Implementation of InBedder using Hugging Face models with parallel processing
    """
    
    def __init__(self, path='KomeijiForce/inbedder-roberta-large', device=None, batch_size=32):
        """
        Initialize InBedder with a pretrained model
        
        Args:
            path: Path to Hugging Face model
            device: Device to run the model (cuda:0, cpu, etc.)
            batch_size: Size of batches for parallel processing
        """
        # Determine device automatically if not specified
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Set number of threads for PyTorch
        n_cores = multiprocessing.cpu_count()
        torch.set_num_threads(n_cores)
        logger.info(f"Using {n_cores} CPU cores for parallel processing")
        
        # Load model
        model = AutoModelForMaskedLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Extract components
        self.model = model.roberta
        self.dense = model.lm_head.dense
        self.layer_norm = model.lm_head.layer_norm
        
        # Move to device
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.dense = self.dense.to(self.device)
        self.layer_norm = self.layer_norm.to(self.device)
        
        # Create inverse vocabulary
        self.vocab = self.tokenizer.get_vocab()
        self.vocab = {self.vocab[key]: key for key in self.vocab}
        
        # Set batch size for parallel processing
        self.batch_size = batch_size
    
    async def encode_batch(self, batch_texts: List[str], instruction: str, n_mask: int = 3):
        """
        Process a single batch of texts
        """
        if isinstance(instruction, str):
            prompts = [instruction + self.tokenizer.mask_token*n_mask for _ in batch_texts]
        elif isinstance(instruction, list):
            prompts = [inst + self.tokenizer.mask_token*n_mask for inst in instruction]
        
        # Move inputs to correct device
        inputs = self.tokenizer(batch_texts, prompts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            mask = inputs['input_ids'].eq(self.tokenizer.mask_token_id)
            outputs = self.model(**inputs)
            logits = outputs.last_hidden_state[mask]
            logits = self.layer_norm(gelu(self.dense(logits)))
            logits = logits.reshape(len(batch_texts), n_mask, -1)
            logits = logits.mean(1)
            logits = (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True)
            logits = logits.cpu()
        
        return logits
    
    async def encode(self, input_texts: List[str], instruction: str, n_mask: int = 3):
        """
        Generate contextual embeddings based on instruction using parallel processing
        
        Args:
            input_texts: List of texts to encode
            instruction: Instruction to contextualize embeddings
            n_mask: Number of mask tokens to use
            
        Returns:
            Tensor with generated embeddings
        """
        # Calculate number of batches
        num_texts = len(input_texts)
        num_batches = math.ceil(num_texts / self.batch_size)
        
        if num_batches > 1:
            logger.info(f"Processing {num_texts} texts in {num_batches} batches of size {self.batch_size}")
        
        # Create batches
        batches = [
            input_texts[i:i + self.batch_size] 
            for i in range(0, num_texts, self.batch_size)
        ]
        
        # Process batches in parallel
        tasks = [
            self.encode_batch(batch, instruction, n_mask) 
            for batch in batches
        ]
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        # Combine results
        if len(results) == 1:
            return results[0]
        else:
            return torch.cat(results, dim=0)
    
    def compute_similarity(self, embeddings1, embeddings2):
        """
        Calculate cosine similarity between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Tensor with cosine similarities
        """
        return cosine_similarity(embeddings1, embeddings2, dim=1)
    
    async def combine_embeddings(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Combine query with documents to create query-aware embeddings
        
        Args:
            query: User query
            documents: List of documents to combine with query
            
        Returns:
            Array of combined embeddings
        """
        # Generate embeddings using InBedder model
        instruction = f"Find documents relevant to: {query}"
        doc_embeddings = await self.encode(documents, instruction)
        
        # Convert to numpy if necessary
        if torch.is_tensor(doc_embeddings):
            doc_embeddings = doc_embeddings.cpu().numpy()
            
        return doc_embeddings 