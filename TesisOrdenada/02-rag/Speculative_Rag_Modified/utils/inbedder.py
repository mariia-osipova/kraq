"""
Custom InBedder implementation using Hugging Face models with sequential processing (one-by-one)
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
import math
import time

class InBedder:
    """
    Implementation of InBedder using Hugging Face models with one-by-one sequential processing on GPU
    """
    
    def __init__(self, path='KomeijiForce/inbedder-roberta-large', device='cuda:1', batch_size=1):
        """
        Initialize InBedder with a pretrained model
        
        Args:
            path: Path to Hugging Face model
            device: Device to run the model (forced to cuda:1)
            batch_size: Not used as we process one document at a time
        """
        # Forzar uso de GPU si está disponible
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = torch.device('cuda:1')
                logger.info("Usando GPU:1 para InBedder")
            else:
                self.device = torch.device('cuda:0')
                logger.info("GPU:1 no disponible, usando GPU:0 en su lugar")
        else:
            self.device = torch.device('cpu')
            logger.warning("No hay GPUs disponibles, usando CPU en su lugar")
        
        # Configurar PyTorch para uso de un solo thread por proceso
        torch.set_num_threads(2)  # Mínimo de threads
        logger.info(f"InBedder: usando {2} threads para procesamiento")
        
        # Cargar modelo
        logger.info(f"Cargando modelo InBedder desde {path}...")
        model = AutoModelForMaskedLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Extraer componentes
        self.model = model.roberta
        self.dense = model.lm_head.dense
        self.layer_norm = model.lm_head.layer_norm
        
        # Mover a device
        logger.info(f"Moviendo modelo InBedder a {self.device}")
        self.model = self.model.to(self.device)
        self.dense = self.dense.to(self.device)
        self.layer_norm = self.layer_norm.to(self.device)
        
        # Crear vocabulario inverso
        self.vocab = self.tokenizer.get_vocab()
        self.vocab = {self.vocab[key]: key for key in self.vocab}
        
        logger.info("InBedder inicializado para procesamiento secuencial uno a uno")
    
    async def encode_single_text(self, text: str, instruction: str, n_mask: int = 3):
        """
        Procesa un solo texto a la vez
        """
        # Crear prompt con instruction y máscaras
        prompt = instruction + self.tokenizer.mask_token * n_mask
        
        # Tokenizar
        inputs = self.tokenizer([text], [prompt], padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            mask = inputs['input_ids'].eq(self.tokenizer.mask_token_id)
            outputs = self.model(**inputs)
            logits = outputs.last_hidden_state[mask]
            logits = self.layer_norm(gelu(self.dense(logits)))
            logits = logits.reshape(1, n_mask, -1)  # Reshape para un solo texto
            logits = logits.mean(1)
            logits = (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True)
            logits = logits.cpu()
        
        # Liberar memoria CUDA
        torch.cuda.empty_cache()
        
        return logits
    
    async def encode(self, input_texts: List[str], instruction: str, n_mask: int = 3):
        """
        Genera embeddings contextuales procesando textos en batches en paralelo
        
        Args:
            input_texts: Lista de textos a codificar
            instruction: Instrucción para contextualizar los embeddings
            n_mask: Número de tokens de máscara a usar
            
        Returns:
            Tuple de (Tensor con embeddings generados, tiempo transcurrido)
        """
        num_texts = len(input_texts)
        batch_size = 30  # Tamaño de batch fijo en 30
        
        # Calcular número de batches
        num_batches = math.ceil(num_texts / batch_size)
        logger.info(f"InBedder: procesando {num_texts} textos en {num_batches} batches de hasta {batch_size} textos en paralelo")
        
        all_results = []
        total_start_time = time.time()  # Medir tiempo total
        
        # Procesar por batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_texts)
            batch_texts = input_texts[start_idx:end_idx]
            batch_size_actual = len(batch_texts)
            
            logger.info(f"InBedder: procesando batch {batch_idx+1}/{num_batches} con {batch_size_actual} textos")
            start_time = time.time()
            
            # Procesar textos del batch en paralelo
            tasks = [self.encode_single_text(text, instruction, n_mask) for text in batch_texts]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            elapsed = time.time() - start_time
            logger.info(f"InBedder: batch {batch_idx+1}/{num_batches} completado en {elapsed:.2f} segundos")
            
            # Pequeña pausa entre batches para evitar sobrecarga y permitir liberación de memoria
            if batch_idx < num_batches - 1:
                await asyncio.sleep(0.1)
                torch.cuda.empty_cache()  # Liberar memoria CUDA entre batches
        
        total_elapsed = time.time() - total_start_time
        
        # Combinar resultados
        if all_results:
            return torch.cat(all_results, dim=0), total_elapsed
        else:
            logger.warning("No se generaron embeddings. Lista de textos vacía.")
            return torch.zeros((0, self.model.config.hidden_size)), total_elapsed
    
    def compute_similarity(self, embeddings1, embeddings2):
        """
        Calcula similitud coseno entre dos conjuntos de embeddings
        
        Args:
            embeddings1: Primer conjunto de embeddings
            embeddings2: Segundo conjunto de embeddings
            
        Returns:
            Tensor con similitudes coseno
        """
        return cosine_similarity(embeddings1, embeddings2, dim=1)
    
    async def combine_embeddings(self, query: str, documents: List[str]) -> np.ndarray:
        """
        Combina query con documentos para crear embeddings conscientes de la consulta
        
        Args:
            query: Consulta del usuario
            documents: Lista de documentos a combinar con la consulta
            
        Returns:
            Array de embeddings combinados
        """
        # Generar embeddings usando modelo InBedder
        instruction = f"Find documents relevant to: {query}"
        doc_embeddings, _ = await self.encode(documents, instruction)
        
        # Convertir a numpy si es necesario
        if torch.is_tensor(doc_embeddings):
            doc_embeddings = doc_embeddings.cpu().numpy()
            
        return doc_embeddings 