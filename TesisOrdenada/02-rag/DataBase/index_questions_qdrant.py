import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from tqdm.asyncio import tqdm
from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
import uuid
from dotenv import load_dotenv

# Configurar logger
logger.add("index_questions_qdrant_benchmark.log", rotation="500 MB")

# Cargar variables de entorno
load_dotenv()

# Configuración para Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
BENCHMARK_FILE = "DataBase/qa_benchmark.json"
BATCH_SIZE = 100  # Número de preguntas a procesar en cada batch
VECTOR_DIMENSION = 768  # Dimensión para nomic-embed-text

# Configuración de Qdrant
path: Path = Path("qdrant_client")
qdrant_client: AsyncQdrantClient = AsyncQdrantClient(path=path)
collection_name: str = "questions-benchmark"

# Cliente Ollama usando la API de OpenAI
ollama_client = AsyncOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key='ollama',  # Clave API requerida, 'ollama' es común para Ollama local
)

async def create_collection_if_not_exists():
    """Crear la colección si no existe"""
    try:
        collections = await qdrant_client.get_collections()
        if not any(collection.name == collection_name for collection in collections.collections):
            logger.info(f"Creando colección {collection_name}...")
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=VECTOR_DIMENSION,  # Dimensión de los embeddings de nomic-embed-text
                    distance=rest.Distance.COSINE
                )
            )
            logger.info("Colección creada exitosamente")
        else:
            logger.info(f"La colección {collection_name} ya existe")
    except Exception as e:
        logger.error(f"Error creando la colección: {e}")
        raise

async def get_embedding(text: str) -> List[float]:
    """Obtener el embedding de un texto usando Ollama"""
    try:
        # Limpiar/reemplazar saltos de línea que pueden causar problemas
        cleaned_text = text.replace("\n", " ")

        # Usar el cliente OpenAI para obtener embeddings
        response = await ollama_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[cleaned_text]  # La API espera una lista de textos
        )

        # Extraer el embedding de la respuesta
        if response.data and len(response.data) > 0 and response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.error(f"Error: No embedding data received from Ollama for text: {cleaned_text[:50]}...")
            return []
    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        raise

async def process_benchmark_file(file_path: str) -> List[Dict[str, Any]]:
    """Procesar archivo de benchmark y preparar datos para indexación"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        processed_questions = []
        
        for question_entry in questions_data:
            question_id = question_entry["id"]
            question_text = question_entry["question"]
            answer = question_entry["answer"]
            
            # Crear payload simplificado solo con id, pregunta y respuesta
            payload = {
                "question_id": question_id,
                "question": question_text,
                "answer": answer
            }
            
            processed_questions.append({
                "text": question_text,
                "payload": payload
            })
        
        logger.info(f"Procesadas {len(processed_questions)} preguntas de benchmark")
        return processed_questions
    except Exception as e:
        logger.error(f"Error procesando archivo {file_path}: {e}")
        raise

def generate_valid_uuid(input_value: Union[int, str]) -> str:
    """
    Genera un UUID válido basado en un valor de entrada.
    Si el input es un número o string, lo usa como semilla para generar un UUID v5.
    """
    # Usar un namespace UUID fijo para generar UUIDs consistentes
    NAMESPACE_CUSTOM = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    
    # Convertir el input a string si es necesario
    input_str = str(input_value)
    
    # Generar un UUID v5 basado en el namespace y el input
    return str(uuid.uuid5(NAMESPACE_CUSTOM, input_str))

async def index_batch(points: List[rest.PointStruct]) -> None:
    """Indexar un batch de puntos en Qdrant"""
    try:
        await qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Batch de {len(points)} puntos indexado correctamente")
    except Exception as e:
        logger.error(f"Error indexando batch: {e}")
        raise

async def index_benchmark_questions():
    """Indexar preguntas del benchmark en Qdrant"""
    try:
        # Asegurar que la colección existe
        await create_collection_if_not_exists()
        
        # Procesar archivo de benchmark
        questions = await process_benchmark_file(BENCHMARK_FILE)
        total_questions = len(questions)
        logger.info(f"Encontradas {total_questions} preguntas en {BENCHMARK_FILE}")
        
        # Procesar en batches
        for i in range(0, total_questions, BATCH_SIZE):
            batch = questions[i:i + BATCH_SIZE]
            batch_num = i//BATCH_SIZE + 1
            total_batches = (total_questions + BATCH_SIZE - 1)//BATCH_SIZE
            logger.info(f"Procesando batch {batch_num}/{total_batches}")
            
            # Obtener embeddings para el batch
            embeddings = []
            for question in tqdm(batch, desc="Generando embeddings"):
                embedding = await get_embedding(question["text"])
                embeddings.append(embedding)
            
            # Preparar puntos para Qdrant
            points = []
            for question, embedding in zip(batch, embeddings):
                if not embedding:  # Saltar si el embedding está vacío
                    logger.warning(f"Saltando pregunta con embedding vacío: {question['text'][:50]}...")
                    continue
                
                points.append(rest.PointStruct(
                    id=generate_valid_uuid(question["payload"]["question_id"]),
                    vector=embedding,
                    payload=question["payload"]
                ))
            
            # Indexar batch en Qdrant
            if points:
                await index_batch(points)
            else:
                logger.warning("No hay puntos válidos para indexar en este batch")
            
            # Pequeña pausa para no sobrecargar el sistema
            await asyncio.sleep(1)
        
        logger.info("\n¡Indexación completada!")
        
    except Exception as e:
        logger.error(f"Error durante la indexación: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(index_benchmark_questions())
    except KeyboardInterrupt:
        logger.warning("\nIndexación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
        sys.exit(1) 