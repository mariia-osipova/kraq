import asyncio
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
logger.add("index_chunks_qdrant.log", rotation="500 MB")

# Cargar variables de entorno
load_dotenv()

# Configuración para Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHUNKS_DIR = "DataBase/chunks"
VECTOR_DIMENSION = 768  # Dimensión para nomic-embed-text

# Configuración de Qdrant
path: Path = Path("qdrant_client")
qdrant_client: AsyncQdrantClient = AsyncQdrantClient(path=path)
collection_name: str = "chunks"

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

async def read_chunk_files() -> List[Dict[str, Any]]:
    """Leer archivos de chunks y preparar datos para indexación"""
    try:
        chunks_dir = Path(CHUNKS_DIR)
        if not chunks_dir.exists():
            logger.error(f"El directorio {CHUNKS_DIR} no existe. Creándolo...")
            chunks_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        # Obtener todos los archivos .txt en la carpeta chunks
        chunk_files = list(chunks_dir.glob("*.txt"))
        
        if not chunk_files:
            logger.warning(f"No se encontraron archivos .txt en {CHUNKS_DIR}")
            return []
        
        logger.info(f"Encontrados {len(chunk_files)} archivos de chunks")
        
        processed_chunks = []
        
        for chunk_file in chunk_files:
            try:
                # Leer contenido del archivo
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_text = f.read().strip()
                
                if not chunk_text:
                    logger.warning(f"Archivo vacío: {chunk_file}. Saltando...")
                    continue
                
                # Crear payload con la información del chunk
                payload = {
                    "filename": chunk_file.name,
                    "text": chunk_text,
                    "path": str(chunk_file),
                    "size_bytes": chunk_file.stat().st_size,
                    "created_at": chunk_file.stat().st_ctime,
                    "modified_at": chunk_file.stat().st_mtime
                }
                
                processed_chunks.append({
                    "text": chunk_text,
                    "payload": payload
                })
                
            except Exception as e:
                logger.error(f"Error procesando archivo {chunk_file}: {e}")
                continue
        
        logger.info(f"Procesados {len(processed_chunks)} chunks")
        return processed_chunks
    except Exception as e:
        logger.error(f"Error procesando archivos de chunks: {e}")
        raise

def generate_valid_uuid(input_value: Union[str, bytes]) -> str:
    """
    Genera un UUID válido basado en un valor de entrada.
    Para los chunks, usamos el nombre del archivo o su contenido.
    """
    # Usar un namespace UUID fijo para generar UUIDs consistentes
    NAMESPACE_CUSTOM = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
    
    # Convertir el input a string si es necesario
    input_str = str(input_value)
    
    # Generar un UUID v5 basado en el namespace y el input
    return str(uuid.uuid5(NAMESPACE_CUSTOM, input_str))

async def index_chunks():
    """Indexar todos los chunks de una vez en Qdrant"""
    try:
        # Asegurar que la colección existe
        await create_collection_if_not_exists()
        
        # Leer archivos de chunks
        chunks = await read_chunk_files()
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            logger.warning("No hay chunks para indexar. Asegúrate de que los archivos .txt estén en la carpeta chunks/")
            return
        
        logger.info(f"Se indexarán {total_chunks} chunks de una sola vez")
        
        # Obtener embeddings para todos los chunks
        embeddings = []
        for chunk in tqdm(chunks, desc="Generando embeddings para todos los chunks"):
            embedding = await get_embedding(chunk["text"])
            embeddings.append(embedding)
        
        logger.info(f"Embeddings generados para {len(embeddings)} chunks")
        
        # Preparar todos los puntos para Qdrant
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            if not embedding:  # Saltar si el embedding está vacío
                logger.warning(f"Saltando chunk con embedding vacío: {chunk['payload']['filename']}...")
                continue
            
            # Usar el nombre del archivo como base para el ID
            chunk_id = generate_valid_uuid(chunk["payload"]["filename"])
            
            # Si hay duplicados, añadir contenido truncado para diferenciarlo
            if any(p.id == chunk_id for p in points):
                truncated_content = chunk["text"][:50]
                chunk_id = generate_valid_uuid(f"{chunk['payload']['filename']}_{truncated_content}")
            
            points.append(rest.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=chunk["payload"]
            ))
        
        # Indexar todos los puntos de una vez
        if points:
            logger.info(f"Indexando {len(points)} puntos en Qdrant")
            await qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"¡Se indexaron {len(points)} chunks exitosamente!")
        else:
            logger.warning("No hay puntos válidos para indexar")
        
        logger.info("\n¡Indexación de chunks completada!")
        
    except Exception as e:
        logger.error(f"Error durante la indexación de chunks: {e}")
        raise
    finally:
        # Cerrar cliente de Qdrant
        await qdrant_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(index_chunks())
    except KeyboardInterrupt:
        logger.warning("\nIndexación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {e}")
        sys.exit(1)
