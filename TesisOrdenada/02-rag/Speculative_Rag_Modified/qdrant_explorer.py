"""
Explorador de la estructura de datos en Qdrant
Este script ayuda a entender la estructura exacta de los datos almacenados
en las colecciones de Qdrant para mejorar el algoritmo de recuperación.
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from qdrant_client import AsyncQdrantClient, models

# Ajustar sys.path para incluir el directorio del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Función para obtener el cliente de Qdrant
async def get_qdrant_client():
    path = Path(__file__).parent.parent / "qdrant_client"
    return AsyncQdrantClient(path=path)

async def explore_collection(collection_name: str, limit: int = 5):
    """
    Explora la estructura de una colección en Qdrant
    """
    client = None
    try:
        client = await get_qdrant_client()
        
        # 1. Verificar que la colección existe
        collections = await client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        print(f"\n=== Colecciones disponibles ({len(collection_names)}) ===")
        for name in collection_names:
            print(f"- {name}")
            
        if collection_name not in collection_names:
            print(f"\n❌ La colección '{collection_name}' no existe")
            return
            
        # 2. Obtener información de la colección
        collection_info = await client.get_collection(collection_name)
        print(f"\n=== Información de la colección '{collection_name}' ===")
        print(f"Puntos: {collection_info.config.params.vectors.size}")
        print(f"Dimensiones: {collection_info.config.params.vectors.size}")
        
        # 3. Extraer algunos puntos de muestra
        print(f"\n=== Muestra de {limit} puntos de '{collection_name}' ===")
        points = await client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        
        # 4. Analizar estructura de payload
        if not points[0]:
            print("No se encontraron puntos en la colección")
            return
            
        for i, point in enumerate(points[0]):
            print(f"\n--- Punto {i+1} (ID: {point.id}) ---")
            
            # Analizar el payload
            if point.payload:
                print("Campos en payload:")
                for key, value in point.payload.items():
                    if isinstance(value, str) and len(value) > 100:
                        preview = value[:100] + "..."
                    else:
                        preview = value
                    print(f"  {key}: {preview} ({type(value).__name__})")
            else:
                print("El punto no tiene payload")
                
            # Verificar si tiene vector
            if hasattr(point, "vector") and point.vector:
                vector_length = len(point.vector)
                print(f"Vector: {vector_length} dimensiones")
            else:
                print("El punto no tiene vector")
                
        return points[0]
            
    except Exception as e:
        print(f"Error explorando colección: {e}")
    finally:
        if client:
            await client.close()

async def test_search_similar_questions(query: str, limit: int = 5):
    """
    Prueba la búsqueda de preguntas similares
    """
    client = None
    try:
        from openai import AsyncOpenAI
        import os
        from dotenv import load_dotenv
        
        # Cargar variables de entorno
        load_dotenv(dotenv_path=project_root / ".env")
        
        # Crear cliente OpenAI
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generar embedding para la consulta
        response = await openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Conectar a Qdrant
        client = await get_qdrant_client()
        
        # Buscar en la colección de preguntas
        questions_collection = "questions"
        
        # Ver todas las preguntas disponibles primero
        print(f"\n=== Mostrando todas las preguntas en '{questions_collection}' ===")
        all_questions = await client.scroll(
            collection_name=questions_collection,
            limit=20,
            with_payload=True
        )
        
        if not all_questions[0]:
            print("No hay preguntas en la colección")
        else:
            for i, q in enumerate(all_questions[0]):
                q_text = None
                if q.payload:
                    # Intentar diferentes campos
                    for field in ["question", "text", "content"]:
                        if field in q.payload:
                            q_text = q.payload[field]
                            break
                            
                print(f"{i+1}. ID: {q.id}, Texto: {q_text}")
                print(f"   Payload completo: {q.payload}")
        
        # Buscar preguntas similares
        print(f"\n=== Buscando preguntas similares a: '{query}' ===")
        similar_questions = await client.search(
            collection_name=questions_collection,
            query_vector=query_embedding,
            limit=limit
        )
        
        if not similar_questions:
            print("No se encontraron preguntas similares")
            return
            
        # Mostrar resultados
        print(f"Se encontraron {len(similar_questions)} preguntas similares:")
        for i, q in enumerate(similar_questions):
            print(f"\n--- Pregunta similar {i+1} (Score: {q.score:.4f}) ---")
            
            # Intentar obtener el texto de la pregunta
            q_text = None
            if q.payload:
                print("Campos en payload:")
                for key, value in q.payload.items():
                    if isinstance(value, str) and len(value) > 100:
                        preview = value[:100] + "..."
                    else:
                        preview = value
                    print(f"  {key}: {preview} ({type(value).__name__})")
                    
                    # Intentar identificar el campo con la pregunta
                    if key in ["question", "text", "content"] and isinstance(value, str):
                        q_text = value
            else:
                print("La pregunta no tiene payload")
                
            print(f"Texto identificado: {q_text}")
            
            if hasattr(q, "vector") and q.vector:
                print(f"Vector disponible: Sí ({len(q.vector)} dimensiones)")
            else:
                print("Vector disponible: No")
                
        # Probar a buscar documentos para la primera pregunta similar
        if similar_questions and hasattr(similar_questions[0], "vector") and similar_questions[0].vector:
            print("\n=== Probando búsqueda de documentos con vector de pregunta similar ===")
            docs = await client.search(
                collection_name="test",
                query_vector=similar_questions[0].vector,
                limit=3
            )
            
            print(f"Se encontraron {len(docs)} documentos:")
            for i, doc in enumerate(docs):
                doc_text = doc.payload.get("text", "No text")[:100] + "..."
                print(f"{i+1}. Score: {doc.score:.4f}, Texto: {doc_text}")
                
    except Exception as e:
        print(f"Error en test_search_similar_questions: {e}")
    finally:
        if client:
            await client.close()

async def test_inbedding_with_sample_docs():
    """
    Prueba el proceso de InBedding con documentos de muestra
    """
    client = None
    try:
        from Speculative_Rag_Modified.utils.inbedder import InBedder
        import numpy as np
        import torch
        
        client = await get_qdrant_client()
        
        # Obtener algunos documentos de muestra
        print("\n=== Obteniendo documentos de muestra para InBedding ===")
        points = await client.scroll(
            collection_name="test",
            limit=5,
            with_payload=True
        )
        
        if not points[0]:
            print("No se encontraron documentos")
            return
            
        doc_texts = [p.payload.get("text", "") for p in points[0] if p.payload and "text" in p.payload]
        
        if not doc_texts:
            print("No se encontraron textos de documentos")
            return
            
        print(f"Se obtuvieron {len(doc_texts)} textos de documentos")
        
        # Probar InBedding con un query simple
        query = "What is Diagon Alley?"
        print(f"\nProbando InBedding con query: '{query}'")
        
        inbedder = InBedder(batch_size=2)
        doc_embeddings = await inbedder.encode(doc_texts, query, n_mask=2)
        
        with torch.no_grad():
            doc_embeddings = doc_embeddings.numpy() if torch.is_tensor(doc_embeddings) else doc_embeddings
        
        print(f"InBedding exitoso. Shape de resultados: {doc_embeddings.shape}")
        
    except Exception as e:
        print(f"Error en test_inbedding_with_sample_docs: {e}")
    finally:
        if client:
            await client.close()

async def main():
    print("=== Explorador de Qdrant para Speculative RAG ===")
    
    # 1. Explorar colección de documentos
    documents = await explore_collection("test")
    
    # 2. Explorar colección de preguntas
    questions = await explore_collection("questions")
    
    # 3. Probar búsqueda de preguntas similares
    await test_search_similar_questions("What is Diagon Alley?")
    
    # 4. Probar InBedding
    await test_inbedding_with_sample_docs()

if __name__ == "__main__":
    asyncio.run(main()) 