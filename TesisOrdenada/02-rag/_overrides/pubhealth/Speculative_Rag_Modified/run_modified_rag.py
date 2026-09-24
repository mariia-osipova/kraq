import asyncio
import os
import sys
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

# Ajustar el path para incluir el directorio del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importar desde el mismo directorio
from Speculative_Rag_Modified.modified_rag import modified_rag

# Cargar variables de entorno desde el directorio raíz
load_dotenv(dotenv_path=project_root / ".env")

# Configurar logger
logger.add(Path(__file__).parent / "modified_rag_logs.log", rotation="500 MB")

# Configuración
EMBEDDING_MODEL = "nomic-embed-text"  # Modelo de embeddings de Ollama
DRAFTER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Modelo vLLM
VERIFIER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Modelo vLLM

# Lista de preguntas de prueba
TEST_QUESTIONS = [
    "If Congress does not pass the renewal of the payroll tax cut before the end of the year, nearly 160 million working families will see their taxes go up by roughly $1,000."
]

async def run_modified_rag_test():
    """
    Ejecuta pruebas del RAG modificado con un conjunto de preguntas
    """
    # Crear cliente de Ollama para embeddings
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Crear cliente de vLLM para LLMs
    llm_client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    # Clientes específicos para drafter y verifier (por ahora iguales, pero podrían ser diferentes)
    drafter_client = llm_client
    verifier_client = llm_client

    # Parámetros para Modified RAG
    n_cores = multiprocessing.cpu_count()
    params = {
        "embedding_model": EMBEDDING_MODEL,
        "k": 2,  # Número de clusters
        "seed": 42,
        "client": ollama_client,  # Cliente genérico
        "drafter_model": DRAFTER_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "m": 5,  # Número de subsets de documentos
        "top_k": 10,  # Documentos a recuperar
        "verbose": True,
        "debug_prints": True,
        "print_documents": True,
        "print_embeddings": True,
        "print_clusters": True,
        "print_drafts": True,
        "print_scores": True,
        "collection_name": "chunks",  # Colección para documentos
        "questions_collection": "questions-index-finetuned",  # Colección para preguntas similares
        "ollama_client": ollama_client,  # Cliente para embeddings
        "llm_client": llm_client,  # Cliente general para LLM
        "drafter_client": drafter_client,  # Cliente específico para drafter
        "verifier_client": verifier_client  # Cliente específico para verifier
    }
    
    # Procesar cada pregunta
    total_start_time = asyncio.get_event_loop().time()
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Procesando pregunta {i}/{len(TEST_QUESTIONS)}")
        logger.info(f"Pregunta: {question}")
        logger.info('='*80)
        
        question_start_time = asyncio.get_event_loop().time()
        
        try:
            # Clase personalizada para capturar los tiempos de ejecución
            class TimeCaptureLogger:
                def __init__(self):
                    self.times = {}
                    self.draft_times = []
                    self.verify_times = []
                
                def write(self, text):
                    # Capturar tiempos de fases principales
                    if "⏱️ Query embedding time:" in text:
                        self.times["query_embedding_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "⏱️ InBedder processing time:" in text:
                        self.times["inbedder_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "⏱️ Document retrieval time:" in text:
                        self.times["retrieval_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "⏱️ Clustering time:" in text:
                        self.times["clustering_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "Similar question search time:" in text:
                        self.times["similar_question_time"] = float(text.split(":")[1].strip().split()[0])
                    
                    # Capturar tiempos de drafts
                    elif "Draft " in text and "generado en" in text:
                        time_str = text.split("generado en")[1].strip().split()[0]
                        self.draft_times.append(float(time_str))
                    
                    # Capturar tiempos de verificaciones
                    elif "Verificación " in text and "completada en" in text:
                        time_str = text.split("completada en")[1].strip().split()[0]
                        self.verify_times.append(float(time_str))
                    
                    # Capturar tiempos secuenciales y paralelos
                    elif "Drafting time (secuencial):" in text:
                        self.times["draft_sequential_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "Drafting tiempo promedio:" in text:
                        self.times["draft_average_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "Drafting estimado en paralelo:" in text:
                        self.times["draft_parallel_estimate"] = float(text.split(":")[1].strip().split()[0])
                    elif "Verification time (secuencial):" in text:
                        self.times["verify_sequential_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "Verification tiempo promedio:" in text:
                        self.times["verify_average_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "Verification estimado en paralelo:" in text:
                        self.times["verify_parallel_estimate"] = float(text.split(":")[1].strip().split()[0])
                    
                    # Capturar tiempos totales LLM
                    elif "Tiempo total LLM (secuencial):" in text:
                        self.times["total_llm_sequential"] = float(text.split(":")[1].strip().split()[0])
                    elif "Tiempo total LLM (estimado paralelo):" in text:
                        self.times["total_llm_parallel"] = float(text.split(":")[1].strip().split()[0])
                    elif "Tiempo estimado ahorrado:" in text:
                        self.times["time_saved"] = float(text.split(":")[1].strip().split()[0])
                    elif "Speedup estimado:" in text:
                        self.times["speedup"] = float(text.split(":")[1].strip().split()[0].rstrip('x'))
                    elif "Total execution time:" in text:
                        self.times["total_execution_time"] = float(text.split(":")[1].strip().split()[0])
                    
                    return
            
            # Capturar la salida para extraer los tiempos
            time_logger = TimeCaptureLogger()
            original_stdout = sys.stdout
            sys.stdout = time_logger
            
            # Ejecutar el algoritmo
            response, total_time = await modified_rag(
                query=question,
                **params
            )
            
            # Restaurar salida original
            sys.stdout = original_stdout
            
            # Obtener tiempos capturados
            question_time = asyncio.get_event_loop().time() - question_start_time
            
            # Mostrar resumen de resultados
            logger.info("\nRESPUESTA FINAL:")
            logger.info("-"*80)
            logger.info(response[0] if isinstance(response, tuple) else response)
            logger.info("-"*80)
            
            # Crear tabla de tiempos
            from tabulate import tabulate
            
            # Obtener todos los tiempos de las diferentes fases
            query_embedding_time = time_logger.times.get("query_embedding_time", 0)
            similar_question_time = time_logger.times.get("similar_question_time", 0)
            retrieval_time = time_logger.times.get("retrieval_time", 0)
            inbedder_time = time_logger.times.get("inbedder_time", 0)
            clustering_time = time_logger.times.get("clustering_time", 0)
            draft_sequential = time_logger.times.get("draft_sequential_time", 0)
            draft_parallel = time_logger.times.get("draft_parallel_estimate", 0)
            verify_sequential = time_logger.times.get("verify_sequential_time", 0)
            verify_parallel = time_logger.times.get("verify_parallel_estimate", 0)
            
            # Calcular el tiempo total con todas las fases
            # El tiempo secuencial incluye todas las fases, con drafts y verification secuenciales
            total_sequential = query_embedding_time + similar_question_time + retrieval_time + inbedder_time + clustering_time + draft_sequential + verify_sequential
            
            # El tiempo paralelo incluye todas las fases, pero con drafts y verification en paralelo
            total_parallel = query_embedding_time + similar_question_time + retrieval_time + inbedder_time + clustering_time + draft_parallel + verify_parallel
            
            # Calcular el ahorro estimado utilizando ejecución paralela
            time_saved = total_sequential - total_parallel
            speedup = (total_sequential / total_parallel) if total_parallel > 0 else 0
            
            # Crear tabla para mostrar tiempos
            timing_data = []
            timing_data.append(["Query Embedding", f"{query_embedding_time:.2f}s", f"{query_embedding_time:.2f}s", "0.00s"])
            timing_data.append(["Similar Question Search", f"{similar_question_time:.2f}s", f"{similar_question_time:.2f}s", "0.00s"])
            timing_data.append(["Document Retrieval", f"{retrieval_time:.2f}s", f"{retrieval_time:.2f}s", "0.00s"])
            timing_data.append(["InBedder", f"{inbedder_time:.2f}s", f"{inbedder_time:.2f}s", "0.00s"])
            timing_data.append(["Clustering", f"{clustering_time:.2f}s", f"{clustering_time:.2f}s", "0.00s"])
            timing_data.append(["Drafting", f"{draft_sequential:.2f}s", f"{draft_parallel:.2f}s", f"{draft_sequential - draft_parallel:.2f}s"])
            timing_data.append(["Verification", f"{verify_sequential:.2f}s", f"{verify_parallel:.2f}s", f"{verify_sequential - verify_parallel:.2f}s"])
            timing_data.append(["TOTAL", f"{total_sequential:.2f}s", f"{total_parallel:.2f}s", f"{time_saved:.2f}s"])
            
            # Mostrar tabla
            logger.info("\n⏱️ COMPARACIÓN DE TIEMPOS (SECUENCIAL VS PARALELO):")
            logger.info("-"*80)
            
            table = tabulate(
                timing_data,
                headers=["Fase", "Tiempo Secuencial", "Tiempo Paralelo Est.", "Tiempo Ahorrado"],
                tablefmt="pretty"
            )
            logger.info(table)
            
            # Mostrar speedup
            logger.info(f"\nSpeedup total estimado: {speedup:.2f}x")
            logger.info(f"Tiempo total medido: {total_time:.2f}s")
            
            # Análisis de porcentajes de tiempo
            logger.info("\nPorcentaje de tiempo por fase (secuencial):")
            if total_sequential > 0:
                percentage_data = [
                    ["Query Embedding", f"{query_embedding_time:.2f}s", f"{(query_embedding_time/total_sequential)*100:.1f}%"],
                    ["Similar Question Search", f"{similar_question_time:.2f}s", f"{(similar_question_time/total_sequential)*100:.1f}%"],
                    ["Document Retrieval", f"{retrieval_time:.2f}s", f"{(retrieval_time/total_sequential)*100:.1f}%"],
                    ["InBedder", f"{inbedder_time:.2f}s", f"{(inbedder_time/total_sequential)*100:.1f}%"],
                    ["Clustering", f"{clustering_time:.2f}s", f"{(clustering_time/total_sequential)*100:.1f}%"],
                    ["Drafting", f"{draft_sequential:.2f}s", f"{(draft_sequential/total_sequential)*100:.1f}%"],
                    ["Verification", f"{verify_sequential:.2f}s", f"{(verify_sequential/total_sequential)*100:.1f}%"]
                ]
                percentage_table = tabulate(percentage_data, headers=["Fase", "Tiempo", "Porcentaje"], tablefmt="pretty")
                logger.info(percentage_table)
            
            # Información detallada de tiempos individuales
            if time_logger.draft_times:
                draft_avg = sum(time_logger.draft_times) / len(time_logger.draft_times)
                draft_min = min(time_logger.draft_times)
                draft_max = max(time_logger.draft_times)
                import numpy as np
                draft_std = np.std(time_logger.draft_times)
                
                logger.info("\nTiempos de drafts individuales:")
                for i, t in enumerate(time_logger.draft_times):
                    logger.info(f"Draft {i+1}: {t:.2f}s")
                logger.info(f"Media: {draft_avg:.2f}s, Min: {draft_min:.2f}s, Max: {draft_max:.2f}s, Desv. Estándar: {draft_std:.2f}s")
            
            if time_logger.verify_times:
                verify_avg = sum(time_logger.verify_times) / len(time_logger.verify_times)
                verify_min = min(time_logger.verify_times)
                verify_max = max(time_logger.verify_times)
                verify_std = np.std(time_logger.verify_times)
                
                logger.info("\nTiempos de verificación individuales:")
                for i, t in enumerate(time_logger.verify_times):
                    logger.info(f"Verificación {i+1}: {t:.2f}s")
                logger.info(f"Media: {verify_avg:.2f}s, Min: {verify_min:.2f}s, Max: {verify_max:.2f}s, Desv. Estándar: {verify_std:.2f}s")
            
        except Exception as e:
            logger.error(f"Error procesando la pregunta: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Pequeña pausa entre preguntas
        if i < len(TEST_QUESTIONS):
            await asyncio.sleep(2)
    
    total_time = asyncio.get_event_loop().time() - total_start_time
    logger.info(f"\nTiempo total de ejecución: {total_time:.2f} segundos")
    logger.info(f"Tiempo promedio por pregunta: {total_time/len(TEST_QUESTIONS):.2f} segundos")

if __name__ == "__main__":
    asyncio.run(run_modified_rag_test()) 