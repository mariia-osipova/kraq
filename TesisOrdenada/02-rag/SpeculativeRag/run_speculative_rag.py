import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from time import perf_counter
import numpy as np
from tabulate import tabulate

# Añadir el directorio raíz al sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ahora importa usando rutas absolutas
from SpeculativeRag.main import speculative_rag

# Lista de preguntas de prueba
TEST_QUESTIONS = [
    "Can sorafenib activate AMPK?"
]

async def run_speculative_rag_test():
    """
    Ejecuta Speculative RAG con una lista de preguntas de prueba
    """
    # Cargar variables de entorno desde el directorio raíz
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # Cliente para embeddings Ollama
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    
    # Cliente principal para LLM
    llm_client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    # Clientes específicos para drafter y verifier (en este caso usamos el mismo)
    # En un caso real, podrían configurarse con diferentes endpoints o parámetros
    drafter_client = llm_client
    verifier_client = llm_client
    
    # Configuración de modelos
    EMBEDDING_MODEL = "nomic-embed-text"  # Usando nomic-embed-text de Ollama
    #DRAFTER_MODEL = "llama3-awq"  # Modelo cuantizado
    #VERIFIER_MODEL = "llama3-awq"
     
    DRAFTER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Modelo cuantizado
    VERIFIER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" # Modelo cuantizado
    
    # Parámetros para Speculative RAG
    params = {
        "embedding_model": EMBEDDING_MODEL,
        "k": 4,  # Número de clusters
        "seed": 42,
        "client": ollama_client,  # Cliente de Ollama para embeddings
        "drafter_model": DRAFTER_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "top_k": 15,  # Documentos a recuperar
        "verbose": True,
        "m":4,
        "debug_prints": True,
        "print_documents": True,
        "print_embeddings": True,
        "print_clusters": True,
        "print_drafts": True,
        "print_scores": True,
        "collection_name": "chunks",  # Usar la colección de benchmark
        "ollama_client": ollama_client,  # Cliente para embeddings
        "llm_client": llm_client,  # Cliente para LLM
        "drafter_client": drafter_client,  # Cliente específico para el drafter
        "verifier_client": verifier_client  # Cliente específico para el verifier
    }
    
    # Procesar cada pregunta
    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Procesando pregunta {i}/{len(TEST_QUESTIONS)}")
        logger.info(f"Pregunta: {question}")
        logger.info('='*80)
        
        try:
            # Clase personalizada para capturar los tiempos de ejecución y los scores
            class TimeCaptureLogger:
                def __init__(self):
                    self.times = {}
                    self.draft_times = []
                    self.verify_times = []
                    self.draft_scores = []
                    self.draft_info = []
                
                def write(self, text):
                    # Capturar tiempos de todas las fases
                    if "⏱️ Query embedding time:" in text:
                        self.times["query_embedding_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "⏱️ InBedder processing time:" in text:
                        self.times["inbedder_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "⏱️ Document retrieval time:" in text:
                        self.times["retrieval_time"] = float(text.split(":")[1].strip().split()[0])
                    elif "Time for clustering and subset generation:" in text:
                        self.times["cluster_time"] = float(text.split(":")[1].strip().split()[0])
                    
                    # Capturar tiempos de drafts
                    elif "Draft " in text and "generado en" in text:
                        time_str = text.split("generado en")[1].strip().split()[0]
                        self.draft_times.append(float(time_str))
                    
                    # Capturar tiempos de verificación
                    elif "Verificación " in text and "completada en" in text:
                        time_str = text.split("completada en")[1].strip().split()[0]
                        self.verify_times.append(float(time_str))
                    
                    # Capturar tiempos de drafting y verificación
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
                    
                    # Capturar información de scores
                    elif "Draft probability:" in text:
                        draft_prob = float(text.split(":")[1].strip().split()[0])
                        if not any(info.get('draft_prob') == draft_prob for info in self.draft_info):
                            self.draft_info.append({'draft_prob': draft_prob})
                    elif "Consistency:" in text:
                        if self.draft_info and 'consistency' not in self.draft_info[-1]:
                            consistency = float(text.split(":")[1].strip().split()[0])
                            self.draft_info[-1]['consistency'] = consistency
                    elif "Verifier probability:" in text:
                        if self.draft_info and 'verifier_prob' not in self.draft_info[-1]:
                            verifier_prob = float(text.split(":")[1].strip().split()[0])
                            self.draft_info[-1]['verifier_prob'] = verifier_prob
                    elif "Final score:" in text and not "SCORES:" in text:
                        if self.draft_info and 'final_score' not in self.draft_info[-1]:
                            final_score = float(text.split(":")[1].strip().split()[0])
                            self.draft_info[-1]['final_score'] = final_score
                            self.draft_scores.append(final_score)
                    
                    return
            
            # Capturar la salida para extraer los tiempos
            time_logger = TimeCaptureLogger()
            
            # Reemplazar temporalmente sys.stdout para capturar los tiempos
            original_stdout = sys.stdout
            sys.stdout = time_logger
            
            # Ejecutar Speculative RAG y medir tiempo total
            start_time = perf_counter()
            response, execution_time = await speculative_rag(query=question, **params)
            total_time = perf_counter() - start_time
            
            # Restaurar sys.stdout
            sys.stdout = original_stdout
            
            # Obtener y mostrar la respuesta
            logger.info("\nRESPUESTA FINAL:")
            logger.info("-"*80)
            logger.info(response)
            logger.info("-"*80)
            
            # Mostrar detalles de los scores de cada draft
            if time_logger.draft_info:
                logger.info("\n📊 SCORES DE CADA DRAFT:")
                logger.info("-"*80)
                scores_data = []
                
                for i, info in enumerate(time_logger.draft_info):
                    draft_prob = info.get('draft_prob', 'N/A')
                    consistency = info.get('consistency', 'N/A')
                    verifier_prob = info.get('verifier_prob', 'N/A')
                    final_score = info.get('final_score', 'N/A')
                    
                    scores_data.append([
                        f"Draft {i+1}",
                        f"{draft_prob:.4f}" if isinstance(draft_prob, float) else draft_prob,
                        f"{consistency:.4f}" if isinstance(consistency, float) else consistency,
                        f"{verifier_prob:.4f}" if isinstance(verifier_prob, float) else verifier_prob,
                        f"{final_score:.4f}" if isinstance(final_score, float) else final_score
                    ])
                
                # Mostrar tabla de scores usando tabulate
                scores_table = tabulate(
                    scores_data,
                    headers=["Draft", "P(Draft)", "P(Consistency)", "P(Verifier)", "Score Final"],
                    tablefmt="pretty"
                )
                logger.info(scores_table)
                
                # Mostrar el draft ganador
                if time_logger.draft_scores:
                    best_draft_idx = np.argmax(time_logger.draft_scores)
                    logger.info(f"\n🏆 El draft ganador es: Draft {best_draft_idx+1} con score {time_logger.draft_scores[best_draft_idx]:.4f}")
            
            # Mostrar un resumen de los tiempos en formato de tabla
            timing_data = []
            
            # Añadir información de tiempos secuenciales vs paralelos
            if time_logger.times:
                # Obtener tiempos de las diferentes fases
                query_embedding_time = time_logger.times.get("query_embedding_time", 0)
                inbedder_time = time_logger.times.get("inbedder_time", 0)
                retrieval_time = time_logger.times.get("retrieval_time", 0)
                cluster_time = time_logger.times.get("cluster_time", 0) if "cluster_time" in time_logger.times else 0
                draft_sequential = time_logger.times.get("draft_sequential_time", 0)
                draft_parallel = time_logger.times.get("draft_parallel_estimate", 0)
                verify_sequential = time_logger.times.get("verify_sequential_time", 0)
                verify_parallel = time_logger.times.get("verify_parallel_estimate", 0)
                
                # Calcular el tiempo total con todas las fases
                # El tiempo secuencial incluye todas las fases, con drafts y verification secuenciales
                total_sequential = query_embedding_time + inbedder_time + retrieval_time + cluster_time + draft_sequential + verify_sequential
                
                # El tiempo paralelo incluye todas las fases, pero con drafts y verification en paralelo
                total_parallel = query_embedding_time + inbedder_time + retrieval_time + cluster_time + draft_parallel + verify_parallel
                
                # Calcular el ahorro estimado utilizando ejecución paralela
                time_saved = total_sequential - total_parallel
                speedup = (total_sequential / total_parallel) if total_parallel > 0 else 0
                
                # Información sobre tiempos de ejecución
                logger.info("\n⏱️ COMPARACIÓN DE TIEMPOS (SECUENCIAL VS PARALELO):")
                logger.info("-"*80)
                
                # Crear tabla para mostrar tiempos de todas las fases
                timing_data.append(["Query Embedding", f"{query_embedding_time:.2f}s", f"{query_embedding_time:.2f}s", "0.00s"])
                timing_data.append(["Document Retrieval", f"{retrieval_time:.2f}s", f"{retrieval_time:.2f}s", "0.00s"])
                timing_data.append(["InBedder", f"{inbedder_time:.2f}s", f"{inbedder_time:.2f}s", "0.00s"])
                timing_data.append(["Clustering", f"{cluster_time:.2f}s", f"{cluster_time:.2f}s", "0.00s"])
                timing_data.append(["Drafting", f"{draft_sequential:.2f}s", f"{draft_parallel:.2f}s", f"{draft_sequential - draft_parallel:.2f}s"])
                timing_data.append(["Verification", f"{verify_sequential:.2f}s", f"{verify_parallel:.2f}s", f"{verify_sequential - verify_parallel:.2f}s"])
                timing_data.append(["TOTAL", f"{total_sequential:.2f}s", f"{total_parallel:.2f}s", f"{time_saved:.2f}s"])
                
                # Mostrar tabla usando tabulate
                table = tabulate(
                    timing_data,
                    headers=["Fase", "Tiempo Secuencial", "Tiempo Paralelo Est.", "Tiempo Ahorrado"],
                    tablefmt="pretty"
                )
                logger.info(table)
                
                # Mostrar speedup total del algoritmo
                logger.info(f"\nSpeedup total estimado del algoritmo: {speedup:.2f}x")
                
                # Análisis de porcentajes de tiempo
                logger.info("\nPorcentaje de tiempo por fase (secuencial):")
                if total_sequential > 0:
                    percentage_data = [
                        ["Query Embedding", f"{query_embedding_time:.2f}s", f"{(query_embedding_time/total_sequential)*100:.1f}%"],
                        ["Document Retrieval", f"{retrieval_time:.2f}s", f"{(retrieval_time/total_sequential)*100:.1f}%"],
                        ["InBedder", f"{inbedder_time:.2f}s", f"{(inbedder_time/total_sequential)*100:.1f}%"],
                        ["Clustering", f"{cluster_time:.2f}s", f"{(cluster_time/total_sequential)*100:.1f}%"],
                        ["Drafting", f"{draft_sequential:.2f}s", f"{(draft_sequential/total_sequential)*100:.1f}%"],
                        ["Verification", f"{verify_sequential:.2f}s", f"{(verify_sequential/total_sequential)*100:.1f}%"]
                    ]
                    percentage_table = tabulate(percentage_data, headers=["Fase", "Tiempo", "Porcentaje"], tablefmt="pretty")
                    logger.info(percentage_table)
                
                # Información detallada de tiempos individuales
                if time_logger.draft_times:
                    draft_avg = np.mean(time_logger.draft_times)
                    draft_min = np.min(time_logger.draft_times)
                    draft_max = np.max(time_logger.draft_times)
                    draft_std = np.std(time_logger.draft_times)
                    
                    logger.info("\nTiempos de drafts individuales:")
                    for i, t in enumerate(time_logger.draft_times):
                        logger.info(f"Draft {i+1}: {t:.2f}s")
                    logger.info(f"Media: {draft_avg:.2f}s, Min: {draft_min:.2f}s, Max: {draft_max:.2f}s, Desv. Estándar: {draft_std:.2f}s")
                
                if time_logger.verify_times:
                    verify_avg = np.mean(time_logger.verify_times)
                    verify_min = np.min(time_logger.verify_times)
                    verify_max = np.max(time_logger.verify_times)
                    verify_std = np.std(time_logger.verify_times)
                    
                    logger.info("\nTiempos de verificación individuales:")
                    for i, t in enumerate(time_logger.verify_times):
                        logger.info(f"Verificación {i+1}: {t:.2f}s")
                    logger.info(f"Media: {verify_avg:.2f}s, Min: {verify_min:.2f}s, Max: {verify_max:.2f}s, Desv. Estándar: {verify_std:.2f}s")
                
                # Mostrar tiempo total medido de la ejecución
                logger.info("\nTiempo total de ejecución medido (incluyendo todas las fases):")
                logger.info(f"Tiempo total medido: {total_time:.2f}s")
                logger.info(f"Tiempo total calculado (secuencial): {total_sequential:.2f}s")
                logger.info(f"Tiempo total estimado (paralelo): {total_parallel:.2f}s")
            
        except Exception as e:
            logger.error(f"Error procesando la pregunta: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Pequeña pausa entre preguntas
        if i < len(TEST_QUESTIONS):
            await asyncio.sleep(2)

def main():
    """Función principal"""
    logger.info("Iniciando pruebas de Speculative RAG con recursos locales")
    
    try:
        asyncio.run(run_speculative_rag_test())
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
    
    logger.info("Pruebas completadas")

if __name__ == "__main__":
    main() 