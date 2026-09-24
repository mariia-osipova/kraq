import os
import json
import time
import asyncio
import numpy as np
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Optional
from qdrant_client import AsyncQdrantClient

# Ajustar sys.path para incluir el directorio del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importar usando ruta absoluta
from Speculative_Rag_Modified.modified_rag import modified_rag

# Configurar logger
logger.add(Path(__file__).parent / "logs/benchmark_modified_qdrant.log", rotation="500 MB")

# Cargar variables de entorno desde el directorio raíz
load_dotenv(dotenv_path=project_root / ".env")

# Configuración para modelos locales
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
EMBEDDING_MODEL = "nomic-embed-text"  # Usando nomic-embed-text de Ollama
DRAFTER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Modelo cuantizado
VERIFIER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Mismo modelo para verificación

# Configuración de Qdrant
QUESTIONS_COLLECTION = "questions-benchmark"  # Colección para las preguntas
RETRIEVAL_COLLECTION = "chunks"               # Colección para la recuperación de documentos
SIMILAR_QUESTIONS_COLLECTION = "questions-index-finetuned"  # Colección para preguntas similares

# Modelo para el cálculo de similitud coseno
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1')

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calcula la similitud coseno entre dos textos
    """
    # Verificar que los textos no sean None o vacíos
    if not text1 or not text2:
        logger.warning("Uno de los textos está vacío o es None. Retornando similitud 0.0")
        return 0.0
        
    try:
        # Generar embeddings
        embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
        
        # Calcular similitud coseno
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity
    except Exception as e:
        logger.error(f"Error calculando similitud coseno: {e}")
        return 0.0

def calculate_bert_score(candidate: str, reference: str) -> Tuple[float, float, float]:
    """
    Calcula BERTScore (Precision, Recall, F1) entre respuesta candidata y referencia
    """
    try:
        # Specify device='cuda:1' to use GPU 1
        P, R, F1 = score([candidate], [reference], lang="en", verbose=False, device='cuda:1')
        return P.item(), R.item(), F1.item()
    except Exception as e:
        logger.error(f"Error al calcular BERTScore: {e}")
        return 0.0, 0.0, 0.0

async def evaluate_correctness(
    question: str,
    generated_answer: str,
    reference_answer: str,
    answer_type: str = "answer",
    client: AsyncOpenAI = None
) -> int:
    """
    Evalúa si la respuesta generada es correcta para preguntas de tipo trivia simple
    usando exclusivamente el LLM para el juicio
    """
    try:
        # Preparar el prompt según el tipo de respuesta de referencia
        
            # Para respuesta única
        prompt = f"""You are an expert evaluator for question answering systems. Your task is to determine if the generated answer correctly responds to the question according to the reference answer.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

The reference answer represents the truth. The label of the generated answer must match the reference answer to be considered correct. If the generated answer is more specific but the core meaning is the same, it is also considered correct.

Respond with ONLY a single digit:
1 - CORRECT: 
0 - INCORRECT: 

Your verdict (just the digit 1 or 0):"""
        
        # Usar el modelo para la evaluación
        response = await client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        
        answer_text = response.choices[0].message.content.strip()
        return 1 if answer_text.startswith("1") else 0
            
    except Exception as e:
        logger.error(f"Error evaluando corrección: {e}")
        return 0  # Por defecto, si hay error, consideramos la respuesta incorrecta

async def get_questions_from_qdrant(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Obtiene preguntas directamente desde la colección de Qdrant
    
    Args:
        limit: Número máximo de preguntas a recuperar
        
    Returns:
        Lista de preguntas con sus respuestas
    """
    try:
        logger.info(f"Obteniendo hasta {limit} preguntas de Qdrant de la colección questions-benchmark")
        
        # Crear cliente Qdrant
        qdrant_path = project_root / "qdrant_client"
        qdrant_client = AsyncQdrantClient(path=qdrant_path)
        
        # Recuperar puntos de la colección de preguntas
        scroll_response = await qdrant_client.scroll(
            collection_name="questions-benchmark",
            limit=limit * 2,  # Recuperamos más para compensar las que filtraremos
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_response[0] if isinstance(scroll_response, tuple) else scroll_response.points
        
        # Filtrar a solo las preguntas que tienen campo "answer"
        questions = []
        skipped_questions = 0
        
        for point in points:
            question_data = {
                "question_id": str(point.payload.get("question_id", "")),
                "question": point.payload.get("question", "")
            }
            
            # Solo procesar preguntas con campo "answer"
            if "answer" in point.payload and point.payload["answer"]:
                answer = point.payload["answer"]
                # Mantener el campo original
                question_data["answer"] = answer
                
                # Añadir como exact_answer e ideal_answer para mantener compatibilidad
                if isinstance(answer, list):
                    question_data["exact_answer"] = answer
                else:
                    question_data["exact_answer"] = answer
                
                question_data["ideal_answer"] = answer if isinstance(answer, str) else str(answer)
                
                questions.append(question_data)
            else:
                skipped_questions += 1
        
        await qdrant_client.close()
        
        logger.info(f"Se obtuvieron {len(questions)} preguntas válidas de tipo 'answer' de Qdrant")
        if skipped_questions > 0:
            logger.info(f"Se saltaron {skipped_questions} preguntas sin campo 'answer'")
        
        # Si tenemos más preguntas que el límite, recortamos
        if len(questions) > limit:
            logger.info(f"Limitando resultado a {limit} preguntas")
            questions = questions[:limit]
            
        return questions
        
    except Exception as e:
        logger.error(f"Error obteniendo preguntas de Qdrant: {e}")
        return []

async def run_benchmark_with_qdrant(limit: int = 1000, questions=None, k=2, m=5, top_k=10, results_dir=None):
    """
    Ejecuta el benchmark usando las preguntas almacenadas en Qdrant
    Args:
        limit: Número máximo de preguntas a evaluar
        questions: Lista de preguntas a usar (opcional)
        k, m, top_k: parámetros para Modified RAG
    """
    logger.info(f"Iniciando benchmark de Modified RAG con {limit} preguntas de Qdrant")
    
    # Inicializar clientes
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    ollama_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    # Clientes específicos para drafter y verifier (por ahora usan el mismo, pero podrían ser diferentes)
    drafter_client = vllm_client
    verifier_client = vllm_client
    
    # Crear directorios para resultados
    if results_dir is None:
        results_dir = Path(__file__).parent / "benchmark_results_speculative_qdrant"  # o el nombre correspondiente
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Crear subdirectorios para cada tipo de respuesta
    ideal_dir = results_dir / "ideal_answers"
    exact_dir = results_dir / "exact_answers"
    ideal_dir.mkdir(exist_ok=True)
    exact_dir.mkdir(exist_ok=True)
    
    # Verificar si existen checkpoints previos
    ideal_checkpoint = ideal_dir / "partial_results.json"
    exact_checkpoint = exact_dir / "partial_results.json"
    
    # Cargar resultados previos si existen
    ideal_results = []
    exact_results = []
    processed_question_ids = set()
    
    if ideal_checkpoint.exists():
        try:
            with open(ideal_checkpoint, "r", encoding="utf-8") as f:
                ideal_results = json.load(f)
                logger.info(f"Cargados {len(ideal_results)} resultados previos de respuestas ideales")
                # Añadir IDs de preguntas ya procesadas
                processed_question_ids.update([r["question_id"] for r in ideal_results])
        except Exception as e:
            logger.error(f"Error cargando checkpoint de respuestas ideales: {e}")
            ideal_results = []
    
    if exact_checkpoint.exists():
        try:
            with open(exact_checkpoint, "r", encoding="utf-8") as f:
                exact_results = json.load(f)
                logger.info(f"Cargados {len(exact_results)} resultados previos de respuestas exactas")
                # Añadir IDs de preguntas ya procesadas
                processed_question_ids.update([r["question_id"] for r in exact_results])
        except Exception as e:
            logger.error(f"Error cargando checkpoint de respuestas exactas: {e}")
            exact_results = []
    
    # Obtener preguntas de Qdrant
    if questions is None:
        all_questions = await get_questions_from_qdrant(limit=limit)
    else:
        all_questions = questions[:limit]
    
    if not all_questions:
        logger.error("No se pudieron obtener preguntas de Qdrant. Abortando benchmark.")
        return
    
    # Filtrar preguntas que ya se procesaron
    questions = []
    for q in all_questions:
        if q["question_id"] not in processed_question_ids:
            questions.append(q)
    
    total_questions = len(all_questions)
    remaining_questions = len(questions)
    
    logger.info(f"Total de preguntas: {total_questions}")
    logger.info(f"Preguntas ya procesadas: {total_questions - remaining_questions}")
    logger.info(f"Preguntas restantes por procesar: {remaining_questions}")
    
    # Si ya se procesaron todas las preguntas, mostrar mensaje y salir
    if remaining_questions == 0:
        logger.info("Todas las preguntas ya fueron procesadas. Generando informes finales.")
        # Guardar todos los resultados
        with open(ideal_dir / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(ideal_results, f, indent=2, ensure_ascii=False)
        
        with open(exact_dir / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(exact_results, f, indent=2, ensure_ascii=False)
        
        # Calcular estadísticas para cada tipo de respuesta
        await generate_stats_report(ideal_results, ideal_dir, "Ideal Answers")
        await generate_stats_report(exact_results, exact_dir, "Exact Answers")
        
        logger.info(f"Benchmark completado. Resultados guardados en {results_dir}")
        return
    
    # Procesar cada pregunta restante, generando una única respuesta para evaluar con ambos tipos de referencia
    for i, question_data in enumerate(tqdm(questions, desc="Evaluando preguntas")):
        question = question_data["question"]
        question_id = question_data.get("question_id", "unknown")
    
        logger.info(f"Procesando pregunta {i+1}/{len(questions)} (ID: {question_id}): {question[:50]}...")
        
        # Ejecutar Modified RAG una sola vez para obtener respuesta
        has_response = False
        response = ""
        execution_times = {}
        inbedder_time = 0
        retrieval_time = 0
        similar_question_time = 0
        total_time = 0
    
        # Solo ejecutar el algoritmo si tiene al menos una respuesta de referencia
        if "ideal_answer" in question_data or "exact_answer" in question_data:
            try:
                # Clase personalizada para capturar los tiempos de ejecución
                class TimeCaptureLogger:
                    def __init__(self):
                        self.times = {}
                        self.draft_times = []
                        self.verify_times = []
                    
                    def write(self, text):
                        # Capturar tiempos básicos
                        if "⏱️ Query embedding time:" in text:
                            self.times["query_embedding_time"] = float(text.split(":")[1].strip().split()[0])
                        elif "⏱️ InBedder processing time:" in text:
                            self.times["inbedder_time"] = float(text.split(":")[1].strip().split()[0])
                        elif "⏱️ Document retrieval time:" in text:
                            self.times["retrieval_time"] = float(text.split(":")[1].strip().split()[0])
                        elif "similar question search time:" in text.lower():
                            self.times["similar_question_time"] = float(text.split(":")[1].strip().split()[0])
                        elif "⏱️ Clustering time:" in text:
                            self.times["clustering_time"] = float(text.split(":")[1].strip().split()[0])
                        
                        # Capturar tiempos de drafts individuales
                        elif "Draft " in text and "generado en" in text:
                            time_str = text.split("generado en")[1].strip().split()[0]
                            self.draft_times.append(float(time_str))
                        
                        # Capturar tiempos de verificaciones individuales
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
                        
                        return
                
                # Capturar la salida para extraer los tiempos
                time_logger = TimeCaptureLogger()
                
                # Configurar parámetros para Modified RAG
                params = {
                    "embedding_model": EMBEDDING_MODEL,
                    "k": k,
                    "seed": 23,
                    "client": ollama_client,
                    "drafter_model": DRAFTER_MODEL,
                    "verifier_model": VERIFIER_MODEL,
                    "m": m,
                    "top_k": top_k,
                    "verbose": False,
                    "debug_prints": True,
                    "collection_name": RETRIEVAL_COLLECTION,  # Colección para documentos
                    "questions_collection": SIMILAR_QUESTIONS_COLLECTION,  # Colección para preguntas similares
                    "ollama_client": ollama_client,  # Cliente para embeddings
                    "llm_client": vllm_client,  # Cliente general para LLM
                    "drafter_client": drafter_client,  # Cliente específico para drafter
                    "verifier_client": verifier_client  # Cliente específico para verifier
                }
                
                # Reemplazar temporalmente sys.stdout para capturar los tiempos
                original_stdout = sys.stdout
                sys.stdout = time_logger
                
                # Ejecutar Modified RAG una sola vez
                response, total_time = await modified_rag(query=question, **params)
                
                has_response = True
                
                # Restaurar sys.stdout
                sys.stdout = original_stdout
                
                # Obtener los tiempos capturados internos del algoritmo
                execution_times = time_logger.times
                inbedder_time = execution_times.get("inbedder_time", 0)
                retrieval_time = execution_times.get("retrieval_time", 0)
                similar_question_time = execution_times.get("similar_question_time", 0)
                
            except Exception as e:
                logger.error(f"Error ejecutando Modified RAG: {e}")
                response = "Error generando respuesta"
                sys.stdout = original_stdout
        
            # Evaluar la misma respuesta contra la respuesta ideal si está disponible
            if "ideal_answer" in question_data and has_response:
                eval_start_time = time.time()
                ideal_reference = question_data["ideal_answer"]
        
                # Calcular métricas
                cosine_sim = calculate_cosine_similarity(response, ideal_reference)
                precision, recall, f1 = calculate_bert_score(response, ideal_reference)
        
                # Evaluar corrección usando modelo local
                is_correct = await evaluate_correctness(
                    question=question,
                    generated_answer=response,
                    reference_answer=ideal_reference,
                    answer_type="ideal_answer",
                    client=verifier_client
                )
                
                eval_time = time.time() - eval_start_time
                logger.debug(f"Tiempo de evaluación para respuesta ideal: {eval_time:.2f}s")
                
                # Crear diccionario con resultados, usando solo el tiempo de ejecución del algoritmo
                ideal_result = {
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": ideal_reference,
                    "answer_type": "ideal_answer",
                    "generated_answer": response,
                    "cosine_similarity": cosine_sim,
                    "bert_score_precision": precision,
                    "bert_score_recall": recall,
                    "bert_score_f1": f1,
                    "is_correct": is_correct,
                    "total_time": total_time,
                    "query_embedding_time": execution_times.get("query_embedding_time", 0),
                    "inbedder_time": execution_times.get("inbedder_time", 0),
                    "retrieval_time": execution_times.get("retrieval_time", 0),
                    "similar_question_time": execution_times.get("similar_question_time", 0),
                    "clustering_time": execution_times.get("clustering_time", 0),
                    "draft_sequential_time": execution_times.get("draft_sequential_time", 0),
                    "draft_average_time": execution_times.get("draft_average_time", 0),
                    "draft_parallel_estimate": execution_times.get("draft_parallel_estimate", 0),
                    "verify_sequential_time": execution_times.get("verify_sequential_time", 0),
                    "verify_average_time": execution_times.get("verify_average_time", 0),
                    "verify_parallel_estimate": execution_times.get("verify_parallel_estimate", 0)
                }
                
                ideal_results.append(ideal_result)
                
                # Guardar resultados parciales después de cada 5 preguntas
                if len(ideal_results) % 5 == 0:
                    with open(ideal_dir / "partial_results.json", "w", encoding="utf-8") as f:
                        json.dump(ideal_results, f, indent=2, ensure_ascii=False)
            
            # Evaluar la misma respuesta contra las respuestas exactas si están disponibles
            if "exact_answer" in question_data and has_response:
                eval_start_time = time.time()
                exact_references = question_data["exact_answer"]
                
                # Evaluar en una sola vez usando toda la lista de respuestas exactas
                # Calculamos similitud coseno con el mejor elemento individual para mantener coherencia en métricas
                if isinstance(exact_references, list) and len(exact_references) > 0:
                    # Para cosine similarity, usamos la mejor puntuación individual
                    best_cosine_sim = max([calculate_cosine_similarity(response, str(ref)) for ref in exact_references])
                    
                    # Para BERTScore, hacemos una sola evaluación con la referencia más larga o representativa
                    # Otra opción sería concatenar todas las referencias con separador
                    representative_ref = str(max(exact_references, key=lambda x: len(str(x))))  # Tomamos la más larga
                    precision, recall, f1 = calculate_bert_score(response, representative_ref)
                    
                    # Para la evaluación de corrección, pasamos toda la lista al evaluador
                    exact_is_correct = await evaluate_correctness(
                        question=question,
                        generated_answer=response,
                        reference_answer=exact_references,  # Pasamos la lista completa 
                        answer_type="exact_answer",
                        client=verifier_client
                    )
                else:
                    # Si solo hay una respuesta exacta o no es una lista
                    single_ref = exact_references[0] if isinstance(exact_references, list) else exact_references
                    single_ref_str = str(single_ref)
                    best_cosine_sim = calculate_cosine_similarity(response, single_ref_str)
                    precision, recall, f1 = calculate_bert_score(response, single_ref_str)
                    exact_is_correct = await evaluate_correctness(
                        question=question,
                        generated_answer=response,
                        reference_answer=single_ref,
                        answer_type="exact_answer",
                        client=verifier_client
                    )
                
                eval_time = time.time() - eval_start_time
                logger.debug(f"Tiempo de evaluación para respuesta exacta: {eval_time:.2f}s")
                
                # Crear diccionario con los resultados
                exact_result = {
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": exact_references,  # Guardamos todas las referencias
                    "answer_type": "exact_answer",
                    "generated_answer": response,
                    "cosine_similarity": best_cosine_sim,
                    "bert_score_precision": precision,
                    "bert_score_recall": recall,
                    "bert_score_f1": f1,
                    "is_correct": exact_is_correct,
                    "total_time": total_time,
                    "query_embedding_time": execution_times.get("query_embedding_time", 0),
                    "inbedder_time": execution_times.get("inbedder_time", 0),
                    "retrieval_time": execution_times.get("retrieval_time", 0),
                    "similar_question_time": execution_times.get("similar_question_time", 0),
                    "clustering_time": execution_times.get("clustering_time", 0),
                    "draft_sequential_time": execution_times.get("draft_sequential_time", 0),
                    "draft_average_time": execution_times.get("draft_average_time", 0),
                    "draft_parallel_estimate": execution_times.get("draft_parallel_estimate", 0),
                    "verify_sequential_time": execution_times.get("verify_sequential_time", 0),
                    "verify_average_time": execution_times.get("verify_average_time", 0),
                    "verify_parallel_estimate": execution_times.get("verify_parallel_estimate", 0)
                }
                
                exact_results.append(exact_result)
                
                # Guardar resultados parciales después de cada 5 preguntas
                if len(exact_results) % 5 == 0:
                    with open(exact_dir / "partial_results.json", "w", encoding="utf-8") as f:
                        json.dump(exact_results, f, indent=2, ensure_ascii=False)
            
            # Pequeña pausa para evitar sobrecarga
            await asyncio.sleep(1)
    
    # Guardar todos los resultados
    with open(ideal_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(ideal_results, f, indent=2, ensure_ascii=False)
    
    with open(exact_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(exact_results, f, indent=2, ensure_ascii=False)
    
    # Calcular estadísticas para cada tipo de respuesta
    await generate_stats_report(ideal_results, ideal_dir, "Modified RAG - Ideal Answer Results")
    await generate_stats_report(exact_results, exact_dir, "Modified RAG - Exact Answer Results")
    
    logger.info(f"Benchmark completado. Resultados guardados en {results_dir}")

async def generate_stats_report(results: List[Dict[str, Any]], output_dir: Path, report_title: str):
    """
    Genera un informe de estadísticas a partir de los resultados
    
    Args:
        results: Lista de resultados de evaluación
        output_dir: Directorio donde guardar el informe
        report_title: Título del informe
    """
    if not results:
        logger.warning(f"No hay resultados para generar el informe: {report_title}")
        return
    
    # Calcular estadísticas básicas
    total_questions = len(results)
    correct_answers = sum(result["is_correct"] for result in results)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    avg_cosine = np.mean([result["cosine_similarity"] for result in results])
    avg_precision = np.mean([result["bert_score_precision"] for result in results])
    avg_recall = np.mean([result["bert_score_recall"] for result in results])
    avg_f1 = np.mean([result["bert_score_f1"] for result in results])
    
    # Tiempos promedio
    avg_total_time = np.mean([result["total_time"] for result in results])
    avg_query_embedding_time = np.mean([result.get("query_embedding_time", 0) for result in results if result.get("query_embedding_time", 0) > 0])
    avg_inbedder_time = np.mean([result["inbedder_time"] for result in results if result["inbedder_time"] > 0])
    avg_retrieval_time = np.mean([result["retrieval_time"] for result in results if result["retrieval_time"] > 0])
    avg_similar_question_time = np.mean([result["similar_question_time"] for result in results if "similar_question_time" in result and result["similar_question_time"] > 0])
    avg_clustering_time = np.mean([result.get("clustering_time", 0) for result in results if result.get("clustering_time", 0) > 0])
    
    # Tiempos de drafting y verification
    avg_draft_sequential = np.mean([result.get("draft_sequential_time", 0) for result in results if result.get("draft_sequential_time", 0) > 0])
    avg_draft_parallel = np.mean([result.get("draft_parallel_estimate", 0) for result in results if result.get("draft_parallel_estimate", 0) > 0])
    avg_verify_sequential = np.mean([result.get("verify_sequential_time", 0) for result in results if result.get("verify_sequential_time", 0) > 0])
    avg_verify_parallel = np.mean([result.get("verify_parallel_estimate", 0) for result in results if result.get("verify_parallel_estimate", 0) > 0])
    
    # Calcular tiempo secuencial vs paralelo
    seq_llm_time = avg_draft_sequential + avg_verify_sequential
    par_llm_time = avg_draft_parallel + avg_verify_parallel if (avg_draft_parallel > 0 and avg_verify_parallel > 0) else 0
    speedup = seq_llm_time / par_llm_time if par_llm_time > 0 else 0
    
    # Crear informe
    report = {
        "title": report_title,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "similarity_metrics": {
            "avg_cosine_similarity": avg_cosine,
            "avg_bert_precision": avg_precision,
            "avg_bert_recall": avg_recall,
            "avg_bert_f1": avg_f1
        },
        "timing": {
            "avg_total_time": avg_total_time,
            "avg_query_embedding_time": avg_query_embedding_time,
            "avg_inbedder_time": avg_inbedder_time,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_similar_question_time": avg_similar_question_time,
            "avg_clustering_time": avg_clustering_time,
            "sequential_vs_parallel": {
                "avg_draft_sequential": avg_draft_sequential,
                "avg_draft_parallel": avg_draft_parallel,
                "avg_verify_sequential": avg_verify_sequential,
                "avg_verify_parallel": avg_verify_parallel,
                "total_sequential_llm": seq_llm_time,
                "total_parallel_llm": par_llm_time,
                "speedup": speedup
            }
        }
    }
    
    # Guardar informe
    with open(output_dir / "stats_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # También crear una versión legible en formato TXT
    with open(output_dir / "stats_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{report_title}\n")
        f.write("=" * len(report_title) + "\n\n")
        
        f.write(f"Total de preguntas evaluadas: {total_questions}\n")
        f.write(f"Respuestas correctas: {correct_answers}\n")
        f.write(f"Precisión (accuracy): {accuracy:.2%}\n\n")
        
        f.write("Métricas de similitud:\n")
        f.write(f"- Similitud coseno promedio: {avg_cosine:.4f}\n")
        f.write(f"- BERTScore precisión promedio: {avg_precision:.4f}\n")
        f.write(f"- BERTScore recall promedio: {avg_recall:.4f}\n")
        f.write(f"- BERTScore F1 promedio: {avg_f1:.4f}\n\n")
        
        f.write("Tiempos de ejecución:\n")
        f.write(f"- Tiempo total promedio: {avg_total_time:.2f} segundos\n")
        f.write(f"- Tiempo de consulta de embedding: {avg_query_embedding_time:.2f} segundos\n")
        f.write(f"- Tiempo de InBedder promedio: {avg_inbedder_time:.2f} segundos\n")
        f.write(f"- Tiempo de recuperación promedio: {avg_retrieval_time:.2f} segundos\n")
        f.write(f"- Tiempo de búsqueda de pregunta similar: {avg_similar_question_time:.2f} segundos\n")
        f.write(f"- Tiempo de clustering: {avg_clustering_time:.2f} segundos\n")
        
        f.write("\nTiempos de drafting y verificación:\n")
        f.write(f"- Tiempo secuencial de LLM: {avg_draft_sequential:.2f} segundos\n")
        f.write(f"- Tiempo paralelo de LLM: {avg_draft_parallel:.2f} segundos\n")
        f.write(f"- Tiempo secuencial de verificación: {avg_verify_sequential:.2f} segundos\n")
        f.write(f"- Tiempo paralelo de verificación: {avg_verify_parallel:.2f} segundos\n")
        
        f.write("\nComparación secuencial vs paralelo:\n")
        f.write(f"- Tiempo secuencial total de LLM: {seq_llm_time:.2f} segundos\n")
        f.write(f"- Tiempo paralelo total de LLM: {par_llm_time:.2f} segundos\n")
        f.write(f"- Speedup: {speedup:.2f}\n")
    
    logger.info(f"Informe de estadísticas generado: {output_dir / 'stats_report.txt'}")

if __name__ == "__main__":
    # Crear directorio para logs si no existe
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Usar valor de límite desde línea de comandos si se proporciona
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark para Modified RAG con preguntas de Qdrant")
    parser.add_argument("--limit", type=int, default=400, help="Número máximo de preguntas a evaluar")
    args = parser.parse_args()
    
    logger.info(f"Iniciando benchmark con límite de {args.limit} preguntas")
    asyncio.run(run_benchmark_with_qdrant(limit=args.limit))