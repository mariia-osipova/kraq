import os
import csv
import json
import time
import sys
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from qdrant_client import AsyncQdrantClient
import torch

# Añadir el directorio raíz del proyecto a sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importar los módulos necesarios
from TraditionalRag_SimilarRetrieve.traditional_rag_similar_question import traditional_rag_similar_question

# Configurar logger específico para train
logger.add("benchmark_traditional_train_logs.log", rotation="500 MB")

# Cargar variables de entorno
load_dotenv()

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Modelo para el cálculo de similitud coseno
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')



def calculate_cosine_similarity(text1: str, text2: Any) -> float:
    """
    Calcula la similitud coseno entre dos textos
    """
    # Convertir a string si es una lista (respuestas exactas pueden ser listas)
    if isinstance(text1, list):
        text1 = str(text1)
    if isinstance(text2, list):
        text2 = str(text2)
    
    if not text1 or not text2:
        logger.warning("Uno de los textos está vacío o es None. Retornando similitud 0.0")
        return 0.0
    
    try:
        embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
        
        # Asegúrate de que los tensores sean unidimensionales (vectores)
        if len(embedding1.shape) > 1 and embedding1.shape[0] > 1:
            embedding1 = torch.mean(embedding1, dim=0)
        if len(embedding2.shape) > 1 and embedding2.shape[0] > 1:
            embedding2 = torch.mean(embedding2, dim=0)
            
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        # Asegúrate de que el resultado es un escalar
        if isinstance(similarity, torch.Tensor) and similarity.numel() > 1:
            similarity = similarity.mean()
            
        return similarity.item()
    except Exception as e:
        logger.error(f"Error calculando similitud coseno: {e}")
        return 0.0

def calculate_bert_score(candidate: str, reference: Any) -> Tuple[float, float, float]:
    """
    Calcula BERTScore entre respuesta candidata y referencia
    """
    # Convertir a string si es una lista
    if isinstance(candidate, list):
        candidate = str(candidate)
    if isinstance(reference, list):
        # Si la referencia es una lista, tomar el elemento más largo como representativo
        try:
            reference = str(max(reference, key=lambda x: len(str(x))))
        except:
            reference = str(reference)
    
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
    answer_type: str = "ideal_answer",
    client: AsyncOpenAI = None
) -> int:
    """
    Evalúa si la respuesta generada es correcta según criterios más estrictos
    """
    try:
        # Diferentes prompts según el tipo de respuesta
        if answer_type == "exact_answer":
            # Para respuestas exactas (con lista de verdades)
            if isinstance(reference_answer, list) and len(reference_answer) > 0:
                # Convertir cada elemento a string y unirlos
                reference_str = ", ".join([str(ans) for ans in reference_answer])
                prompt = f"""
You are an expert evaluator of question answering systems with a focus on factual accuracy. Assess if the generated answer correctly addresses the question based on the reference answers.

Question: {question}
Generated Answer: {generated_answer}
Reference Answers: {reference_str}

The reference answers are provided as a list of true answers to the question.

The generated answer should cover most items in the reference answers

Contradicting ANY of the reference answers is a critical error
Including incorrect information invalidates the answer

Is the generated answer correct?
Answer with a single digit:
1 - Yes, the generated answer meets the criteria for correctness
0 - No, the generated answer fails to meet one or more key criteria

Your response (just the digit 1 or 0):
"""
            else:
                # Una sola respuesta exacta
                prompt = f"""
You are an expert evaluator of question answering systems with a focus on factual accuracy. Assess if the generated answer correctly addresses the question based on the reference answer.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

The reference answer represents the ground truth to the question.

The generated answer MUST contain the essential information from the reference answer

It must not contain any statements that contradict the reference answer

While the wording may differ, the core facts and meaning must be preserved. Partial answers that miss key points should be considered incorrect.

Is the generated answer correct?
Answer with a single digit:
1 - Yes, the generated answer meets the criteria for correctness
0 - No, the generated answer fails to meet one or more key criteria

Your response (just the digit 1 or 0):
"""
        else:
            # Para respuestas ideales (considerando referencia como verdad)
            prompt = f"""
You are an expert evaluator of question answering systems with a focus on completeness and accuracy. Assess if the generated answer correctly addresses the question based on the reference answer.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

The reference answer represents the ground truth to the question.

The generated answer MUST contain the essential information from the reference answer

It must not contain any statements that contradict the reference answer

While the wording may differ, the core facts and meaning must be preserved. Partial answers that miss key points should be considered incorrect.

Is the generated answer correct?
Answer with a single digit:
1 - Yes, the generated answer meets the criteria for correctness
0 - No, the generated answer fails to meet one or more key criteria

Your response (just the digit 1 or 0):
"""

        # Usar el modelo para la evaluación
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        answer_text = response.choices[0].message.content.strip()
        return 1 if answer_text.startswith("1") else 0
            
    except Exception as e:
        logger.error(f"Error evaluando corrección con modelo local: {e}")
        return 0  # Por defecto, si hay error, consideramos la respuesta incorrecta


# Vamos a crear una nueva función para generar la respuesta una sola vez
async def generate_traditional_rag_response(
    question: str,
    llm_client: AsyncOpenAI,
    embeddings_client: AsyncOpenAI
) -> Tuple[str, float, float]:
    """
    Genera una respuesta con Traditional RAG y captura los tiempos
    
    Args:
        question: La pregunta a responder
        llm_client: Cliente para el LLM
        embeddings_client: Cliente para los embeddings
        
    Returns:
        Tuple con (respuesta, tiempo_total, tiempo_recuperacion)
    """
    # Inicializar variables
    start_time = time.time()
    retrieval_time = 0
    
    try:
        # Clase personalizada para capturar los tiempos de ejecución
        class TimeCaptureLogger:
            def __init__(self):
                self.times = {}
            
            def write(self, text):
                if "retrieval_time:" in text:
                    self.times["retrieval_time"] = float(text.split(":")[1].strip())
                elif "⏱️ Document retrieval time:" in text:
                    self.times["retrieval_time"] = float(text.split(":")[1].strip().split()[0])
                return
        
        # Capturar la salida para extraer los tiempos
        time_logger = TimeCaptureLogger()
        
        # Configurar parámetros para Traditional RAG
        params = {
            "query": question,
            "embeddings_client": embeddings_client,  # Cliente Ollama para embeddings
            "llm_client": llm_client,                # Cliente vLLM para LLM
            "embedding_model": EMBEDDING_MODEL,      # nomic-embed-text (Ollama)
            "llm_model": LLM_MODEL,                  # Llama 3.1 Instruct (vLLM)
            "top_k": 20,
            "debug_prints": False,
            "verbose_timing": False,
            "question_collection": "questions-index-finetuned"
        }
        
        # Reemplazar temporalmente sys.stdout para capturar los tiempos
        original_stdout = sys.stdout
        sys.stdout = time_logger
        
        # Ejecutar Traditional RAG y obtener respuesta y tiempo total
        response, total_time = await traditional_rag_similar_question(**params)
        
        # Restaurar sys.stdout
        sys.stdout = original_stdout
        
        # Obtener los tiempos capturados
        retrieval_time = time_logger.times.get("retrieval_time", 0)
        
    except Exception as e:
        logger.error(f"Error ejecutando Traditional RAG: {e}")
        response = "Error generando respuesta"
        total_time = 0
        sys.stdout = original_stdout
    
    # Si total_time es 0, algo salió mal, usar el tiempo desde el inicio
    if total_time == 0:
        total_time = time.time() - start_time
        
    return response, total_time, retrieval_time

# Ahora modifiquemos la función evaluate_question para que acepte la respuesta ya generada
async def evaluate_question(
    question_data: Dict[str, str],
    llm_client: AsyncOpenAI,
    response: str,
    total_time: float,
    retrieval_time: float,
    answer_type: str = "ideal_answer"
) -> Dict[str, Any]:
    """
    Evalúa una respuesta ya generada contra un tipo de referencia específico
    """
    question = question_data["question"]
    reference_answer = question_data.get(answer_type, "")
    question_id = question_data.get("question_id", "unknown")
    
    # Si no hay respuesta de referencia, devolver None
    if not reference_answer:
        return None
    
    # Calcular métricas
    try:
        if isinstance(reference_answer, list):
            # Para respuestas exactas, usamos la mejor puntuación individual para similitud coseno
            best_cosine_sim = max([calculate_cosine_similarity(response, str(ref)) for ref in reference_answer]) if reference_answer else 0.0
            # Para BERTScore, tomamos la referencia más larga
            representative_ref = str(max(reference_answer, key=lambda x: len(str(x)))) if reference_answer else ""
            precision, recall, f1 = calculate_bert_score(response, representative_ref)
        else:
            # Para respuestas ideales o exactas singulares
            cosine_sim = calculate_cosine_similarity(response, reference_answer)
            best_cosine_sim = cosine_sim
            precision, recall, f1 = calculate_bert_score(response, reference_answer)
    except Exception as e:
        logger.error(f"Error calculando métricas: {e}")
        best_cosine_sim = 0.0
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    # Evaluar corrección usando modelo local (vLLM)
    is_correct = await evaluate_correctness(
        question=question,
        generated_answer=response,
        reference_answer=reference_answer,
        answer_type=answer_type,
        client=llm_client
    )
    
    # Crear diccionario con resultados
    result = {
        "question_id": question_id,
        "question": question,
        "reference_answer": reference_answer,
        "answer_type": answer_type,
        "generated_answer": response,
        "cosine_similarity": best_cosine_sim,
        "bert_score_precision": precision,
        "bert_score_recall": recall,
        "bert_score_f1": f1,
        "is_correct": is_correct,
        "total_time": total_time,
        "retrieval_time": retrieval_time
    }
    
    return result

async def run_benchmark():
    """
    Ejecuta el benchmark con soporte para respuestas ideales y exactas
    """
    logger.info("Iniciando benchmark de Traditional RAG con datos de questions-benchmark")
    
    # Crear directorio para resultados dentro de la carpeta Benchmark
    # Obtener la ruta absoluta del archivo actual (benchmark_traditional_train.py)
    current_file_path = Path(__file__).resolve()
    
    # El directorio padre del archivo es la carpeta Benchmark
    benchmark_dir = current_file_path.parent
    
    # Crear la carpeta de resultados dentro de la carpeta Benchmark
    results_dir = benchmark_dir / "benchmark_results_traditional_qdrant"
    results_dir.mkdir(exist_ok=True)
    
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
    
    # Obtener preguntas de Qdrant usando función similar a SpeculativeRag
    all_questions = await get_questions_from_qdrant(limit=500)
    
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
    
    # Inicializar clientes para LLM (vLLM) y embeddings (Ollama)
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    ollama_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    # Procesar cada pregunta restante
    for i, question_data in enumerate(tqdm(questions, desc="Evaluando preguntas")):
        question = question_data["question"]
        question_id = question_data.get("question_id", "unknown")
        
        logger.info(f"Procesando pregunta {i+1}/{len(questions)} (ID: {question_id}): {question[:50]}...")
        
        # Generar la respuesta una sola vez con Traditional RAG
        response, total_time, retrieval_time = await generate_traditional_rag_response(
            question=question,
            llm_client=vllm_client,
            embeddings_client=ollama_client
        )
        
        # Evaluar contra la respuesta ideal si está disponible
        if "ideal_answer" in question_data:
            ideal_result = await evaluate_question(
                question_data=question_data,
                llm_client=vllm_client,
                response=response,
                total_time=total_time,
                retrieval_time=retrieval_time,
                answer_type="ideal_answer"
            )
            
            if ideal_result:
                ideal_results.append(ideal_result)
                
                # Guardar resultados parciales después de cada 5 preguntas
                if len(ideal_results) % 5 == 0:
                    with open(ideal_dir / "partial_results.json", "w", encoding="utf-8") as f:
                        json.dump(ideal_results, f, indent=2, ensure_ascii=False)
        
        # Evaluar contra las respuestas exactas si están disponibles (usando la misma respuesta generada)
        if "exact_answer" in question_data:
            exact_result = await evaluate_question(
                question_data=question_data,
                llm_client=vllm_client,
                response=response,
                total_time=total_time,
                retrieval_time=retrieval_time,
                answer_type="exact_answer"
            )
            
            if exact_result:
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
    await generate_stats_report(ideal_results, ideal_dir, "Ideal Answers")
    await generate_stats_report(exact_results, exact_dir, "Exact Answers")
    
    logger.info(f"Benchmark completado. Resultados guardados en {results_dir}")

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
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_response[0] if isinstance(scroll_response, tuple) else scroll_response.points
        
        # Filtrar a solo las preguntas que tienen respuestas ideales y/o exactas
        questions = []
        for point in points:
            question_data = {
                "question_id": point.payload.get("question_id", ""),
                "question": point.payload.get("question", "")
            }
            
            # Agregar respuesta ideal si existe
            if "ideal_answer" in point.payload and point.payload["ideal_answer"]:
                question_data["ideal_answer"] = point.payload["ideal_answer"]
            
            # Agregar respuesta exacta si existe
            if "exact_answer" in point.payload and point.payload["exact_answer"]:
                question_data["exact_answer"] = point.payload["exact_answer"]
            
            # Solo agregamos preguntas que tengan al menos un tipo de respuesta
            if "ideal_answer" in question_data or "exact_answer" in question_data:
                questions.append(question_data)
        
        await qdrant_client.close()
        
        logger.info(f"Se obtuvieron {len(questions)} preguntas válidas de Qdrant")
        return questions
        
    except Exception as e:
        logger.error(f"Error obteniendo preguntas de Qdrant: {e}")
        return []

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
    avg_retrieval_time = np.mean([result["retrieval_time"] for result in results if result["retrieval_time"] > 0])
    
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
            "avg_retrieval_time": avg_retrieval_time
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
        f.write(f"- Tiempo de recuperación promedio: {avg_retrieval_time:.2f} segundos\n")
    
    logger.info(f"Informe de estadísticas generado: {output_dir / 'stats_report.txt'}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())