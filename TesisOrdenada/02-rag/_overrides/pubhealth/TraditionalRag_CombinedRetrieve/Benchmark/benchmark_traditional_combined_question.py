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
from TraditionalRag_CombinedRetrieve.traditional_rag_combined_retrieve import traditional_rag_combined_retrieve

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
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1')



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

The reference answer represents the truth. The label of the generated answer must match the reference answer to be considered correct.

Respond with ONLY a single digit:
1 - CORRECT: 
0 - INCORRECT: 

Your verdict (just the digit 1 or 0):"""
        
        # Usar el modelo para la evaluación
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        
        answer_text = response.choices[0].message.content.strip()
        return 1 if answer_text.startswith("1") else 0
            
    except Exception as e:
        logger.error(f"Error evaluando corrección: {e}")
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
            "top_k": 10,
            "debug_prints": False,
            "verbose_timing": False,
            "question_collection": "questions-index-finetuned"
        }
        
        # Reemplazar temporalmente sys.stdout para capturar los tiempos
        original_stdout = sys.stdout
        sys.stdout = time_logger
        
        # Ejecutar Traditional RAG y obtener respuesta y tiempo total
        response, total_time = await traditional_rag_combined_retrieve(**params)
        
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
    Ejecuta el benchmark completo
    """
    # Configurar clientes de API
    llm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"
    )

    embeddings_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    # Crear directorios para resultados
    results_dir = Path(__file__).parent / "results_random"
    results_dir.mkdir(exist_ok=True)
    
    ideal_dir = results_dir / "ideal_answers"
    exact_dir = results_dir / "exact_answers"
    ideal_dir.mkdir(exist_ok=True)
    exact_dir.mkdir(exist_ok=True)
    
    # Obtener preguntas de Qdrant
    all_questions = await get_questions_from_qdrant(limit=1000)
    
    if not all_questions:
        logger.error("No se pudieron obtener preguntas de Qdrant")
        return
    
    # Verificar checkpoints existentes
    ideal_results = []
    exact_results = []
    processed_questions = set()
    
    if (ideal_dir / "partial_results.json").exists():
        try:
            with open(ideal_dir / "partial_results.json", "r", encoding="utf-8") as f:
                ideal_results = json.load(f)
                processed_questions.update([r["question_id"] for r in ideal_results])
                logger.info(f"Cargados {len(ideal_results)} resultados previos para respuestas ideales")
        except Exception as e:
            logger.error(f"Error cargando resultados previos para respuestas ideales: {e}")
    
    if (exact_dir / "partial_results.json").exists():
        try:
            with open(exact_dir / "partial_results.json", "r", encoding="utf-8") as f:
                exact_results = json.load(f)
                processed_questions.update([r["question_id"] for r in exact_results])
                logger.info(f"Cargados {len(exact_results)} resultados previos para respuestas exactas")
        except Exception as e:
            logger.error(f"Error cargando resultados previos para respuestas exactas: {e}")
    
    # Filtrar preguntas ya procesadas
    questions_to_process = [q for q in all_questions if q["question_id"] not in processed_questions]
    
    total_questions = len(all_questions)
    remaining_questions = len(questions_to_process)
    
    # Analizar distribución de tipos de preguntas
    question_types = {}
    for q in all_questions:
        q_type = "unknown"
        if "question_id" in q:
            # Convert question_id to string before checking
            question_id_str = str(q["question_id"])
            if "_" in question_id_str:
                # Intentar inferir tipo a partir del ID si está disponible (e.g., "list_123")
                id_parts = question_id_str.split("_")
                if len(id_parts) > 1:
                    q_type = id_parts[0]
        question_types[q_type] = question_types.get(q_type, 0) + 1
    
    logger.info(f"Total de preguntas: {total_questions}")
    logger.info(f"Distribución de tipos (inferidos del ID): {question_types}")
    logger.info(f"Preguntas ya procesadas: {total_questions - remaining_questions}")
    logger.info(f"Preguntas por procesar: {remaining_questions}")
    
    # Procesar cada pregunta
    for i, question_data in enumerate(tqdm(questions_to_process, desc="Procesando preguntas")):
        question = question_data["question"]
        question_id = question_data.get("question_id", f"unknown_{i}")
        
        logger.info(f"Procesando pregunta {i+1}/{len(questions_to_process)}: {question}")
        
        # Generar respuesta con RAG tradicional
        response, total_time, retrieval_time = await generate_traditional_rag_response(
            question, llm_client, embeddings_client
        )
        
        # Evaluar respuesta con respuesta ideal
        if "ideal_answer" in question_data:
            ideal_result = await evaluate_question(
                question_data, llm_client, response, total_time, retrieval_time, "ideal_answer"
            )
            ideal_results.append(ideal_result)
            
            # Guardar resultados parciales después de cada 5 preguntas
            if len(ideal_results) % 5 == 0:
                with open(ideal_dir / "partial_results.json", "w", encoding="utf-8") as f:
                    json.dump(ideal_results, f, indent=2, ensure_ascii=False)
        
        # Evaluar respuesta con respuesta exacta
        if "exact_answer" in question_data:
            exact_result = await evaluate_question(
                question_data, llm_client, response, total_time, retrieval_time, "exact_answer"
            )
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
                    question_data["exact_answer"] = [answer]
                
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