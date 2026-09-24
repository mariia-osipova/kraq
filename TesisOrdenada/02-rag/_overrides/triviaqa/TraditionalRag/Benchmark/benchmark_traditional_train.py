import os
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
from TraditionalRag.traditional_rag import traditional_rag

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
    client: AsyncOpenAI = None
) -> int:
    """
    Evalúa si la respuesta generada es correcta para preguntas de tipo trivia simple
    usando exclusivamente el LLM para el juicio
    """
    try:
        # Prompt para evaluación
        prompt = f"""You are an expert evaluator for question answering systems. Your task is to determine if the generated answer correctly responds to the question according to the reference answer.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

The reference answer represents the truth. The generated answer must match the meaning of the reference answer to be considered correct. If the generated answer is more specific but the core meaning is the same, it is also considered correct.

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
    embeddings_client: AsyncOpenAI,
    top_k: int
) -> Tuple[str, float, float]:
    """
    Genera una respuesta con Traditional RAG y captura los tiempos
    
    Args:
        question: La pregunta a responder
        llm_client: Cliente para el LLM
        embeddings_client: Cliente para los embeddings
        top_k: Número de documentos a recuperar
        
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
            "top_k": top_k,
            "debug_prints": False,
            "verbose_timing": False
        }
        
        # Reemplazar temporalmente sys.stdout para capturar los tiempos
        original_stdout = sys.stdout
        sys.stdout = time_logger
        
        # Ejecutar Traditional RAG y obtener respuesta y tiempo total
        response, total_time = await traditional_rag(**params)
        
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

async def get_questions_from_qdrant(limit: int = 100) -> List[Dict[str, Any]]:
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
        
        # Procesar las preguntas - ahora solo buscamos "answer"
        questions = []
        for point in points:
            question_data = {
                "question_id": point.payload.get("question_id", ""),
                "question": point.payload.get("question", "")
            }
            
            # Agregar respuesta si existe
            if "answer" in point.payload and point.payload["answer"]:
                question_data["answer"] = point.payload["answer"]
            
            # Solo agregamos preguntas que tengan respuesta
            if "answer" in question_data:
                questions.append(question_data)
        
        await qdrant_client.close()
        
        logger.info(f"Se obtuvieron {len(questions)} preguntas válidas de Qdrant")
        return questions
        
    except Exception as e:
        logger.error(f"Error obteniendo preguntas de Qdrant: {e}")
        return []

async def run_benchmark(questions=None, results_dir=None, top_k=20, n_questions=None):
    """
    Ejecuta el benchmark completo
    
    Args:
        questions: Lista opcional de preguntas predefinidas
        results_dir: Directorio opcional para guardar resultados
        top_k: Número de documentos a recuperar
        n_questions: Número específico de preguntas a procesar
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
    
    # Usar el directorio de resultados proporcionado o el predeterminado
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    else:
        results_dir = Path(results_dir)
    
    results_dir.mkdir(exist_ok=True)
    
    # Crear directorio para respuestas
    answer_dir = results_dir / "answers"
    answer_dir.mkdir(exist_ok=True)
    
    # Verificar si existe checkpoint previo
    answer_checkpoint = answer_dir / "partial_results.json"
    
    # Cargar resultados previos si existen
    answer_results = []
    processed_question_ids = set()
    
    if answer_checkpoint.exists():
        try:
            with open(answer_checkpoint, "r", encoding="utf-8") as f:
                answer_results = json.load(f)
                logger.info(f"Cargados {len(answer_results)} resultados previos")
                processed_question_ids.update([r["question_id"] for r in answer_results])
        except Exception as e:
            logger.error(f"Error cargando checkpoint: {e}")
            answer_results = []
    
    # Si se proporcionaron preguntas y n_questions, usar solo las primeras n_questions
    if questions and n_questions:
        questions = questions[:n_questions]
    elif questions:
        n_questions = len(questions)
    else:
        # Si no se proporcionaron preguntas, obtener n_questions de Qdrant
        questions = await get_questions_from_qdrant(limit=n_questions if n_questions else 400)
        if n_questions:
            questions = questions[:n_questions]
    
    total_questions = len(questions)
    remaining_questions = len(questions)
    
    logger.info(f"Total de preguntas: {total_questions}")
    logger.info(f"Preguntas ya procesadas: {total_questions - remaining_questions}")
    logger.info(f"Preguntas restantes por procesar: {remaining_questions}")
    
    # Si ya se procesaron todas las preguntas, mostrar mensaje y salir
    if remaining_questions == 0:
        logger.info("Todas las preguntas ya fueron procesadas. Generando informes finales.")
        # Guardar todos los resultados
        with open(answer_dir / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(answer_results, f, indent=2, ensure_ascii=False)
        
        # Calcular estadísticas
        await generate_stats_report(answer_results, answer_dir, "Answer Results")
        
        logger.info(f"Benchmark completado. Resultados guardados en {results_dir}")
        return
    
    # Procesar cada pregunta restante
    for i, question_data in enumerate(tqdm(questions, desc="Evaluando preguntas")):
        question = question_data["question"]
        question_id = question_data.get("question_id", "unknown")
    
        logger.info(f"Procesando pregunta {i+1}/{len(questions)} (ID: {question_id}): {question[:50]}...")
        
        # Ejecutar Traditional RAG una sola vez para obtener respuesta
        has_response = False
        response = ""
        total_time = 0
        retrieval_time = 0
    
        # Solo ejecutar el algoritmo si tiene respuesta de referencia
        if "answer" in question_data:
            try:
                # Generar respuesta
                response, total_time, retrieval_time = await generate_traditional_rag_response(
                    question, llm_client, embeddings_client, top_k
                )
                has_response = True
                
            except Exception as e:
                logger.error(f"Error ejecutando Traditional RAG: {e}")
                response = "Error generando respuesta"
        
            # Evaluar la respuesta contra la respuesta de referencia
            if has_response:
                eval_start_time = time.time()
                reference = question_data["answer"]
        
                # Calcular métricas
                cosine_sim = calculate_cosine_similarity(response, reference)
                precision, recall, f1 = calculate_bert_score(response, reference)
        
                # Evaluar corrección usando modelo local
                is_correct = await evaluate_correctness(
                    question=question,
                    generated_answer=response,
                    reference_answer=reference,
                    client=llm_client
                )
                
                eval_time = time.time() - eval_start_time
                logger.debug(f"Tiempo de evaluación: {eval_time:.2f}s")
                
                # Crear diccionario con resultados
                result = {
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": reference,
                    "generated_answer": response,
                    "cosine_similarity": cosine_sim,
                    "bert_score_precision": precision,
                    "bert_score_recall": recall,
                    "bert_score_f1": f1,
                    "is_correct": is_correct,
                    "total_time": total_time,
                    "retrieval_time": retrieval_time
                }
                
                answer_results.append(result)
                
                # Guardar resultados parciales cada 5 preguntas
                if len(answer_results) % 5 == 0:
                    with open(answer_dir / "partial_results.json", "w", encoding="utf-8") as f:
                        json.dump(answer_results, f, indent=2, ensure_ascii=False)
    
    # Guardar todos los resultados al finalizar
    with open(answer_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(answer_results, f, indent=2, ensure_ascii=False)
    
    # Generar informe de estadísticas
    await generate_stats_report(answer_results, answer_dir, "Answer Results")
    
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