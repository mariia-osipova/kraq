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
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(text1: str, text2: Any) -> float:
    """
    Calcula la similitud coseno entre dos textos
    """
    # Convertir a string si es una lista
    if isinstance(text1, list):
        text1 = str(text1)
    if isinstance(text2, list):
        text2 = str(text2)
    
    if not text1 or not text2:
        logger.warning("Uno de los textos está vacío o es None. Retornando similitud 0.0")
        return 0.0
    
    try:
        # Especificar el dispositivo CUDA:1
        device = torch.device("cuda:1")
        
        # Asegurarnos de que el modelo esté en CUDA:1
        if not hasattr(calculate_cosine_similarity, 'sentence_model'):
            calculate_cosine_similarity.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:1")
        
        # Generar embeddings y moverlos a CUDA:1
        embedding1 = calculate_cosine_similarity.sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = calculate_cosine_similarity.sentence_model.encode(text2, convert_to_tensor=True)
        
        embedding1 = embedding1.to(device)
        embedding2 = embedding2.to(device)
        
        # Asegúrate de que los tensores sean unidimensionales (vectores)
        if len(embedding1.shape) > 1 and embedding1.shape[0] > 1:
            embedding1 = torch.mean(embedding1, dim=0)
        if len(embedding2.shape) > 1 and embedding2.shape[0] > 1:
            embedding2 = torch.mean(embedding2, dim=0)
            
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        # Asegúrate de que el resultado es un escalar
        if isinstance(similarity, torch.Tensor) and similarity.numel() > 1:
            similarity = similarity.mean()
            
        # Mover el resultado a CPU antes de convertirlo a Python
        return similarity.cpu().item()
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

The reference answer represents the truth. The generated answer must match the meaning of the reference answer to be considered correct or if its a factoid answer it must be the same as the reference answer. If the generated answer is more specific but the core meaning is the same, it is also considered correct.

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

async def get_questions_from_qdrant(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Obtiene preguntas directamente desde la colección de Qdrant
    
    Args:
        limit: Número máximo de preguntas a recuperar
        
    Returns:
        Lista de preguntas con sus respuestas
    """
    try:
        logger.info(f"Obteniendo hasta {limit} preguntas de Qdrant de la colección {QUESTIONS_COLLECTION}")
        
        # Crear cliente Qdrant
        qdrant_path = Path(__file__).parent.parent.parent / "qdrant_client"
        qdrant_client = AsyncQdrantClient(path=qdrant_path)
        
        # Recuperar puntos de la colección de preguntas
        points, next_page_offset = await qdrant_client.scroll(
            collection_name=QUESTIONS_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
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

async def run_benchmark_with_qdrant(
    limit: int = 300,
    questions: Optional[List[Dict[str, Any]]] = None,
    results_dir: Optional[Path] = None,
    k: int = 5,
    m: int = 10,
    top_k: int = 20
) -> None:
    """
    Ejecuta el benchmark usando las preguntas almacenadas en Qdrant
    
    Args:
        limit: Número máximo de preguntas a evaluar
        questions: Lista opcional de preguntas predefinidas
        results_dir: Directorio opcional para resultados
        k: Número de clusters
        m: Número de subsets de documentos
        top_k: Número de documentos a recuperar
    """
    logger.info(f"Iniciando benchmark de Modified RAG con {limit} preguntas")
    
    # Inicializar clientes
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"
    )
    
    ollama_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    drafter_client = vllm_client
    verifier_client = vllm_client
    
    # Usar directorio de resultados personalizado si se proporciona
    if results_dir is None:
        results_dir = Path(__file__).parent / "benchmark_results_modified_qdrant"
    results_dir.mkdir(exist_ok=True)
    
    # Crear directorio para respuestas
    answer_dir = results_dir / "answers"
    answer_dir.mkdir(exist_ok=True)
    
    # Usar preguntas proporcionadas o obtenerlas de Qdrant
    if questions is not None:
        all_questions = questions[:limit]
    else:
        all_questions = await get_questions_from_qdrant(limit=limit)
    
    # Modificar los parámetros para Modified RAG
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
        "collection_name": RETRIEVAL_COLLECTION,
        "questions_collection": SIMILAR_QUESTIONS_COLLECTION,
        "ollama_client": ollama_client,
        "llm_client": vllm_client,
        "drafter_client": drafter_client,
        "verifier_client": verifier_client
    }
    
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
    
    # Filtrar preguntas que ya se procesaron
    questions = [q for q in all_questions if q["question_id"] not in processed_question_ids]
    
    total_questions = len(all_questions)
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
        
        # Ejecutar Modified RAG una sola vez para obtener respuesta
        has_response = False
        response = ""
        execution_times = {}
        total_time = 0
    
        # Solo ejecutar el algoritmo si tiene respuesta de referencia
        if "answer" in question_data:
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
                        elif "Similar question search time:" in text:
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
                
                # Reemplazar temporalmente sys.stdout para capturar los tiempos
                original_stdout = sys.stdout
                sys.stdout = time_logger
                
                # Ejecutar Modified RAG
                response, total_time = await modified_rag(query=question, **params)
                
                has_response = True
                
                # Restaurar sys.stdout
                sys.stdout = original_stdout
                
                # Obtener los tiempos capturados internos del algoritmo
                execution_times = time_logger.times
                
            except Exception as e:
                logger.error(f"Error ejecutando Modified RAG: {e}")
                response = "Error generando respuesta"
                if 'original_stdout' in locals():
                    sys.stdout = original_stdout
        
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
                    client=verifier_client
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
                    "query_embedding_time": execution_times.get("query_embedding_time", 0),
                    "inbedder_time": execution_times.get("inbedder_time", 0),
                    "retrieval_time": execution_times.get("retrieval_time", 0),
                    "similar_question_time": execution_times.get("similar_question_time", 0),
                    "clustering_time": execution_times.get("clustering_time", 0),
                    "draft_sequential_time": execution_times.get("draft_sequential_time", 0),
                    "draft_parallel_estimate": execution_times.get("draft_parallel_estimate", 0),
                    "verify_sequential_time": execution_times.get("verify_sequential_time", 0),
                    "verify_parallel_estimate": execution_times.get("verify_parallel_estimate", 0)
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
        f.write(f"- Tiempo secuencial de drafting: {avg_draft_sequential:.2f} segundos\n")
        f.write(f"- Tiempo paralelo de drafting: {avg_draft_parallel:.2f} segundos\n")
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
    parser.add_argument("--limit", type=int, default=300, help="Número máximo de preguntas a evaluar")
    args = parser.parse_args()
    
    logger.info(f"Iniciando benchmark con límite de {args.limit} preguntas")
    asyncio.run(run_benchmark_with_qdrant(limit=args.limit))