import os
import json
import time
import asyncio
import numpy as np
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union, Optional
from qdrant_client import AsyncQdrantClient
import torch # Asegurar que torch está importado

# Ajustar sys.path para incluir el directorio del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importar usando ruta absoluta
from SpeculativeRag.main import speculative_rag

# Configurar logger
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(log_dir / "benchmark_speculative_qdrant.log", rotation="500 MB")

# Cargar variables de entorno desde el directorio raíz
load_dotenv(dotenv_path=project_root / ".env")

# Configuración para modelos locales
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
EMBEDDING_MODEL = "nomic-embed-text"  # Usando nomic-embed-text de Ollama
DRAFTER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Modelo cuantizado
VERIFIER_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Mismo modelo para verificación

LLM_MODEL = VERIFIER_MODEL # Modelo para evaluación

# Configuración de Qdrant
QUESTIONS_COLLECTION = "questions-benchmark"  # Colección para las preguntas del benchmark
RETRIEVAL_COLLECTION = "chunks"               # Colección para la recuperación de documentos

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

The reference answer represents the truth. The generated answer must match the meaning of the reference answer to be considered correct or if its a factoid answer it must be the same as the reference answer. If the generated answer is more specific but the core meaning is the same, it is also considered correct.

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


async def get_questions_from_qdrant(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Obtiene preguntas directamente desde la colección de Qdrant (formato actualizado)
    
    Args:
        limit: Número máximo de preguntas a recuperar
        
    Returns:
        Lista de preguntas con sus respuestas
    """
    try:
        logger.info(f"Obteniendo hasta {limit} preguntas de Qdrant de la colección {QUESTIONS_COLLECTION}")
        
        # Crear cliente Qdrant
        qdrant_path = project_root / "qdrant_client"
        qdrant_client = AsyncQdrantClient(path=qdrant_path)
        
        # Recuperar puntos de la colección de preguntas
        scroll_result = await qdrant_client.scroll(
            collection_name=QUESTIONS_COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Manejar tanto el formato antiguo (objeto) como el nuevo (tupla)
        if isinstance(scroll_result, tuple):
            points = scroll_result[0]
        else:
            points = scroll_result.points
        
        # Procesar las preguntas en el nuevo formato (solo con campo answer)
        questions = []
        skipped = 0
        
        for point in points:
            question_data = {
                "question_id": point.payload.get("question_id", ""),
                "question": point.payload.get("question", "")
            }
            
            # Verificar si tiene el campo answer y añadirlo como respuesta exacta
            if "answer" in point.payload and point.payload["answer"]:
                answer = point.payload["answer"]
                # Conservar el campo original para compatibilidad
                question_data["answer"] = answer
                
                # También almacenar como exact_answer para compatibilidad con el código existente
                if isinstance(answer, list):
                    question_data["exact_answer"] = answer
                else:
                    question_data["exact_answer"] = [answer]
                
                # Añadir como ideal_answer también
                question_data["ideal_answer"] = answer if isinstance(answer, str) else str(answer)
                
                questions.append(question_data)
            else:
                skipped += 1
        
        await qdrant_client.close()
        
        logger.info(f"Se obtuvieron {len(questions)} preguntas válidas de Qdrant")
        if skipped > 0:
            logger.info(f"Se saltaron {skipped} preguntas sin respuesta válida")
        
        # Si tenemos más preguntas que el límite, recortamos
        if len(questions) > limit:
            logger.info(f"Limitando resultado a {limit} preguntas")
            questions = questions[:limit]
            
        return questions
        
    except Exception as e:
        logger.error(f"Error obteniendo preguntas de Qdrant: {e}")
        return []

# Clase interna para capturar tiempos
class TimeCaptureLogger:
    def __init__(self):
        self.times = {}
        self.draft_times = []
        self.verify_times = []
        self.captured_output = []
    
    def write(self, text):
        self.captured_output.append(text) # Guardar toda la salida para depuración
        
        # Capturar tiempos de fases principales
        if "⏱️ Query embedding time:" in text:
            self.times["query_embedding_time"] = float(text.split(":")[1].strip().split()[0])
        elif "⏱️ InBedder processing time:" in text: # Cambiar a este nombre si se usa Inbedder
             self.times["inbedder_time"] = float(text.split(":")[1].strip().split()[0])
        elif "⏱️ Document retrieval time:" in text:
            self.times["retrieval_time"] = float(text.split(":")[1].strip().split()[0])
        elif "⏱️ Clustering time:" in text: # Captura del tiempo de clustering
             self.times["clustering_time"] = float(text.split(":")[1].strip().split()[0])
        elif "Time for clustering and subset generation:" in text: # Nombre alternativo si es diferente
             self.times["clustering_time"] = float(text.split(":")[1].strip().split()[0])
        
        # Capturar tiempos de drafts individuales
        elif "Draft " in text and "generado en" in text:
            try:
                time_str = text.split("generado en")[1].strip().split()[0]
                self.draft_times.append(float(time_str))
            except: pass
        
        # Capturar tiempos de verificaciones individuales
        elif "Verificación " in text and "completada en" in text:
            try:
                time_str = text.split("completada en")[1].strip().split()[0]
                self.verify_times.append(float(time_str))
            except: pass
        
        # Capturar tiempos totales de drafting y verificación (secuencial y paralelo)
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
    logger.info(f"Iniciando benchmark de Speculative RAG con {limit} preguntas")
    
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
        results_dir = Path(__file__).parent / "benchmark_results_speculative_qdrant"
    results_dir.mkdir(exist_ok=True)
    
    # Crear directorios para respuestas
    answer_dir = results_dir / "answers"
    answer_dir.mkdir(exist_ok=True)
    
    # Verificar si existen checkpoints previos
    ideal_checkpoint = results_dir / "ideal_answers" / "partial_results.json"
    exact_checkpoint = results_dir / "exact_answers" / "partial_results.json"
    
    # Cargar resultados previos si existen
    ideal_results = []
    exact_results = []
    processed_question_ids = set()
    
    if ideal_checkpoint.exists():
        try:
            with open(ideal_checkpoint, "r", encoding="utf-8") as f:
                ideal_results = json.load(f)
                logger.info(f"Cargados {len(ideal_results)} resultados previos de respuestas ideales")
                processed_question_ids.update([r["question_id"] for r in ideal_results])
        except Exception as e:
            logger.error(f"Error cargando checkpoint de respuestas ideales: {e}")
            ideal_results = []
    
    if exact_checkpoint.exists():
        try:
            with open(exact_checkpoint, "r", encoding="utf-8") as f:
                exact_results = json.load(f)
                logger.info(f"Cargados {len(exact_results)} resultados previos de respuestas exactas")
                processed_question_ids.update([r["question_id"] for r in exact_results])
        except Exception as e:
            logger.error(f"Error cargando checkpoint de respuestas exactas: {e}")
            exact_results = []
    
    # Usar preguntas proporcionadas o obtenerlas de Qdrant
    if questions is not None:
        all_questions = questions[:limit]  # Respetar el límite
    else:
        all_questions = await get_questions_from_qdrant(limit=limit)
    
    if not all_questions:
        logger.error("No se pudieron obtener preguntas de Qdrant. Abortando benchmark.")
        return
    
    # Filtrar preguntas que ya se procesaron
    questions = [q for q in all_questions if q["question_id"] not in processed_question_ids]
    
    total_questions = len(all_questions)
    remaining_questions = len(questions)
    
    logger.info(f"Total de preguntas: {total_questions}")
    logger.info(f"Preguntas ya procesadas: {total_questions - remaining_questions}")
    logger.info(f"Preguntas restantes por procesar: {remaining_questions}")
    
    # Si ya se procesaron todas las preguntas, salir
    if remaining_questions == 0:
        logger.info("Todas las preguntas ya fueron procesadas. Generando informes finales.")
        # Guardar todos los resultados (por si acaso no se guardaron al final)
        with open(results_dir / "ideal_answers" / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(ideal_results, f, indent=2, ensure_ascii=False)
        with open(results_dir / "exact_answers" / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(exact_results, f, indent=2, ensure_ascii=False)
        # Generar estadísticas
        await generate_stats_report(ideal_results, results_dir / "ideal_answers", "Ideal Answers")
        await generate_stats_report(exact_results, results_dir / "exact_answers", "Exact Answers")
        logger.info(f"Benchmark completado. Resultados guardados en {results_dir}")
        return
    
    # Procesar cada pregunta restante
    for i, question_data in enumerate(tqdm(questions, desc="Evaluando preguntas")):
        question = question_data["question"]
        question_id = question_data.get("question_id", "unknown")
        
        logger.info(f"Procesando pregunta {i+1}/{len(questions)} (ID: {question_id}): {question[:50]}...")
        
        # Ejecutar Speculative RAG una sola vez
        has_response = False
        response = ""
        execution_times = {}
        total_time = 0
        
        # Solo ejecutar si hay respuestas de referencia
        if "answer" in question_data or "ideal_answer" in question_data or "exact_answer" in question_data:
            try:
                # Capturar la salida para extraer los tiempos
                time_logger = TimeCaptureLogger()
                
                # Configurar parámetros para Speculative RAG
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
                    "ollama_client": ollama_client,
                    "llm_client": vllm_client,
                    "drafter_client": drafter_client,
                    "verifier_client": verifier_client
                }
                
                # Reemplazar temporalmente sys.stdout para capturar los tiempos
                original_stdout = sys.stdout
                sys.stdout = time_logger
                
                # Ejecutar Speculative RAG
                response, total_time = await speculative_rag(query=question, **params)
                has_response = True
                
                # Restaurar sys.stdout a su valor original
                sys.stdout = original_stdout
                
                # Obtener tiempos internos capturados
                execution_times = time_logger.times
                
            except Exception as e:
                # Registrar el error en el log
                logger.error(f"Error ejecutando Speculative RAG: {e}")
                
                # Establecer mensaje de error como respuesta
                response = "Error generando respuesta"
                
                # IMPORTANTE: Esta parte debe estar indentada
                # Verificar si la variable original_stdout existe y restaurarla
                if 'original_stdout' in locals():  # Comprobar si la variable existe
                    # ESTA LÍNEA DEBE ESTAR INDENTADA CON 4 ESPACIOS
                    sys.stdout = original_stdout  # Restaurar stdout original
        
        # Si no se generó respuesta, continuar con la siguiente pregunta
        if not has_response:
            logger.warning(f"No se generó respuesta para la pregunta ID: {question_id}")
            continue

        # Preferir usar el campo answer cuando está disponible
        if "answer" in question_data:
            # Código para evaluar con el campo answer
            reference_answer = question_data["answer"]
            
            # Evaluar usando la métrica correcta
            correctness = await evaluate_correctness(
                question=question,
                generated_answer=response,
                reference_answer=reference_answer,
                answer_type="answer",
                client=verifier_client
            )
            
            # Calcular otras métricas si es necesario
            cosine_sim = calculate_cosine_similarity(response, reference_answer)
            precision, recall, f1 = calculate_bert_score(response, reference_answer)
            
            # Guardar resultados
            result = {
                "question_id": question_id,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": response,
                "correctness": correctness,
                "cosine_similarity": cosine_sim,
                "bert_precision": precision,
                "bert_recall": recall,
                "bert_f1": f1,
                "total_time": total_time,
                "execution_times": execution_times
            }
            
            # Añadir a resultados
            exact_results.append(result)
            
            # Guardar resultados parciales
            with open(exact_checkpoint, "w", encoding="utf-8") as f:
                json.dump(exact_results, f, indent=2, ensure_ascii=False)
        
        

        await asyncio.sleep(1) # Pausa
    
    # Guardar resultados finales
    with open(results_dir / "ideal_answers" / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(ideal_results, f, indent=2, ensure_ascii=False)
    with open(results_dir / "exact_answers" / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(exact_results, f, indent=2, ensure_ascii=False)
    
    # Generar estadísticas finales
    await generate_stats_report(ideal_results, results_dir / "ideal_answers", "Speculative RAG - Ideal Answers")
    await generate_stats_report(exact_results, results_dir / "exact_answers", "Speculative RAG - Exact Answers")
    
    logger.info(f"Benchmark completado. Resultados guardados en {results_dir}")

async def generate_stats_report(results: List[Dict[str, Any]], output_dir: Path, report_title: str):
    """
    Genera un informe de estadísticas a partir de los resultados
    """
    if not results:
        logger.warning(f"No hay resultados para generar el informe: {report_title}")
        return
    
    # Extraer todas las métricas y tiempos disponibles
    metrics = {key: [r.get(key, 0) for r in results] for key in results[0].keys() if isinstance(results[0].get(key), (int, float))}
    
    # Extraer también los tiempos de ejecución desde execution_times
    execution_time_metrics = {}
    for time_key in ["draft_sequential_time", "draft_parallel_estimate", "verify_sequential_time", "verify_parallel_estimate"]:
        execution_time_metrics[time_key] = [r.get("execution_times", {}).get(time_key, 0) for r in results]
    
    stats = {}
    # Procesar métricas principales
    for metric_name, values in metrics.items():
        values = np.array(values)
        # Filtrar valores no finitos o nan si es necesario
        values = values[np.isfinite(values)]
        if len(values) == 0: continue # Saltar si no hay valores válidos

        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "q25": float(np.percentile(values, 25)),
            "q40": float(np.percentile(values, 40)),
            "q75": float(np.percentile(values, 75))
        }
    
    # Procesar tiempos de ejecución
    for metric_name, values in execution_time_metrics.items():
        values = np.array(values)
        values = values[np.isfinite(values)]
        if len(values) == 0: continue

        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
            "q25": float(np.percentile(values, 25)),
            "q40": float(np.percentile(values, 40)),
            "q75": float(np.percentile(values, 75))
        }
    
    # Calcular accuracy
    if "correctness" in stats:
        stats["accuracy"] = {"value": stats["correctness"]["mean"]}
    
    # Guardar estadísticas como JSON
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Crear informe markdown
    report = f"# Informe de Benchmark - {report_title}\n\n"
    report += f"Total de preguntas evaluadas: {len(results)}\n\n"
    
    if "accuracy" in stats:
        report += f"## Accuracy (Evaluada por modelo local)\n"
        report += f"- Accuracy: {stats['accuracy']['value']:.4f}\n\n"
    
    report += "## Estadísticas de Métricas de Calidad\n\n"
    quality_metrics = ["cosine_similarity", "bert_precision"]
    for metric in quality_metrics:
        if metric in stats:
            report += f"### {metric}\n"
            for key, value in stats[metric].items():
                report += f"- {key}: {value:.4f}\n"
            report += "\n"
            
    report += "## Estadísticas de Tiempos de Ejecución (segundos)\n\n"
    time_metrics = [key for key in stats if "time" in key and key != "draft_sequential_time" and 
                   key != "draft_parallel_estimate" and key != "verify_sequential_time" and 
                   key != "verify_parallel_estimate"]
    for metric in time_metrics:
        report += f"### {metric}\n"
        for key, value in stats[metric].items():
            report += f"- {key}: {value:.4f}\n"
        report += "\n"

    # Añadir sección de comparación secuencial vs paralelo
    report += "## Comparación de Tiempos Secuenciales vs Paralelos (Promedios)\n\n"
    avg_draft_seq = stats.get("draft_sequential_time", {}).get("mean", 0)
    avg_draft_par = stats.get("draft_parallel_estimate", {}).get("mean", 0)
    avg_verify_seq = stats.get("verify_sequential_time", {}).get("mean", 0)
    avg_verify_par = stats.get("verify_parallel_estimate", {}).get("mean", 0)
    
    # Añadir estos tiempos al informe
    report += f"### Tiempo de Drafting\n"
    report += f"- Secuencial (promedio): {avg_draft_seq:.4f}s\n"
    report += f"- Paralelo estimado (promedio): {avg_draft_par:.4f}s\n"
    if avg_draft_par > 0:
        report += f"- Speedup: {avg_draft_seq/avg_draft_par:.2f}x\n\n"
    else:
        report += f"- Speedup: N/A\n\n"
        
    report += f"### Tiempo de Verificación\n"
    report += f"- Secuencial (promedio): {avg_verify_seq:.4f}s\n"
    report += f"- Paralelo estimado (promedio): {avg_verify_par:.4f}s\n"
    if avg_verify_par > 0:
        report += f"- Speedup: {avg_verify_seq/avg_verify_par:.2f}x\n\n"
    else:
        report += f"- Speedup: N/A\n\n"
    
    # Tiempo total de LLM y speedup
    seq_llm_time = avg_draft_seq + avg_verify_seq
    par_llm_time = avg_draft_par + avg_verify_par
    
    report += f"### Tiempo Total LLM\n"
    report += f"- Secuencial (Drafting + Verificación): {seq_llm_time:.4f}s\n"
    report += f"- Paralelo Estimado (Drafting + Verificación): {par_llm_time:.4f}s\n"
    if par_llm_time > 0:
        report += f"- Speedup Total Estimado: {seq_llm_time/par_llm_time:.2f}x\n\n"
    else:
        report += f"- Speedup Total Estimado: N/A\n\n"

    # Guardar informe markdown
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info(f"Informe de estadísticas generado: {output_dir / 'report.md'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Speculative RAG con preguntas de Qdrant")
    parser.add_argument("--limit", type=int, default=300, help="Número máximo de preguntas a evaluar (default: 1000)")
    args = parser.parse_args()
    
    logger.info(f"Benchmark configurado para usar preguntas de '{QUESTIONS_COLLECTION}' y recuperar documentos de '{RETRIEVAL_COLLECTION}'")
    logger.info(f"Se evaluarán hasta {args.limit} preguntas")
    
    asyncio.run(run_benchmark_with_qdrant(limit=args.limit))
