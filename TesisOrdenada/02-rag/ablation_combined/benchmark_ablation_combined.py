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
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client import AsyncQdrantClient
import torch
import itertools
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Añadir el directorio raíz del proyecto a sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importar los módulos necesarios
from TraditionalRag_CombinedRetrieve.traditional_rag_combined_retrieve_flexible import traditional_rag_combined_retrieve_flexible

# Configurar logger específico
logger.add("benchmark_ablation_combined_logs.log", rotation="500 MB")

# Cargar variables de entorno
load_dotenv()

# Configuración
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Modelo para el cálculo de similitud coseno
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:1")

def calculate_cosine_similarity(text1: str, text2: Any) -> float:
    """Calcula la similitud coseno entre dos textos"""
    if isinstance(text1, list):
        text1 = str(text1)
    if isinstance(text2, list):
        text2 = str(text2)
    
    if not text1 or not text2:
        logger.warning("Uno de los textos está vacío o es None. Retornando similitud 0.0")
        return 0.0
    
    try:
        device = torch.device("cuda:1")
        
        if not hasattr(calculate_cosine_similarity, 'sentence_model'):
            calculate_cosine_similarity.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:1")
        
        embedding1 = calculate_cosine_similarity.sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = calculate_cosine_similarity.sentence_model.encode(text2, convert_to_tensor=True)
        
        embedding1 = embedding1.to(device)
        embedding2 = embedding2.to(device)
        
        if len(embedding1.shape) > 1 and embedding1.shape[0] > 1:
            embedding1 = torch.mean(embedding1, dim=0)
        if len(embedding2.shape) > 1 and embedding2.shape[0] > 1:
            embedding2 = torch.mean(embedding2, dim=0)
            
        similarity = util.pytorch_cos_sim(embedding1, embedding2)
        
        if isinstance(similarity, torch.Tensor) and similarity.numel() > 1:
            similarity = similarity.mean()
            
        return similarity.cpu().item()
    except Exception as e:
        logger.error(f"Error calculando similitud coseno: {e}")
        return 0.0

def calculate_bert_score(candidate: str, reference: Any) -> Tuple[float, float, float]:
    """Calcula BERTScore entre respuesta candidata y referencia"""
    if isinstance(candidate, list):
        candidate = str(candidate)
    if isinstance(reference, list):
        try:
            reference = str(max(reference, key=lambda x: len(str(x))))
        except:
            reference = str(reference)
    
    try:
        P, R, F1 = score([candidate], [reference], lang="en", verbose=False, device='cuda:1')
        return P.item(), R.item(), F1.item()
    except Exception as e:
        logger.error(f"Error al calcular BERTScore: {e}")
        return 0.0, 0.0, 0.0

def extract_answer_before_explanation(text: str) -> str:
    """Extrae solo la respuesta antes de 'Explanation:' si existe"""
    # Buscar diferentes variaciones de "Explanation:"
    explanation_markers = ["Explanation:", "### Explanation:", "**Explanation:**", "\n\nExplanation:"]
    
    for marker in explanation_markers:
        if marker in text:
            return text.split(marker)[0].strip()
    return text.strip()

def calculate_exact_match(generated_answer: str, reference_answer: str) -> int:
    """
    Calcula exact match entre respuesta generada y referencia
    Usa contención en lugar de igualdad exacta para manejar puntuación y espacios
    """
    clean_generated = extract_answer_before_explanation(generated_answer).lower().strip()
    clean_reference = reference_answer.lower().strip()
    
    # Usar contención en lugar de igualdad exacta para manejar puntuación y espacios
    return 1 if clean_reference in clean_generated or clean_generated in clean_reference else 0

async def evaluate_correctness(
    question: str,
    generated_answer: str,
    reference_answer: str,
    client: AsyncOpenAI = None
) -> int:
    """Evalúa si la respuesta generada es correcta usando el LLM con criterios más flexibles"""
    try:
        clean_generated_answer = extract_answer_before_explanation(generated_answer)
        
        # Prompt mejorado y más flexible
        prompt = f"""You are an expert evaluator for question answering systems. Your task is to determine if the generated answer correctly responds to the question according to the reference answer.

Question: {question}
Generated Answer: {clean_generated_answer}
Reference Answer: {reference_answer}

EVALUATION CRITERIA:
The generated answer is CORRECT if it contains or conveys the same information as the reference answer, even if it includes additional details or context.

Examples of CORRECT evaluations:
- Reference: "Paris" | Generated: "The story takes place in Paris" → CORRECT
- Reference: "1926 Paris" | Generated: "The story takes place in 1926 Paris during the Lost Generation" → CORRECT  
- Reference: "yes" | Generated: "Yes, that is correct" → CORRECT
- Reference: "Einstein" | Generated: "Albert Einstein developed the theory" → CORRECT

Examples of INCORRECT evaluations:
- Reference: "yes" | Generated: "No" → INCORRECT
- Reference: "Paris" | Generated: "London" → INCORRECT
- Reference: "1926" | Generated: "1925" → INCORRECT

IMPORTANT RULES:
1. If the reference answer is contained within the generated answer, it is CORRECT
2. Additional explanation, context, or details do not make an answer incorrect
3. Only mark as INCORRECT if the core factual claim contradicts or is missing from the generated answer
4. For yes/no questions, the stance must match (yes=yes, no=no)

Respond with ONLY a single digit:
1 - CORRECT: The generated answer contains or aligns with the reference answer
0 - INCORRECT: The generated answer contradicts or lacks the reference answer

Your verdict (just the digit 1 or 0):"""
        
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
        return 0

async def generate_combined_rag_response(
    question: str,
    llm_client: AsyncOpenAI,
    embeddings_client: AsyncOpenAI,
    top_k: int = 15,
    alpha: float = 0.5,
    num_similar_questions: int = 2
) -> Tuple[str, float, float]:
    """
    Genera una respuesta con Combined RAG flexible y captura los tiempos
    """
    start_time = time.time()
    retrieval_time = 0
    
    # Capturar tiempo de recuperación desde la salida estándar
    class TimeCaptureLogger:
        def __init__(self):
            self.retrieval_time = 0
            
        def write(self, text):
            if "retrieval_time:" in text:
                try:
                    time_str = text.split("retrieval_time:")[1].strip()
                    self.retrieval_time = float(time_str)
                except:
                    pass
    
    # Redirigir stdout temporalmente para capturar el tiempo
    import sys
    original_stdout = sys.stdout
    time_logger = TimeCaptureLogger()
    sys.stdout = time_logger
    
    try:
        response, total_time = await traditional_rag_combined_retrieve_flexible(
            query=question,
            embeddings_client=embeddings_client,
            llm_client=llm_client,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL,
            top_k=top_k,
            alpha=alpha,
            num_similar_questions=num_similar_questions,
            debug_prints=False,
            verbose_timing=False
        )
        
        # Restaurar stdout
        sys.stdout = original_stdout
        
        # Obtener el tiempo de recuperación capturado
        retrieval_time = time_logger.retrieval_time
        
        return response, total_time, retrieval_time
        
    except Exception as e:
        # Restaurar stdout en caso de error
        sys.stdout = original_stdout
        logger.error(f"Error generando respuesta Combined RAG: {e}")
        raise
    finally:
        # Asegurar que stdout se restaure
        sys.stdout = original_stdout

async def load_questions_from_results_file(results_file_path: str) -> List[Dict[str, Any]]:
    """
    Carga las preguntas específicas usadas en un archivo de resultados previo
    """
    try:
        logger.info(f"Cargando preguntas desde: {results_file_path}")
        
        # Leer el archivo JSON de resultados
        with open(results_file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Extraer los question_ids únicos
        question_ids = []
        for result in results_data:
            if 'question_id' in result:
                question_ids.append(result['question_id'])
        
        unique_question_ids = list(set(question_ids))
        logger.info(f"Se encontraron {len(unique_question_ids)} question_ids únicos")
        
        # Obtener las preguntas de Qdrant usando los question_ids específicos
        qdrant_path = project_root / "qdrant_client"
        qdrant_client = AsyncQdrantClient(path=qdrant_path)
        
        questions = []
        for question_id in unique_question_ids:
            try:
                # Buscar la pregunta específica por question_id usando la sintaxis correcta
                search_response = await qdrant_client.scroll(
                    collection_name="questions-benchmark",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="question_id",
                                match=MatchValue(value=question_id)
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                
                points = search_response[0] if isinstance(search_response, tuple) else search_response.points
                
                if points:
                    point = points[0]
                    if "answer" in point.payload and point.payload["answer"]:
                        question_data = {
                            "question_id": str(point.payload.get("question_id", "")),
                            "question": point.payload.get("question", ""),
                            "answer": point.payload["answer"]
                        }
                        
                        # Añadir campos de compatibilidad
                        answer = point.payload["answer"]
                        if isinstance(answer, list):
                            question_data["exact_answer"] = answer
                        else:
                            question_data["exact_answer"] = answer
                        
                        question_data["ideal_answer"] = answer if isinstance(answer, str) else str(answer)
                        questions.append(question_data)
                        
            except Exception as e:
                logger.warning(f"Error obteniendo pregunta {question_id}: {e}")
                continue
        
        await qdrant_client.close()
        
        logger.info(f"Se cargaron {len(questions)} preguntas válidas desde Qdrant")
        return questions
        
    except Exception as e:
        logger.error(f"Error cargando preguntas desde archivo de resultados: {e}")
        return []

async def get_questions_from_qdrant(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Obtiene preguntas directamente desde la colección de Qdrant
    Usa la misma lógica que el benchmark original
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

async def run_ablation_experiment(
    alpha: float,
    num_similar_questions: int,
    questions: List[Dict[str, Any]],
    results_dir: Path,
    top_k: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Ejecuta un experimento de ablación con parámetros específicos
    """
    logger.info(f"Iniciando experimento: alpha={alpha}, num_similar={num_similar_questions}")
    
    # Crear directorio específico para este experimento
    exp_name = f"alpha_{alpha:.2f}_similar_{num_similar_questions}"
    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar clientes
    llm_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    embeddings_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    
    results = []
    total_questions = len(questions)
    
    # Métricas acumulativas
    total_exact_matches = 0
    total_llm_correct = 0
    total_cosine_similarity = 0.0
    total_bert_f1 = 0.0
    total_response_time = 0.0
    total_retrieval_time = 0.0
    
    logger.info(f"Procesando {total_questions} preguntas...")
    
    for i, question_data in enumerate(tqdm(questions, desc=f"Alpha {alpha}, Similar {num_similar_questions}")):
        try:
            question = question_data['question']
            reference_answer = question_data['answer']  # Usar 'answer' en lugar de 'exact_answer'
            
            # Generar respuesta
            start_time = time.time()
            generated_answer, response_time, retrieval_time = await generate_combined_rag_response(
                question=question,
                llm_client=llm_client,
                embeddings_client=embeddings_client,
                top_k=top_k,
                alpha=alpha,
                num_similar_questions=num_similar_questions
            )
            
            # Calcular métricas - Manejar tanto listas como strings
            try:
                if isinstance(reference_answer, list):
                    # Para respuestas exactas, usamos la mejor puntuación individual para similitud coseno
                    best_cosine_sim = max([calculate_cosine_similarity(generated_answer, str(ref)) for ref in reference_answer]) if reference_answer else 0.0
                    # Para BERTScore, tomamos la referencia más larga
                    representative_ref = str(max(reference_answer, key=lambda x: len(str(x)))) if reference_answer else ""
                    bert_p, bert_r, bert_f1 = calculate_bert_score(generated_answer, representative_ref)
                    # Para exact match, probamos con todas las referencias
                    exact_match = max([calculate_exact_match(generated_answer, str(ref)) for ref in reference_answer]) if reference_answer else 0
                    # Para LLM evaluation, usar la referencia representativa
                    llm_correct = await evaluate_correctness(question, generated_answer, representative_ref, llm_client)
                else:
                    # Para respuestas exactas singulares
                    cosine_sim = calculate_cosine_similarity(generated_answer, str(reference_answer))
                    best_cosine_sim = cosine_sim
                    bert_p, bert_r, bert_f1 = calculate_bert_score(generated_answer, str(reference_answer))
                    exact_match = calculate_exact_match(generated_answer, str(reference_answer))
                    llm_correct = await evaluate_correctness(question, generated_answer, str(reference_answer), llm_client)
            except Exception as e:
                logger.error(f"Error calculando métricas: {e}")
                best_cosine_sim = 0.0
                bert_p, bert_r, bert_f1 = 0.0, 0.0, 0.0
                exact_match = 0
                llm_correct = 0
            
            # Acumular métricas
            total_exact_matches += exact_match
            total_llm_correct += llm_correct
            total_cosine_similarity += best_cosine_sim
            total_bert_f1 += bert_f1
            total_response_time += response_time
            total_retrieval_time += retrieval_time
            
            # Guardar resultado individual
            result = {
                'question_id': question_data['question_id'],
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference_answer,
                'exact_match': exact_match,
                'llm_correct': llm_correct,
                'cosine_similarity': best_cosine_sim,
                'bert_precision': bert_p,
                'bert_recall': bert_r,
                'bert_f1': bert_f1,
                'response_time': response_time,
                'retrieval_time': retrieval_time,
                'alpha': alpha,
                'num_similar_questions': num_similar_questions,
                'top_k': top_k
            }
            results.append(result)
            
            # Log progreso cada 50 preguntas
            if (i + 1) % 50 == 0:
                current_exact_match_rate = total_exact_matches / (i + 1)
                current_llm_accuracy = total_llm_correct / (i + 1)
                logger.info(f"Progreso: {i+1}/{total_questions}, Exact Match: {current_exact_match_rate:.4f}, LLM Accuracy: {current_llm_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error procesando pregunta {i}: {e}")
            continue
    
    # Calcular métricas finales
    num_processed = len(results)
    if num_processed > 0:
        metrics = {
            'alpha': alpha,
            'num_similar_questions': num_similar_questions,
            'top_k': top_k,
            'total_questions': num_processed,
            'exact_match_rate': total_exact_matches / num_processed,
            'llm_accuracy': total_llm_correct / num_processed,
            'avg_cosine_similarity': total_cosine_similarity / num_processed,
            'avg_bert_f1': total_bert_f1 / num_processed,
            'avg_response_time': total_response_time / num_processed,
            'avg_retrieval_time': total_retrieval_time / num_processed,
            'seed': seed
        }
    else:
        metrics = {
            'alpha': alpha,
            'num_similar_questions': num_similar_questions,
            'error': 'No questions processed successfully'
        }
    
    # Guardar resultados
    with open(exp_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Experimento completado: {exp_name}")
    logger.info(f"Exact Match Rate: {metrics.get('exact_match_rate', 0):.4f}")
    logger.info(f"LLM Accuracy: {metrics.get('llm_accuracy', 0):.4f}")
    
    return metrics

async def run_ablation_study(
    n_questions: int = 300,
    seed: int = 42,
    top_k: int = 15,
    results_file_path: Optional[str] = None
) -> None:
    """
    Ejecuta el estudio completo de ablación
    """
    logger.info(f"=== INICIANDO ESTUDIO DE ABLACIÓN ===")
    logger.info(f"Parámetros: {n_questions} preguntas, seed={seed}, top_k={top_k}")
    
    # Crear directorio de resultados
    timestamp = int(time.time())
    results_dir = Path("ablation_combined") / f"ablation_study_seed{seed}_topk{top_k}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener preguntas
    if results_file_path:
        logger.info(f"Usando preguntas específicas del archivo: {results_file_path}")
        all_questions = await load_questions_from_results_file(results_file_path)
        if not all_questions:
            logger.error("No se pudieron cargar preguntas del archivo especificado, usando método predeterminado")
            all_questions = await get_questions_from_qdrant(limit=1000)
    else:
        logger.info("Obteniendo preguntas de Qdrant...")
        all_questions = await get_questions_from_qdrant(limit=1000)
    
    if len(all_questions) < n_questions:
        logger.warning(f"Solo se encontraron {len(all_questions)} preguntas, usando todas")
        n_questions = len(all_questions)
    
    # Seleccionar preguntas con seed fija
    import random
    random.seed(seed)
    if results_file_path:
        # Si estamos usando preguntas específicas, usar todas sin selección aleatoria
        selected_questions = all_questions[:n_questions]
        logger.info(f"Usando {len(selected_questions)} preguntas específicas del archivo")
    else:
        selected_questions = random.sample(all_questions, n_questions)
        logger.info(f"Seleccionadas {len(selected_questions)} preguntas aleatoriamente")
    
    # Definir parámetros para el estudio de ablación
    alpha_values = [0.75, 0]  # Porcentaje para pregunta original
    num_similar_values = [1, 2, 3, 4]  # Número de preguntas similares

    logger.info(f"Configuración del estudio:")
    logger.info(f"- Alpha values: {alpha_values}")
    logger.info(f"- Num similar values: {num_similar_values}")
    logger.info(f"- Total experimentos: {len(alpha_values) * len(num_similar_values)}")
    
    # Ejecutar todos los experimentos
    all_metrics = []
    total_experiments = len(alpha_values) * len(num_similar_values)
    experiment_count = 0
    
    for alpha in alpha_values:
        for num_similar in num_similar_values:
            experiment_count += 1
            logger.info(f"\n--- Experimento {experiment_count}/{total_experiments} ---")
            
            try:
                metrics = await run_ablation_experiment(
                    alpha=alpha,
                    num_similar_questions=num_similar,
                    questions=selected_questions,
                    results_dir=results_dir,
                    top_k=top_k,
                    seed=seed
                )
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"Error en experimento alpha={alpha}, num_similar={num_similar}: {e}")
                continue
    
    # Crear resumen final
    summary = {
        'study_parameters': {
            'n_questions': n_questions,
            'seed': seed,
            'top_k': top_k,
            'alpha_values': alpha_values,
            'num_similar_values': num_similar_values,
            'total_experiments': len(all_metrics),
            'results_file_path': results_file_path
        },
        'results': all_metrics,
        'timestamp': timestamp,
        'results_dir': str(results_dir)
    }
    
    # Guardar resumen
    with open(results_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Crear análisis de resultados
    await create_ablation_analysis(all_metrics, results_dir)
    
    logger.info(f"\n=== ESTUDIO DE ABLACIÓN COMPLETADO ===")
    logger.info(f"Resultados guardados en: {results_dir}")
    logger.info(f"Experimentos completados: {len(all_metrics)}/{total_experiments}")

async def create_ablation_analysis(metrics: List[Dict[str, Any]], results_dir: Path) -> None:
    """Crea análisis detallado de los resultados de ablación"""
    
    # Convertir a DataFrame para análisis
    df = pd.DataFrame(metrics)
    
    if df.empty:
        logger.warning("No hay métricas para analizar")
        return
    
    # Encontrar mejores configuraciones
    best_exact_match = df.loc[df['exact_match_rate'].idxmax()]
    best_llm_accuracy = df.loc[df['llm_accuracy'].idxmax()]
    best_speed = df.loc[df['avg_response_time'].idxmin()]
    
    # Crear reporte de análisis
    analysis = {
        'best_configurations': {
            'exact_match': {
                'alpha': best_exact_match['alpha'],
                'num_similar_questions': best_exact_match['num_similar_questions'],
                'exact_match_rate': best_exact_match['exact_match_rate'],
                'llm_accuracy': best_exact_match['llm_accuracy'],
                'avg_response_time': best_exact_match['avg_response_time']
            },
            'llm_accuracy': {
                'alpha': best_llm_accuracy['alpha'],
                'num_similar_questions': best_llm_accuracy['num_similar_questions'],
                'exact_match_rate': best_llm_accuracy['exact_match_rate'],
                'llm_accuracy': best_llm_accuracy['llm_accuracy'],
                'avg_response_time': best_llm_accuracy['avg_response_time']
            },
            'fastest': {
                'alpha': best_speed['alpha'],
                'num_similar_questions': best_speed['num_similar_questions'],
                'exact_match_rate': best_speed['exact_match_rate'],
                'llm_accuracy': best_speed['llm_accuracy'],
                'avg_response_time': best_speed['avg_response_time']
            }
        },
        'statistics': {
            'exact_match_rate': {
                'mean': df['exact_match_rate'].mean(),
                'std': df['exact_match_rate'].std(),
                'min': df['exact_match_rate'].min(),
                'max': df['exact_match_rate'].max()
            },
            'llm_accuracy': {
                'mean': df['llm_accuracy'].mean(),
                'std': df['llm_accuracy'].std(),
                'min': df['llm_accuracy'].min(),
                'max': df['llm_accuracy'].max()
            },
            'avg_response_time': {
                'mean': df['avg_response_time'].mean(),
                'std': df['avg_response_time'].std(),
                'min': df['avg_response_time'].min(),
                'max': df['avg_response_time'].max()
            }
        }
    }
    
    # Guardar análisis
    with open(results_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Crear CSV para fácil visualización
    df.to_csv(results_dir / "ablation_results.csv", index=False)
    
    logger.info("Análisis de ablación creado")
    logger.info(f"Mejor Exact Match: alpha={best_exact_match['alpha']}, similar={best_exact_match['num_similar_questions']}, rate={best_exact_match['exact_match_rate']:.4f}")
    logger.info(f"Mejor LLM Accuracy: alpha={best_llm_accuracy['alpha']}, similar={best_llm_accuracy['num_similar_questions']}, rate={best_llm_accuracy['llm_accuracy']:.4f}")

async def main():
    """Función principal"""
    # Especificar la ruta del archivo de resultados previo
    results_file_path = "solver_exact_match_runs/traditional_vs_combined_seed1_topk15_1748213811/combined/exact_answers/full_results_llm_evaluated.json"
    
    await run_ablation_study(
        n_questions=300,
        seed=23,
        top_k=15,
        results_file_path=results_file_path
    )

if __name__ == "__main__":
    asyncio.run(main()) 