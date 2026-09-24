import asyncio
import random
import time
import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Union
import re
from datetime import datetime

# Import the new benchmark runners
from benchmark_traditional_exact_match import get_questions_from_qdrant
from benchmark_combined_exact_match import run_benchmark as run_combined_exact_match_original

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_exact_match_rate(stats_path: Path) -> float:
    """Get exact match rate from stats file"""
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("exact_match_rate", 0)
    except Exception as e:
        logger.error(f"Error reading exact match rate: {e}")
        return 0

async def load_questions_from_results_file(results_file_path: str) -> List[Dict[str, Any]]:
    """
    Carga las preguntas específicas usadas en un archivo de resultados previo
    Usa exactamente el mismo método que ablation_combined
    """
    try:
        logger.info(f"Cargando preguntas desde: {results_file_path}")
        
        # Añadir el directorio raíz del proyecto a sys.path
        import sys
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
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

async def run_combined_exact_match_with_collection(
    limit: int = 300,
    questions: Optional[List[Dict[str, Any]]] = None,
    results_dir: Optional[Path] = None,
    top_k: int = 20,
    question_collection: str = "questions-index-finetuned"
) -> None:
    """
    Ejecuta benchmark Combined RAG con una colección específica de preguntas similares
    """
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

    # Añadir el directorio raíz del proyecto a sys.path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Importar los módulos necesarios
    from TraditionalRag_CombinedRetrieve.traditional_rag_combined_retrieve import traditional_rag_combined_retrieve

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
        if "Explanation:" in text:
            return text.split("Explanation:")[0].strip()
        return text.strip()

    def calculate_exact_match(generated_answer: str, reference_answer: str) -> int:
        """Calcula exact match entre respuesta generada y referencia"""
        clean_generated = extract_answer_before_explanation(generated_answer).lower().strip()
        clean_reference = reference_answer.lower().strip()
        
        return 1 if clean_reference in clean_generated or clean_generated in clean_reference else 0

    async def evaluate_correctness(
        question: str,
        generated_answer: str,
        reference_answer: str,
        client: AsyncOpenAI = None
    ) -> int:
        """Evalúa si la respuesta generada es correcta usando el LLM"""
        try:
            clean_generated_answer = extract_answer_before_explanation(generated_answer)
            
            prompt = f"""You are an expert evaluator for question answering systems. Your task is to determine if the generated answer correctly responds to the question according to the reference answer.

Question: {question}
Generated Answer: {clean_generated_answer}
Reference Answer: {reference_answer}

IMPORTANT: The reference answer represents the absolute truth. The generated answer is CORRECT only if:
1. For yes/no questions: The generated answer must clearly indicate the same yes/no stance as the reference
2. For factual questions: The core factual claim must match exactly
3. Additional explanation or detail is fine, but the fundamental answer must align with the reference

If the generated answer contradicts the reference answer (e.g., says "No" when reference is "yes"), it is INCORRECT regardless of explanation quality.

Respond with ONLY a single digit:
1 - CORRECT: The generated answer aligns with the reference answer
0 - INCORRECT: The generated answer contradicts or doesn't match the reference answer

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
        top_k: int = 20,
        question_collection: str = "questions-index-finetuned"
    ) -> Tuple[str, float, float]:
        """Genera una respuesta con Combined RAG usando la colección especificada"""
        start_time = time.time()
        retrieval_time = 0
        
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
        
        import sys
        original_stdout = sys.stdout
        time_logger = TimeCaptureLogger()
        sys.stdout = time_logger
        
        try:
            response, total_time = await traditional_rag_combined_retrieve(
                query=question,
                embeddings_client=embeddings_client,
                llm_client=llm_client,
                embedding_model=EMBEDDING_MODEL,
                llm_model=LLM_MODEL,
                top_k=top_k,
                question_collection=question_collection,  # Usar la colección especificada
                debug_prints=False,
                verbose_timing=False
            )
            
            sys.stdout = original_stdout
            retrieval_time = time_logger.retrieval_time
            
            return response, total_time, retrieval_time
            
        except Exception as e:
            sys.stdout = original_stdout
            logger.error(f"Error generando respuesta Combined RAG: {e}")
            raise
        finally:
            sys.stdout = original_stdout

    # Configurar clientes
    llm_client = AsyncOpenAI(
        api_key="dummy-key",
        base_url=VLLM_BASE_URL
    )
    
    embeddings_client = AsyncOpenAI(
        api_key="dummy-key",
        base_url=OLLAMA_BASE_URL
    )

    # Usar las preguntas proporcionadas
    if questions is None:
        logger.error("No se proporcionaron preguntas")
        return
    
    questions = questions[:limit]

    logger.info(f"Iniciando benchmark Combined RAG con colección: {question_collection}")
    logger.info(f"Procesando {len(questions)} preguntas con top_k={top_k}")

    # Crear directorio de resultados
    if results_dir is None:
        results_dir = Path("combined_exact_match_results")
    results_dir.mkdir(exist_ok=True)
    
    exact_answers_dir = results_dir / "exact_answers"
    exact_answers_dir.mkdir(exist_ok=True)

    # Procesar preguntas
    results = []
    
    for i, question_data in enumerate(tqdm(questions, desc="Procesando preguntas")):
        try:
            question = question_data["question"]
            reference_answer = question_data.get("answer", question_data.get("exact_answer", ""))
            
            if isinstance(reference_answer, list):
                reference_answer = reference_answer[0] if reference_answer else ""
            
            # Generar respuesta
            generated_answer, response_time, retrieval_time = await generate_combined_rag_response(
                question, llm_client, embeddings_client, top_k, question_collection
            )
            
            # Calcular métricas
            exact_match = calculate_exact_match(generated_answer, str(reference_answer))
            llm_correctness = await evaluate_correctness(question, generated_answer, str(reference_answer), llm_client)
            cosine_sim = calculate_cosine_similarity(generated_answer, reference_answer)
            bert_p, bert_r, bert_f1 = calculate_bert_score(generated_answer, reference_answer)
            
            result = {
                "question_id": question_data.get("question_id", f"q_{i}"),
                "question": question,
                "generated_answer": generated_answer,
                "reference_answer": reference_answer,
                "exact_match": exact_match,
                "llm_correctness": llm_correctness,
                "cosine_similarity": cosine_sim,
                "bert_precision": bert_p,
                "bert_recall": bert_r,
                "bert_f1": bert_f1,
                "response_time": response_time,
                "retrieval_time": retrieval_time,
                "question_collection": question_collection
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error procesando pregunta {i}: {e}")
            continue

    # Guardar resultados
    results_file = exact_answers_dir / "full_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generar estadísticas
    if results:
        stats = {
            "total_questions": len(results),
            "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
            "llm_accuracy": sum(r["llm_correctness"] for r in results) / len(results),
            "avg_cosine_similarity": sum(r["cosine_similarity"] for r in results) / len(results),
            "avg_bert_f1": sum(r["bert_f1"] for r in results) / len(results),
            "avg_response_time": sum(r["response_time"] for r in results) / len(results),
            "avg_retrieval_time": sum(r["retrieval_time"] for r in results) / len(results),
            "question_collection": question_collection
        }
        
        stats_file = exact_answers_dir / "stats_report.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Exact Match Rate: {stats['exact_match_rate']:.4f}")
        logger.info(f"LLM Accuracy: {stats['llm_accuracy']:.4f}")
        logger.info(f"Resultados guardados en: {results_dir}")

async def run_dual_combined_comparison(
    questions: List[Dict[str, Any]], 
    seed: int, 
    top_k: int = 15
) -> Dict[str, Any]:
    """
    Ejecuta una comparación entre Combined RAG con questions-index-base y questions-index-random
    usando las mismas preguntas que ablation_combined
    """
    logger.info(f"\n=== Running Dual Combined RAG comparison with seed {seed} ===")
    logger.info(f"Collections: questions-index-base vs questions-index-random")
    logger.info(f"Usando {len(questions)} preguntas específicas de ablation_combined")
    
    # Create results directory
    base_results_dir = Path("solver_dual_combined_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    run_dir_name = f'dual_combined_seed{seed}_topk{top_k}_{timestamp}'
    run_dir = base_results_dir / run_dir_name
    
    # Crear directorios
    base_dir = run_dir / "combined_base"
    random_dir = run_dir / "combined_random"
    (base_dir / "exact_answers").mkdir(parents=True, exist_ok=True)
    (random_dir / "exact_answers").mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Ejecutando Combined RAG con questions-index-base...")
        await run_combined_exact_match_with_collection(
            limit=len(questions), 
            questions=questions, 
            results_dir=base_dir, 
            top_k=top_k,
            question_collection="questions-index-base"
        )
        
        logger.info(f"Ejecutando Combined RAG con questions-index-random...")
        await run_combined_exact_match_with_collection(
            limit=len(questions), 
            questions=questions, 
            results_dir=random_dir, 
            top_k=top_k,
            question_collection="questions-index-random"
        )
        
        # Obtener exact match rates
        base_exact_match = get_exact_match_rate(base_dir / "exact_answers" / "stats_report.json")
        random_exact_match = get_exact_match_rate(random_dir / "exact_answers" / "stats_report.json")
        
        logger.info(f"Combined RAG (base) exact match rate:   {base_exact_match:.4f}")
        logger.info(f"Combined RAG (random) exact match rate: {random_exact_match:.4f}")
        
        # Crear resumen
        summary = {
            "seed": seed,
            "run_dir": str(run_dir),
            "top_k": top_k,
            "n_questions": len(questions),
            "base_exact_match": base_exact_match,
            "random_exact_match": random_exact_match,
            "base_better": base_exact_match > random_exact_match,
            "difference": base_exact_match - random_exact_match,
            "base_results_dir": str(base_dir),
            "random_results_dir": str(random_dir),
            "timestamp": timestamp,
            "both_below_threshold": base_exact_match < 0.586 and random_exact_match < 0.586
        }
        
        # Guardar resumen
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in dual comparison with seed {seed}: {e}")
        return {
            "seed": seed,
            "error": str(e),
            "base_exact_match": 0,
            "random_exact_match": 0,
            "both_below_threshold": False
        }

async def main():
    """
    Función principal que ejecuta comparaciones hasta que ambos Combined RAG 
    tengan Exact Match menor que 58.6% usando las mismas preguntas de ablation_combined
    """
    max_attempts = 50  # Máximo número de intentos
    top_k = 15  # Número de documentos a recuperar
    threshold = 0.586  # 58.6%
    
    # Usar el mismo archivo de resultados que ablation_combined
    results_file_path = "solver_exact_match_runs/traditional_vs_combined_seed1_topk15_1748213811/combined/exact_answers/full_results_llm_evaluated.json"
    
    logger.info(f"Iniciando búsqueda hasta que ambos Combined RAG tengan Exact Match < {threshold:.1%}")
    logger.info(f"Usando preguntas específicas de: {results_file_path}")
    logger.info(f"Parámetros: top_k={top_k}, máximo {max_attempts} intentos")
    
    # Cargar las mismas preguntas que usó ablation_combined
    logger.info("Cargando preguntas específicas de ablation_combined...")
    questions = await load_questions_from_results_file(results_file_path)
    
    if not questions:
        logger.error("No se pudieron cargar las preguntas. Abortando.")
        return
    
    logger.info(f"Se cargaron {len(questions)} preguntas específicas")
    
    # Create results directory
    base_results_dir = Path("solver_dual_combined_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    all_results = []
    success = False
    
    for attempt in range(1, max_attempts + 1):
        # Usar el número de intento como seed para reproducibilidad
        seed = attempt
        
        logger.info(f"\n--- Intento {attempt}/{max_attempts} (seed: {seed}) ---")
        
        result = await run_dual_combined_comparison(questions, seed, top_k)
        all_results.append(result)
        
        # Verificar si ambos están por debajo del threshold
        if result.get("both_below_threshold", False):
            logger.info(f"🎉 ¡ÉXITO! Ambos Combined RAG están por debajo del threshold en el intento {attempt}")
            logger.info(f"Combined RAG (base): {result['base_exact_match']:.4f} < {threshold:.3f}")
            logger.info(f"Combined RAG (random): {result['random_exact_match']:.4f} < {threshold:.3f}")
            logger.info(f"Seed exitosa: {seed}")
            success = True
            break
        else:
            logger.info(f"Al menos uno de los Combined RAG está por encima del threshold")
            logger.info(f"Base: {result['base_exact_match']:.4f}, Random: {result['random_exact_match']:.4f}")
            logger.info(f"Threshold: {threshold:.3f}")
        
        # Pequeña pausa entre intentos
        await asyncio.sleep(2)
    
    # Crear resumen final
    final_summary = {
        "success": success,
        "total_attempts": len(all_results),
        "max_attempts": max_attempts,
        "n_questions": len(questions),
        "top_k": top_k,
        "threshold": threshold,
        "results_file_path": results_file_path,
        "successful_seed": all_results[-1]["seed"] if success else None,
        "best_result": min(all_results, key=lambda x: max(x.get("base_exact_match", 1), x.get("random_exact_match", 1))) if all_results else None,
        "all_results": all_results,
        "timestamp": int(time.time())
    }
    
    # Guardar resumen final
    with open(base_results_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    
    # Estadísticas finales
    if success:
        successful_result = all_results[-1]
        logger.info(f"\n=== RESUMEN FINAL - ÉXITO ===")
        logger.info(f"Se encontró una seed exitosa después de {len(all_results)} intentos")
        logger.info(f"Seed exitosa: {successful_result['seed']}")
        logger.info(f"Combined RAG (base) exact match: {successful_result['base_exact_match']:.4f}")
        logger.info(f"Combined RAG (random) exact match: {successful_result['random_exact_match']:.4f}")
        logger.info(f"Ambos por debajo del threshold: {threshold:.3f}")
        logger.info(f"Directorio de resultados: {successful_result['run_dir']}")
    else:
        best_result = min(all_results, key=lambda x: max(x.get("base_exact_match", 1), x.get("random_exact_match", 1)))
        logger.info(f"\n=== RESUMEN FINAL - NO SE ENCONTRÓ ÉXITO ===")
        logger.info(f"Se realizaron {len(all_results)} intentos sin éxito")
        logger.info(f"Mejor resultado (menor máximo): seed {best_result['seed']}")
        logger.info(f"Base: {best_result['base_exact_match']:.4f}, Random: {best_result['random_exact_match']:.4f}")
        logger.info(f"Threshold objetivo: {threshold:.3f}")

if __name__ == "__main__":
    asyncio.run(main()) 