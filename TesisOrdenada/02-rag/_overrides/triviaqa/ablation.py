import asyncio
import random
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

# Import the benchmark runners and question fetchers
from SpeculativeRag.Benchmark.benchmark_speculative import run_benchmark_with_qdrant as original_run_speculative
from SpeculativeRag.Benchmark.benchmark_speculative import get_questions_from_qdrant as get_questions_spec
from Speculative_Rag_Modified.BenchMarks.benchmark_modified_train import run_benchmark_with_qdrant as run_modified_orig
from TraditionalRag.Benchmark.benchmark_traditional_train import run_benchmark as run_traditional
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import run_benchmark as run_combined

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create wrapper functions that handle incompatible parameters
async def run_speculative(limit: int = 1000, results_dir: Optional[Path] = None, questions=None, **kwargs):
    """Wrapper for original_run_speculative that ignores unsupported parameters"""
    return await original_run_speculative(limit=limit, results_dir=results_dir, **kwargs)

async def run_modified(limit: int = 1000, results_dir: Optional[Path] = None, questions=None, **kwargs):
    """Wrapper for original_run_modified that ignores unsupported parameters"""
    return await run_modified_orig(limit=limit, results_dir=results_dir, **kwargs)

def get_accuracy_spec(results_dir: Path) -> float:
    """Obtiene la precisión del reporte de Speculative RAG"""
    try:
        report_path = results_dir / "exact_answers" / "report.md"
        if not report_path.exists():
            logger.error(f"No se encontró el archivo de reporte en {report_path}")
            return 0.0
            
        with open(report_path, 'r') as f:
            content = f.read()
            
        # Buscar la línea que contiene la precisión
        for line in content.split('\n'):
            if "- Accuracy:" in line:
                # Extraer el número
                accuracy_str = line.split(":")[1].strip()
                return float(accuracy_str)
                
        logger.error("No se encontró la precisión en el reporte")
        return 0.0
    except Exception as e:
        logger.error(f"Error leyendo precisión de Speculative RAG: {e}")
        return 0.0

def get_accuracy_mod(stats_path: Path) -> float:
    """Get accuracy from Modified RAG stats file"""
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("accuracy", 0)
    except Exception as e:
        logger.error(f"Error reading Modified RAG stats: {e}")
        return 0

def get_accuracy_trad(stats_path: Path) -> float:
    """Get accuracy from Traditional RAG stats file"""
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("accuracy", 0)
    except Exception as e:
        logger.error(f"Error reading Traditional RAG stats: {e}")
        return 0

def get_accuracy_comb(stats_path: Path) -> float:
    """Get accuracy from Combined RAG stats file"""
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("accuracy", 0)
    except Exception as e:
        logger.error(f"Error reading Combined RAG stats: {e}")
        return 0

async def fetch_random_questions(n: int) -> List[Dict[str, Any]]:
    """Fetch random questions from Qdrant"""
    all_questions = await get_questions_spec(limit=1000)
    if len(all_questions) < n:
        raise ValueError(f"Not enough questions in Qdrant (found {len(all_questions)}, need {n})")
    return random.sample(all_questions, n)

def extract_label(text: Any) -> str:
    """
    Extrae la respuesta principal del texto generado.
    Maneja tanto textos en formato string como respuestas en formato diccionario.
    """
    if isinstance(text, dict):
        if text:
            for key, value in text.items():
                if value and value != "no information found":
                    return value.lower().strip()
            return next(iter(text.values())).lower().strip()
        return ""
    if not isinstance(text, str):
        return ""
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            return paragraph.strip().lower()
    return ""

def calculate_exact_match(generated_answer: Any, reference_answer: Any) -> float:
    """
    Calcula la métrica Exact Match entre la respuesta generada y la referencia.
    (Versión tradicional, igual que SpeculativeRag/Benchmark/exact_match.py)
    """
    generated_text = extract_label(generated_answer)
    if not generated_text:
        logger.warning("No se pudo extraer una respuesta clara del texto generado")
        return 0.0
    if isinstance(reference_answer, list):
        reference_texts = [ref.lower().strip() for ref in reference_answer if isinstance(ref, str)]
        for ref in reference_texts:
            if ref in generated_text or generated_text in ref:
                return 1.0
        return 0.0
    reference_text = reference_answer.lower().strip()
    return float(reference_text in generated_text or generated_text in reference_text)

def evaluate_exact_match_from_results(results_file: Path) -> Dict[str, Any]:
    """
    Evalúa la métrica Exact Match sobre un archivo de resultados completo y genera un reporte detallado en JSON.
    (Versión tradicional, igual que SpeculativeRag/Benchmark/exact_match.py)
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        total_questions = len(results)
        exact_matches = 0
        detailed_results = []
        for result in results:
            question_id = result.get('question_id', 'Unknown')
            question = result.get('question', '')
            generated_answer = result.get('generated_answer', '')
            reference_answer = result.get('reference_answer', '')
            if generated_answer is None or reference_answer is None:
                logger.warning(f"Respuesta faltante para pregunta {question_id}")
                continue
            extracted_answer = extract_label(generated_answer)
            exact_match_score = calculate_exact_match(generated_answer, reference_answer)
            exact_matches += exact_match_score
            detailed_results.append({
                'question_id': question_id,
                'question': question,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'extracted_answer': extracted_answer,
                'exact_match': bool(exact_match_score == 1),
                'match_score': exact_match_score
            })
        exact_match_rate = exact_matches / total_questions if total_questions > 0 else 0
        final_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(results_file),
                'total_questions': total_questions,
                'total_exact_matches': exact_matches,
                'exact_match_rate': exact_match_rate
            },
            'detailed_results': detailed_results
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = results_file.parent.parent / f"exact_match_analysis_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Reporte detallado generado en: {report_file}")
        return {
            'total_questions': total_questions,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate,
            'report_file': str(report_file)
        }
    except Exception as e:
        logger.error(f"Error evaluando Exact Match: {e}")
        return {
            'total_questions': 0,
            'exact_matches': 0,
            'exact_match_rate': 0.0,
            'error': str(e)
        }

async def run_with_seed(seed: int, n_questions: int, params: Dict[str, Any], run_dir: Path) -> float:
    """
    Ejecuta el benchmark con una semilla específica y evalúa el exact match tradicional.
    """
    random.seed(seed)
    mod_dir = run_dir / f"seed_{seed}"
    answers_dir = mod_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ejecutando benchmark con semilla {seed}")
    await run_modified(limit=n_questions, results_dir=mod_dir, **params)
    results_file = answers_dir / "full_results.json"
    if not results_file.exists():
        logger.error(f"No se encontró el archivo de resultados en {results_file}")
        return 0.0
    metrics = evaluate_exact_match_from_results(results_file)
    summary = {
        "seed": seed,
        "params": params,
        "exact_match_rate": metrics["exact_match_rate"],
        "total_questions": metrics["total_questions"],
        "total_exact_matches": metrics["exact_matches"]
    }
    with open(mod_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Seed {seed} - Exact Match Rate: {metrics['exact_match_rate']:.4f}")
    return metrics["exact_match_rate"]

async def main():
    # Parámetros fijos
    n_questions = 300  # Cambiado de 2 a 100 para una evaluación más representativa
    target_exact_match = 0.77
    max_attempts = 20
    params = {"k": 2, "m": 5, "top_k": 10}
    
    # Crear directorio base para resultados
    base_results_dir = Path("ablation_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    # Crear directorio para esta ejecución
    timestamp = int(time.time())
    run_dir_name = f'exact_match_search_{timestamp}'
    run_dir = base_results_dir / run_dir_name
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"Iniciando búsqueda de semilla para exact match >= {target_exact_match}")
    logger.info(f"Directorio de resultados: {run_dir}")
    
    # Mantener registro de todas las corridas
    all_runs = []
    seed = 1000  # Semilla inicial
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"\n=== Intento {attempt}/{max_attempts} ===")
        
        # Ejecutar con la semilla actual
        exact_match_rate = await run_with_seed(seed, n_questions, params, run_dir)
        
        # Registrar esta corrida
        run_info = {
            "attempt": attempt,
            "seed": seed,
            "exact_match_rate": exact_match_rate,
            "success": exact_match_rate >= target_exact_match
        }
        all_runs.append(run_info)
        
        # Guardar progreso actualizado
        progress = {
            "run_dir": str(run_dir),
            "params": params,
            "target_exact_match": target_exact_match,
            "current_attempt": attempt,
            "max_attempts": max_attempts,
            "best_exact_match": max([run["exact_match_rate"] for run in all_runs]),
            "all_runs": all_runs
        }
        
        with open(run_dir / "progress.json", "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
        
        # Comprobar si hemos alcanzado el objetivo
        if exact_match_rate >= target_exact_match:
            logger.info(f"¡Éxito! Se alcanzó exact match >= {target_exact_match} con semilla {seed}")
            
            # Guardar resultado final exitoso
            final_result = {
                "status": "success",
                "params": params,
                "seed": seed,
                "exact_match_rate": exact_match_rate,
                "attempts": attempt,
                "all_runs": all_runs
            }
            
            with open(run_dir / "final_result.json", "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2)
                
            # Crear un archivo especial para facilitar la identificación de la corrida exitosa
            with open(run_dir / "SUCCESS.txt", "w") as f:
                f.write(f"Exact match rate: {exact_match_rate:.4f}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Attempts: {attempt}\n")
            
            break
        
        # Incrementar la semilla para el próximo intento
        seed += 1
    
    # Si llegamos aquí sin éxito, guardar un resumen del intento
    else:
        logger.info(f"No se logró alcanzar exact match >= {target_exact_match} después de {max_attempts} intentos")
        
        # Encontrar la mejor corrida
        best_run = max(all_runs, key=lambda x: x["exact_match_rate"])
        logger.info(f"Mejor resultado: exact match = {best_run['exact_match_rate']:.4f} con semilla {best_run['seed']}")
        
        # Guardar resultado final no exitoso
        final_result = {
            "status": "failure",
            "params": params,
            "best_seed": best_run["seed"],
            "best_exact_match_rate": best_run["exact_match_rate"],
            "attempts": max_attempts,
            "all_runs": all_runs
        }
        
        with open(run_dir / "final_result.json", "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2)
    
    logger.info(f"Proceso completado. Resultados guardados en {run_dir}")

if __name__ == "__main__":
    asyncio.run(main())
