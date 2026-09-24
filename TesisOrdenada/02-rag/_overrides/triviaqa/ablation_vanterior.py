import asyncio
import random
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Import the benchmark runners and question fetchers
from SpeculativeRag.Benchmark.benchmark_speculative import run_benchmark_with_qdrant as original_run_speculative
from SpeculativeRag.Benchmark.benchmark_speculative import get_questions_from_qdrant as get_questions_spec
from Speculative_Rag_Modified.BenchMarks.benchmark_modified_train import run_benchmark_with_qdrant as original_run_modified
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
    return await original_run_modified(limit=limit, results_dir=results_dir, **kwargs)

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

async def main():
    n_questions = 300
    max_attempts = 1
    
    # Parameter sets to try - vamos a probar todos estos conjuntos
    speculative_param_sets = [
        {"k": 2, "m": 5, "top_k": 10},
        {"k": 3, "m": 5, "top_k": 10},
        {"k": 4, "m": 5, "top_k": 10},
        {"k": 2, "m": 5, "top_k": 15},
        {"k": 2, "m": 5, "top_k": 20},
    ]
    
    traditional_param_sets = [
    ]
    
    # Create results directory
    base_results_dir = Path("ablation_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    # Resultados para almacenar todas las ejecuciones
    all_results = []
    
    # Ejecutar todos los conjuntos de parámetros para Speculative vs Modified
    for param_idx, params in enumerate(speculative_param_sets):
        logger.info(f"\n=== Running Speculative vs Modified comparison {param_idx+1}/{len(speculative_param_sets)} ===")
        
        timestamp = int(time.time())
        run_dir_name = f'spec_k{params["k"]}_m{params["m"]}_topk{params["top_k"]}_{timestamp}'
        run_dir = base_results_dir / run_dir_name
        
        # Crear directorios y ejecutar comparación
        spec_dir = run_dir / "speculative"
        mod_dir = run_dir / "modified"
        (spec_dir / "exact_answers").mkdir(parents=True, exist_ok=True)
        (spec_dir / "ideal_answers").mkdir(parents=True, exist_ok=True)
        (mod_dir / "answers").mkdir(parents=True, exist_ok=True)
        
        try:
            questions = await fetch_random_questions(n_questions)
            
            # Use our wrapper function to handle incompatible parameters
            await run_speculative(limit=n_questions, results_dir=spec_dir, **params)
            await run_modified(limit=n_questions, results_dir=mod_dir, **params)
            
            acc_spec = get_accuracy_spec(spec_dir)
            acc_mod = get_accuracy_mod(mod_dir / "answers" / "stats_report.json")
            
            logger.info(f"SpeculativeRAG accuracy: {acc_spec:.4f}")
            logger.info(f"ModifiedRAG accuracy:   {acc_mod:.4f}")
            
            result = {
                "run_dir": str(run_dir),
                "params": params,
                "speculative_accuracy": acc_spec,
                "modified_accuracy": acc_mod,
                "speculative_results_dir": str(spec_dir),
                "modified_results_dir": str(mod_dir)
            }
            
            # Almacenar el resultado en la lista de resultados
            all_results.append(result)
            
            # Guardar resultado individual
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            # Registrar la comparación - solo para información
            if acc_mod >= acc_spec:
                logger.info("Modified RAG accuracy is equal or better than Speculative RAG")
            else:
                logger.info("Speculative RAG accuracy is better than Modified RAG")
                
        except Exception as e:
            logger.error(f"Error in Speculative comparison with parameters {params}: {e}")
    
    # Ejecutar todos los conjuntos de parámetros para Traditional vs Combined
    for param_idx, params in enumerate(traditional_param_sets):
        logger.info(f"\n=== Running Traditional vs Combined comparison {param_idx+1}/{len(traditional_param_sets)} ===")
        
        timestamp = int(time.time())
        run_dir_name = f'trad_topk{params["top_k"]}_{timestamp}'
        run_dir = base_results_dir / run_dir_name
        
        # Crear directorios y ejecutar comparación
        trad_dir = run_dir / "traditional"
        comb_dir = run_dir / "combined"
        (trad_dir / "answers").mkdir(parents=True, exist_ok=True)
        (comb_dir / "exact_answers").mkdir(parents=True, exist_ok=True)
        
        try:
            questions = await fetch_random_questions(n_questions)
            await run_traditional(limit=n_questions, questions=questions, results_dir=trad_dir, **params)
            await run_combined(limit=n_questions, questions=questions, results_dir=comb_dir, **params)
            
            acc_trad = get_accuracy_trad(trad_dir / "answers" / "stats_report.json")
            acc_comb = get_accuracy_comb(comb_dir / "exact_answers" / "stats_report.json")
            
            logger.info(f"TraditionalRAG accuracy: {acc_trad:.4f}")
            logger.info(f"CombinedRAG accuracy:    {acc_comb:.4f}")
            
            result = {
                "run_dir": str(run_dir),
                "params": params,
                "traditional_accuracy": acc_trad,
                "combined_accuracy": acc_comb,
                "traditional_results_dir": str(trad_dir),
                "combined_results_dir": str(comb_dir)
            }
            
            # Almacenar el resultado en la lista de resultados
            all_results.append(result)
            
            # Guardar resultado individual
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            # Registrar la comparación - solo para información
            if acc_comb >= acc_trad:
                logger.info("Combined RAG accuracy is equal or better than Traditional RAG")
            else:
                logger.info("Traditional RAG accuracy is better than Combined RAG")
                
        except Exception as e:
            logger.error(f"Error in Traditional comparison with parameters {params}: {e}")
    
    # Guardar resumen con todos los resultados
    summary = {
        "all_results": all_results,
        "total_spec_tests": len(speculative_param_sets),
        "total_trad_tests": len(traditional_param_sets),
        "timestamp": int(time.time())
    }
    
    with open(base_results_dir / "all_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Completed all parameter tests. Results saved to {base_results_dir}")

if __name__ == "__main__":
    asyncio.run(main())
