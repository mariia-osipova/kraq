import asyncio
import random
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Any, Optional
import itertools
import sys
import os

# Add the current directory to the path so we can import the exact_match module
sys.path.append(str(Path(__file__).parent))

# Import the benchmark runners and question fetchers
from SpeculativeRag.Benchmark.benchmark_speculative import run_benchmark_with_qdrant as run_speculative
from SpeculativeRag.Benchmark.benchmark_speculative import get_questions_from_qdrant as get_questions_spec
from Speculative_Rag_Modified.BenchMarks.benchmark_modified_train import run_benchmark_with_qdrant as run_modified

# Import exact match evaluation
from SpeculativeRag.Benchmark.exact_match import evaluate_exact_match_from_results

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def calculate_exact_match_for_results(results_dir: Path, method_name: str) -> float:
    """Calculate exact match for results in a directory"""
    try:
        # Find the full_results.json file
        if method_name == "speculative":
            results_file = results_dir / "exact_answers" / "full_results.json"
        else:  # modified
            results_file = results_dir / "answers" / "full_results.json"
        
        if not results_file.exists():
            logger.warning(f"No se encontró el archivo de resultados en {results_file}")
            return 0.0
        
        # Calculate exact match
        metrics = evaluate_exact_match_from_results(results_file)
        return metrics.get('exact_match_rate', 0.0)
        
    except Exception as e:
        logger.error(f"Error calculating exact match for {method_name}: {e}")
        return 0.0

async def fetch_random_questions(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch random questions from Qdrant with optional seed for reproducibility"""
    if seed is not None:
        random.seed(seed)
    
    all_questions = await get_questions_spec(limit=1000)
    if len(all_questions) < n:
        raise ValueError(f"Not enough questions in Qdrant (found {len(all_questions)}, need {n})")
    return random.sample(all_questions, n)

async def run_ablation_experiment(config: Dict[str, Any], run_id: str, base_dir: Path, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run a single ablation experiment with given configuration"""
    
    # Create run directory
    run_dir = base_dir / run_id
    spec_dir = run_dir / "speculative"
    mod_dir = run_dir / "modified"
    
    # Create directories
    (spec_dir / "exact_answers").mkdir(parents=True, exist_ok=True)
    (spec_dir / "ideal_answers").mkdir(parents=True, exist_ok=True)
    (mod_dir / "answers").mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Running experiment {run_id} with config: {config}")
        
        # Run Speculative RAG
        start_time = time.time()
        await run_speculative(
            limit=len(questions), 
            questions=questions, 
            results_dir=spec_dir, 
            **config
        )
        spec_time = time.time() - start_time
        
        # Run Modified RAG
        start_time = time.time()
        await run_modified(
            limit=len(questions), 
            questions=questions, 
            results_dir=mod_dir, 
            **config
        )
        mod_time = time.time() - start_time
        
        # Get LLM-based accuracies
        acc_spec = get_accuracy_spec(spec_dir)
        acc_mod = get_accuracy_mod(mod_dir / "answers" / "stats_report.json")
        
        # Calculate exact match scores
        logger.info(f"Calculating exact match scores for {run_id}...")
        exact_match_spec = calculate_exact_match_for_results(spec_dir, "speculative")
        exact_match_mod = calculate_exact_match_for_results(mod_dir, "modified")
        
        # Create result summary
        result = {
            "run_id": run_id,
            "config": config,
            "llm_accuracy": {
                "speculative": acc_spec,
                "modified": acc_mod,
                "difference": acc_mod - acc_spec
            },
            "exact_match": {
                "speculative": exact_match_spec,
                "modified": exact_match_mod,
                "difference": exact_match_mod - exact_match_spec
            },
            "execution_time": {
                "speculative": spec_time,
                "modified": mod_time,
                "difference": mod_time - spec_time
            },
            "directories": {
                "speculative": str(spec_dir),
                "modified": str(mod_dir)
            },
            "timestamp": int(time.time())
        }
        
        # Save individual result
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Experiment {run_id} completed:")
        logger.info(f"  Speculative - LLM Acc: {acc_spec:.4f}, Exact Match: {exact_match_spec:.4f} (time: {spec_time:.2f}s)")
        logger.info(f"  Modified    - LLM Acc: {acc_mod:.4f}, Exact Match: {exact_match_mod:.4f} (time: {mod_time:.2f}s)")
        logger.info(f"  Difference  - LLM Acc: {acc_mod - acc_spec:.4f}, Exact Match: {exact_match_mod - exact_match_spec:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in experiment {run_id}: {e}")
        error_result = {
            "run_id": run_id,
            "config": config,
            "error": str(e),
            "timestamp": int(time.time())
        }
        
        with open(run_dir / "error.json", "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
        
        return error_result

async def main():
    """Main ablation study function"""
    
    # Configuration parameters to test
    k_values = [3, 4, 5, 6]
    m_values = [8]
    top_k_values = [10, 15, 20, 25]
    
    # Number of questions per experiment
    n_questions = 300
    
    # Random seed for reproducibility
    random_seed = 42
    
    # Create base results directory
    base_results_dir = Path("ablation_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(k_values, m_values, top_k_values))
    total_experiments = len(param_combinations)
    
    logger.info(f"Starting ablation study with {total_experiments} experiments")
    logger.info(f"Testing k={k_values}, m={m_values}, top_k={top_k_values}")
    logger.info(f"Using {n_questions} questions per experiment")
    
    # Fetch questions once for all experiments (with seed for reproducibility)
    questions = await fetch_random_questions(n_questions, seed=random_seed)
    logger.info(f"Fetched {len(questions)} questions for experiments")
    
    # Store all results
    all_results = []
    failed_experiments = []
    
    # Run experiments
    for i, (k, m, top_k) in enumerate(param_combinations, 1):
        config = {"k": k, "m": m, "top_k": top_k}
        run_id = f"k{k}_m{m}_topk{top_k}_{int(time.time())}"
        
        logger.info(f"\n=== Experiment {i}/{total_experiments}: {run_id} ===")
        
        result = await run_ablation_experiment(config, run_id, base_results_dir, questions)
        
        if "error" in result:
            failed_experiments.append(result)
        else:
            all_results.append(result)
    
    # Generate final analysis
    logger.info("\n=== Generating final analysis ===")
    
    if all_results:
        # Sort results by different metrics
        sorted_by_llm_mod = sorted(all_results, key=lambda x: x["llm_accuracy"]["modified"], reverse=True)
        sorted_by_llm_spec = sorted(all_results, key=lambda x: x["llm_accuracy"]["speculative"], reverse=True)
        sorted_by_llm_diff = sorted(all_results, key=lambda x: x["llm_accuracy"]["difference"], reverse=True)
        
        sorted_by_em_mod = sorted(all_results, key=lambda x: x["exact_match"]["modified"], reverse=True)
        sorted_by_em_spec = sorted(all_results, key=lambda x: x["exact_match"]["speculative"], reverse=True)
        sorted_by_em_diff = sorted(all_results, key=lambda x: x["exact_match"]["difference"], reverse=True)
        
        # Find best configurations
        best_llm_mod = sorted_by_llm_mod[0]
        best_llm_spec = sorted_by_llm_spec[0]
        best_llm_diff = sorted_by_llm_diff[0]
        
        best_em_mod = sorted_by_em_mod[0]
        best_em_spec = sorted_by_em_spec[0]
        best_em_diff = sorted_by_em_diff[0]
        
        # Calculate statistics
        avg_llm_spec = sum(r["llm_accuracy"]["speculative"] for r in all_results) / len(all_results)
        avg_llm_mod = sum(r["llm_accuracy"]["modified"] for r in all_results) / len(all_results)
        avg_llm_diff = sum(r["llm_accuracy"]["difference"] for r in all_results) / len(all_results)
        
        avg_em_spec = sum(r["exact_match"]["speculative"] for r in all_results) / len(all_results)
        avg_em_mod = sum(r["exact_match"]["modified"] for r in all_results) / len(all_results)
        avg_em_diff = sum(r["exact_match"]["difference"] for r in all_results) / len(all_results)
        
        analysis = {
            "experiment_summary": {
                "total_experiments": total_experiments,
                "successful_experiments": len(all_results),
                "failed_experiments": len(failed_experiments),
                "questions_per_experiment": n_questions,
                "random_seed": random_seed
            },
            "parameter_ranges": {
                "k_values": k_values,
                "m_values": m_values,
                "top_k_values": top_k_values
            },
            "overall_statistics": {
                "llm_accuracy": {
                    "average_speculative": avg_llm_spec,
                    "average_modified": avg_llm_mod,
                    "average_difference": avg_llm_diff,
                    "modified_wins": sum(1 for r in all_results if r["llm_accuracy"]["difference"] > 0),
                    "speculative_wins": sum(1 for r in all_results if r["llm_accuracy"]["difference"] < 0),
                    "ties": sum(1 for r in all_results if r["llm_accuracy"]["difference"] == 0)
                },
                "exact_match": {
                    "average_speculative": avg_em_spec,
                    "average_modified": avg_em_mod,
                    "average_difference": avg_em_diff,
                    "modified_wins": sum(1 for r in all_results if r["exact_match"]["difference"] > 0),
                    "speculative_wins": sum(1 for r in all_results if r["exact_match"]["difference"] < 0),
                    "ties": sum(1 for r in all_results if r["exact_match"]["difference"] == 0)
                }
            },
            "best_configurations": {
                "llm_accuracy": {
                    "best_modified": {
                        "config": best_llm_mod["config"],
                        "accuracy": best_llm_mod["llm_accuracy"]["modified"],
                        "run_id": best_llm_mod["run_id"]
                    },
                    "best_speculative": {
                        "config": best_llm_spec["config"],
                        "accuracy": best_llm_spec["llm_accuracy"]["speculative"],
                        "run_id": best_llm_spec["run_id"]
                    },
                    "best_difference": {
                        "config": best_llm_diff["config"],
                        "difference": best_llm_diff["llm_accuracy"]["difference"],
                        "run_id": best_llm_diff["run_id"]
                    }
                },
                "exact_match": {
                    "best_modified": {
                        "config": best_em_mod["config"],
                        "accuracy": best_em_mod["exact_match"]["modified"],
                        "run_id": best_em_mod["run_id"]
                    },
                    "best_speculative": {
                        "config": best_em_spec["config"],
                        "accuracy": best_em_spec["exact_match"]["speculative"],
                        "run_id": best_em_spec["run_id"]
                    },
                    "best_difference": {
                        "config": best_em_diff["config"],
                        "difference": best_em_diff["exact_match"]["difference"],
                        "run_id": best_em_diff["run_id"]
                    }
                }
            },
            "all_results": all_results,
            "failed_experiments": failed_experiments,
            "timestamp": int(time.time())
        }
    else:
        analysis = {
            "experiment_summary": {
                "total_experiments": total_experiments,
                "successful_experiments": 0,
                "failed_experiments": len(failed_experiments)
            },
            "error": "No experiments completed successfully",
            "failed_experiments": failed_experiments,
            "timestamp": int(time.time())
        }
    
    # Save analysis
    analysis_file = base_results_dir / "ablation_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Generate summary report
    report_lines = [
        "# Ablation Study Results",
        "",
        f"**Total Experiments:** {total_experiments}",
        f"**Successful:** {len(all_results)}",
        f"**Failed:** {len(failed_experiments)}",
        f"**Questions per experiment:** {n_questions}",
        ""
    ]
    
    if all_results:
        report_lines.extend([
            "## Overall Statistics",
            "",
            "### LLM Accuracy",
            f"- Average Speculative: {avg_llm_spec:.4f}",
            f"- Average Modified: {avg_llm_mod:.4f}",
            f"- Average Difference (Mod - Spec): {avg_llm_diff:.4f}",
            f"- Modified wins: {sum(1 for r in all_results if r['llm_accuracy']['difference'] > 0)}",
            f"- Speculative wins: {sum(1 for r in all_results if r['llm_accuracy']['difference'] < 0)}",
            f"- Ties: {sum(1 for r in all_results if r['llm_accuracy']['difference'] == 0)}",
            "",
            "### Exact Match",
            f"- Average Speculative: {avg_em_spec:.4f}",
            f"- Average Modified: {avg_em_mod:.4f}",
            f"- Average Difference (Mod - Spec): {avg_em_diff:.4f}",
            f"- Modified wins: {sum(1 for r in all_results if r['exact_match']['difference'] > 0)}",
            f"- Speculative wins: {sum(1 for r in all_results if r['exact_match']['difference'] < 0)}",
            f"- Ties: {sum(1 for r in all_results if r['exact_match']['difference'] == 0)}",
            "",
            "## Best Configurations",
            "",
            "### LLM Accuracy",
            f"**Best Modified:** {best_llm_mod['llm_accuracy']['modified']:.4f}",
            f"- Config: k={best_llm_mod['config']['k']}, m={best_llm_mod['config']['m']}, top_k={best_llm_mod['config']['top_k']}",
            "",
            f"**Best Speculative:** {best_llm_spec['llm_accuracy']['speculative']:.4f}",
            f"- Config: k={best_llm_spec['config']['k']}, m={best_llm_spec['config']['m']}, top_k={best_llm_spec['config']['top_k']}",
            "",
            "### Exact Match",
            f"**Best Modified:** {best_em_mod['exact_match']['modified']:.4f}",
            f"- Config: k={best_em_mod['config']['k']}, m={best_em_mod['config']['m']}, top_k={best_em_mod['config']['top_k']}",
            "",
            f"**Best Speculative:** {best_em_spec['exact_match']['speculative']:.4f}",
            f"- Config: k={best_em_spec['config']['k']}, m={best_em_spec['config']['m']}, top_k={best_em_spec['config']['top_k']}",
            ""
        ])
    
    report_file = base_results_dir / "ablation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"\nAblation study completed!")
    logger.info(f"Results saved to: {base_results_dir}")
    logger.info(f"Analysis file: {analysis_file}")
    logger.info(f"Report file: {report_file}")
    
    if all_results:
        logger.info(f"\nBest results:")
        logger.info(f"LLM - Modified: {best_llm_mod['llm_accuracy']['modified']:.4f}, Speculative: {best_llm_spec['llm_accuracy']['speculative']:.4f}")
        logger.info(f"Exact Match - Modified: {best_em_mod['exact_match']['modified']:.4f}, Speculative: {best_em_spec['exact_match']['speculative']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
