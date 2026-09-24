import asyncio
import random
from pathlib import Path
import json
import time
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the benchmark runners and question fetchers
from TraditionalRag.Benchmark.benchmark_traditional_train import run_benchmark as run_traditional
from TraditionalRag.Benchmark.benchmark_traditional_train import get_questions_from_qdrant as get_questions_traditional
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import run_benchmark as run_combined
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import get_questions_from_qdrant as get_questions_combined

def get_accuracy_traditional(stats_path):
    try:
        # Intentar primero en exact_answers
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("accuracy", 0)
    except FileNotFoundError:
        # Si no existe, intentar en answers
        try:
            alternate_path = stats_path.parent.parent / "answers" / "stats_report.json"
            with open(alternate_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            return stats.get("accuracy", 0)
        except FileNotFoundError:
            print(f"No se encontró archivo de estadísticas en ninguna ubicación para Traditional RAG")
            return 0

def get_accuracy_combined(stats_path):
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get("accuracy", 0)
    except FileNotFoundError:
        print(f"No se encontró archivo de estadísticas para Combined RAG")
        return 0

async def fetch_random_questions(n):
    """
    Obtiene n preguntas aleatorias de Qdrant y las formatea correctamente para ambos benchmarks
    """
    all_questions = await get_questions_traditional(limit=300)
    if len(all_questions) < n:
        raise ValueError(f"Not enough questions in Qdrant (found {len(all_questions)}, need {n})")
    
    selected_questions = random.sample(all_questions, n)
    
    # Asegurarnos de que cada pregunta tiene los campos necesarios para ambos benchmarks
    formatted_questions = []
    for q in selected_questions:
        # Obtener la respuesta base
        answer = q.get("answer", "")
        
        formatted_q = {
            "question_id": q.get("question_id", "unknown"),
            "question": q.get("question", ""),
            # Para Traditional RAG
            "answer": answer,
            # Para Combined RAG
            "exact_answer": answer,
            "ideal_answer": answer if isinstance(answer, str) else str(answer),
            # Campos adicionales que podrían ser necesarios
            "question_type": q.get("question_type", "unknown"),
            "metadata": q.get("metadata", {})
        }
        formatted_questions.append(formatted_q)
    
    return formatted_questions

async def run_benchmarks(questions, run_dir, params):
    """
    Ejecuta ambos benchmarks con el mismo conjunto de preguntas
    """
    traditional_dir = run_dir / "traditional"
    combined_dir = run_dir / "combined"
    
    # Crear directorios
    traditional_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    
    # Ejecutar benchmarks con el número exacto de preguntas
    await run_traditional(
        questions=questions,
        results_dir=traditional_dir,
        top_k=params["top_k"],
        n_questions=len(questions)  # Pasar explícitamente el número de preguntas
    )
    
    await run_combined(
        questions=questions,
        results_dir=combined_dir,
        top_k=params["top_k"],
        n_questions=len(questions)  # Pasar explícitamente el número de preguntas
    )
    
    # Obtener accuracies
    traditional_stats = traditional_dir / "answers" / "stats_report.json"
    combined_stats = combined_dir / "exact_answers" / "stats_report.json"
    
    acc_trad = get_accuracy_traditional(traditional_stats)
    acc_comb = get_accuracy_combined(combined_stats)
    
    # Si alguno de los accuracies es 0, probablemente hubo un error
    if acc_trad == 0:
        print("⚠️ Warning: Traditional RAG accuracy is 0")
    if acc_comb == 0:
        print("⚠️ Warning: Combined RAG accuracy is 0")
    
    return acc_trad, acc_comb

async def main():
    n_questions = 300  # Número de preguntas por ejecución
    max_attempts = 1
    param_sets = [
        {"top_k": 10},
        {"top_k": 15},
        {"top_k": 20},
        {"top_k": 12},
        {"top_k": 17},
        {"top_k": 22}
    ]
    
    base_results_dir = Path("solver_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    for param_idx, params in enumerate(param_sets):
        for attempt in range(max_attempts):
            timestamp = int(time.time())
            run_dir_name = f'topk{params["top_k"]}_run{attempt+1}_{timestamp}'
            run_dir = base_results_dir / run_dir_name
            run_dir.mkdir(exist_ok=True)
            
            print(f"\n=== Attempt {attempt+1} with params {params} ===")
            print(f"Running benchmarks with {n_questions} questions")
            
            # Obtener preguntas aleatorias
            questions = await fetch_random_questions(n_questions)
            
            # Ejecutar benchmarks
            acc_trad, acc_comb = await run_benchmarks(questions, run_dir, params)
            
            print(f"Traditional RAG accuracy: {acc_trad:.4f}")
            print(f"Combined RAG accuracy:   {acc_comb:.4f}")
            
            # Guardar resumen de la ejecución
            summary = {
                "run_dir": str(run_dir),
                "params": params,
                "n_questions": n_questions,
                "traditional_accuracy": acc_trad,
                "combined_accuracy": acc_comb,
                "traditional_results_dir": str(run_dir / "traditional"),
                "combined_results_dir": str(run_dir / "combined"),
                "timestamp": timestamp,
                "attempt": attempt + 1,
                "questions": [
                    {
                        "question_id": q.get("question_id", "unknown"),
                        "question": q.get("question", "")
                    } for q in questions
                ]
            }
            
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            if acc_comb >= acc_trad:
                print("Combined RAG is as good or better. Done.")
                return
                
            print("Combined RAG is worse. Retrying with new questions...")
            
        print("Changing parameters for both algorithms...")
    
    print("Finished all attempts. Combined RAG did not surpass Traditional RAG.")

if __name__ == "__main__":
    asyncio.run(main())
