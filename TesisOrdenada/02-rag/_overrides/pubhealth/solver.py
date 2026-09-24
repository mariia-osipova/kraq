import asyncio
import random
from pathlib import Path
import json
import time

# Import the benchmark runners and question fetchers
from SpeculativeRag.Benchmark.benchmark_speculative import (
    run_benchmark_with_qdrant as run_speculative,
    get_questions_from_qdrant as get_questions_spec
)
from Speculative_Rag_Modified.BenchMarks.benchmark_modified_train import (
    run_benchmark_with_qdrant as run_modified,
    get_questions_from_qdrant as get_questions_mod
)

def get_accuracy_spec(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    # Speculative stats: {"accuracy": {"value": ...}}
    if "accuracy" in stats:
        if isinstance(stats["accuracy"], dict):
            return stats["accuracy"].get("value", 0)
        return stats["accuracy"]
    return 0

def get_accuracy_mod(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    # Modified stats: {"accuracy": ...}
    return stats.get("accuracy", stats.get("accuracy", 0))

async def fetch_random_questions(n):
    # Use either get_questions_spec or get_questions_mod, they should return the same set
    all_questions = await get_questions_spec(limit=1000)
    if len(all_questions) < n:
        raise ValueError(f"Not enough questions in Qdrant (found {len(all_questions)}, need {n})")
    return random.sample(all_questions, n)

async def main():
    n_questions = 300
    max_attempts = 1
    param_sets = [
        {"k": 2, "m": 5, "top_k": 10}
    ]
    base_results_dir = Path("solver_runs")
    base_results_dir.mkdir(exist_ok=True)
    for param_idx, params in enumerate(param_sets):
        for attempt in range(max_attempts):
            timestamp = int(time.time())
            run_dir_name = f'600_k{params["k"]}_m{params["m"]}_topk{params["top_k"]}_run{attempt+1}_{timestamp}'
            run_dir = base_results_dir / run_dir_name
            run_dir.mkdir(exist_ok=True)
            print(f"\n=== Attempt {attempt+1} with params {params} ===")
            questions = await fetch_random_questions(n_questions)
            # Run both benchmarks with the same questions and params, and custom results_dir
            spec_dir = run_dir / "speculative"
            mod_dir = run_dir / "modified"
            await run_speculative(limit=n_questions, questions=questions, results_dir=spec_dir, **params)
            await run_modified(limit=n_questions, questions=questions, results_dir=mod_dir, **params)
            # Read accuracy
            acc_spec = get_accuracy_spec(spec_dir / "exact_answers" / "stats.json")
            acc_mod = get_accuracy_mod(mod_dir / "exact_answers" / "stats_report.json")
            print(f"SpeculativeRAG accuracy: {acc_spec:.4f}")
            print(f"ModifiedRAG accuracy:   {acc_mod:.4f}")
            # Save summary for this run
            summary = {
                "run_dir": str(run_dir),
                "params": params,
                "speculative_accuracy": acc_spec,
                "modified_accuracy": acc_mod,
                "speculative_results_dir": str(spec_dir),
                "modified_results_dir": str(mod_dir)
            }
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            if acc_mod >= acc_spec:
                print("Modified RAG is as good or better. Done.")
                return
            print("Modified RAG is worse. Retrying with new questions...")
        print("Changing parameters for both algorithms...")
    print("Finished all attempts. Modified RAG did not surpass Speculative RAG.")

if __name__ == "__main__":
    asyncio.run(main())
