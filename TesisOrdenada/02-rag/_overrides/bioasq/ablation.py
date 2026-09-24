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
from TraditionalRag.Benchmark.benchmark_traditional_train import get_questions_from_qdrant as get_questions_traditional
from TraditionalRag.Benchmark.benchmark_traditional_train import generate_traditional_rag_response, evaluate_question, generate_stats_report
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import get_questions_from_qdrant as get_questions_combined
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import generate_traditional_rag_response as generate_combined_response
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import evaluate_question as evaluate_combined_question
from TraditionalRag_CombinedRetrieve.Benchmark.benchmark_traditional_combined_question import generate_stats_report as generate_combined_stats_report

from openai import AsyncOpenAI
from tqdm import tqdm

# Configuración para los modelos
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"

def get_accuracies(stats_dir):
    """
    Obtiene las accuracies tanto de exact como de ideal answers de un directorio
    
    Args:
        stats_dir: Directorio que contiene las carpetas exact_answers e ideal_answers
        
    Returns:
        Tuple[float, float]: (exact_accuracy, ideal_accuracy)
    """
    exact_acc = 0.0
    ideal_acc = 0.0
    
    try:
        # Leer exact answers
        exact_report = stats_dir / "exact_answers" / "stats_report.json"
        if exact_report.exists():
            with open(exact_report, "r", encoding="utf-8") as f:
                stats = json.load(f)
                exact_acc = stats.get("accuracy", 0.0)
        else:
            print(f"No se encontró archivo de estadísticas para exact answers en {exact_report}")
        
        # Leer ideal answers
        ideal_report = stats_dir / "ideal_answers" / "stats_report.json"
        if ideal_report.exists():
            with open(ideal_report, "r", encoding="utf-8") as f:
                stats = json.load(f)
                ideal_acc = stats.get("accuracy", 0.0)
        else:
            print(f"No se encontró archivo de estadísticas para ideal answers en {ideal_report}")
                
    except Exception as e:
        print(f"Error leyendo accuracies: {e}")
        
    return exact_acc, ideal_acc

async def fetch_random_questions(n):
    """
    Obtiene n preguntas aleatorias de Qdrant y las formatea correctamente para ambos benchmarks
    """
    all_questions = await get_questions_traditional(limit=1000)
    if len(all_questions) < n:
        raise ValueError(f"Not enough questions in Qdrant (found {len(all_questions)}, need {n})")
    
    selected_questions = random.sample(all_questions, n)
    
    # Asegurarnos de que cada pregunta tiene los campos necesarios para ambos benchmarks
    formatted_questions = []
    for q in selected_questions:
        # Obtener la respuesta exacta e ideal
        exact_answer = q.get("exact_answer", "")
        ideal_answer = q.get("ideal_answer", "")
        
        formatted_q = {
            "question_id": q.get("question_id", "unknown"),
            "question": q.get("question", ""),
            "exact_answer": exact_answer,
            "ideal_answer": ideal_answer,
            "question_type": q.get("question_type", "unknown"),
            "metadata": q.get("metadata", {})
        }
        formatted_questions.append(formatted_q)
    
    return formatted_questions

async def run_traditional_benchmark(questions, results_dir, top_k, n_questions=None):
    """
    Implementación personalizada del benchmark para Traditional RAG
    
    Args:
        questions: Lista de preguntas para evaluar
        results_dir: Directorio donde guardar resultados
        top_k: Número de documentos a recuperar
        n_questions: Número opcional de preguntas a procesar
    """
    # Crear subdirectorios para cada tipo de respuesta
    ideal_dir = results_dir / "ideal_answers"
    exact_dir = results_dir / "exact_answers"
    ideal_dir.mkdir(exist_ok=True)
    exact_dir.mkdir(exist_ok=True)
    
    # Inicializar listas para resultados
    ideal_results = []
    exact_results = []
    
    # Limitar número de preguntas si es necesario
    if n_questions and n_questions < len(questions):
        questions = questions[:n_questions]
    
    # Inicializar clientes para LLM (vLLM) y embeddings (Ollama)
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    ollama_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    # Procesar cada pregunta
    for i, question_data in enumerate(tqdm(questions, desc="Evaluando Traditional RAG")):
        question = question_data["question"]
        question_id = question_data.get("question_id", "unknown")
        
        print(f"Procesando pregunta {i+1}/{len(questions)} (ID: {question_id}): {question[:50]}...")
        
        # Generar la respuesta una sola vez con Traditional RAG
        response, total_time, retrieval_time = await generate_traditional_rag_response(
            question=question,
            llm_client=vllm_client,
            embeddings_client=ollama_client
        )
        
        # Evaluar contra la respuesta ideal si está disponible
        if "ideal_answer" in question_data:
            ideal_result = await evaluate_question(
                question_data=question_data,
                llm_client=vllm_client,
                response=response,
                total_time=total_time,
                retrieval_time=retrieval_time,
                answer_type="ideal_answer"
            )
            
            if ideal_result:
                ideal_results.append(ideal_result)
                
                # Guardar resultados parciales
                with open(ideal_dir / "partial_results.json", "w", encoding="utf-8") as f:
                    json.dump(ideal_results, f, indent=2, ensure_ascii=False)
        
        # Evaluar contra las respuestas exactas
        if "exact_answer" in question_data:
            exact_result = await evaluate_question(
                question_data=question_data,
                llm_client=vllm_client,
                response=response,
                total_time=total_time,
                retrieval_time=retrieval_time,
                answer_type="exact_answer"
            )
            
            if exact_result:
                exact_results.append(exact_result)
                
                # Guardar resultados parciales
                with open(exact_dir / "partial_results.json", "w", encoding="utf-8") as f:
                    json.dump(exact_results, f, indent=2, ensure_ascii=False)
    
    # Guardar todos los resultados
    with open(ideal_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(ideal_results, f, indent=2, ensure_ascii=False)
    
    with open(exact_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(exact_results, f, indent=2, ensure_ascii=False)
    
    # Calcular estadísticas para cada tipo de respuesta
    await generate_stats_report(ideal_results, ideal_dir, "Ideal Answers")
    await generate_stats_report(exact_results, exact_dir, "Exact Answers")
    
    print(f"Benchmark de Traditional RAG completado. Resultados guardados en {results_dir}")

async def run_combined_benchmark(questions, results_dir, top_k, n_questions=None):
    """
    Implementación personalizada del benchmark para Combined RAG
    
    Args:
        questions: Lista de preguntas para evaluar
        results_dir: Directorio donde guardar resultados
        top_k: Número de documentos a recuperar
        n_questions: Número opcional de preguntas a procesar
    """
    # Crear subdirectorios para cada tipo de respuesta
    ideal_dir = results_dir / "ideal_answers"
    exact_dir = results_dir / "exact_answers"
    ideal_dir.mkdir(exist_ok=True)
    exact_dir.mkdir(exist_ok=True)
    
    # Inicializar listas para resultados
    ideal_results = []
    exact_results = []
    
    # Limitar número de preguntas si es necesario
    if n_questions and n_questions < len(questions):
        questions = questions[:n_questions]
    
    # Inicializar clientes para LLM (vLLM) y embeddings (Ollama)
    vllm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    ollama_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    # Procesar cada pregunta
    for i, question_data in enumerate(tqdm(questions, desc="Evaluando Combined RAG")):
        question = question_data["question"]
        question_id = question_data.get("question_id", "unknown")
        
        print(f"Procesando pregunta {i+1}/{len(questions)} (ID: {question_id}): {question[:50]}...")
        
        # Generar la respuesta una sola vez con Combined RAG
        response, total_time, retrieval_time = await generate_combined_response(
            question=question,
            llm_client=vllm_client,
            embeddings_client=ollama_client
        )
        
        # Evaluar contra la respuesta ideal si está disponible
        if "ideal_answer" in question_data:
            ideal_result = await evaluate_combined_question(
                question_data=question_data,
                llm_client=vllm_client,
                response=response,
                total_time=total_time,
                retrieval_time=retrieval_time,
                answer_type="ideal_answer"
            )
            
            if ideal_result:
                ideal_results.append(ideal_result)
                
                # Guardar resultados parciales
                with open(ideal_dir / "partial_results.json", "w", encoding="utf-8") as f:
                    json.dump(ideal_results, f, indent=2, ensure_ascii=False)
        
        # Evaluar contra las respuestas exactas
        if "exact_answer" in question_data:
            exact_result = await evaluate_combined_question(
                question_data=question_data,
                llm_client=vllm_client,
                response=response,
                total_time=total_time,
                retrieval_time=retrieval_time,
                answer_type="exact_answer"
            )
            
            if exact_result:
                exact_results.append(exact_result)
                
                # Guardar resultados parciales
                with open(exact_dir / "partial_results.json", "w", encoding="utf-8") as f:
                    json.dump(exact_results, f, indent=2, ensure_ascii=False)
    
    # Guardar todos los resultados
    with open(ideal_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(ideal_results, f, indent=2, ensure_ascii=False)
    
    with open(exact_dir / "full_results.json", "w", encoding="utf-8") as f:
        json.dump(exact_results, f, indent=2, ensure_ascii=False)
    
    # Calcular estadísticas para cada tipo de respuesta
    await generate_combined_stats_report(ideal_results, ideal_dir, "Ideal Answers")
    await generate_combined_stats_report(exact_results, exact_dir, "Exact Answers")
    
    print(f"Benchmark de Combined RAG completado. Resultados guardados en {results_dir}")

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
    await run_traditional_benchmark(
        questions=questions,
        results_dir=traditional_dir,
        top_k=params["top_k"],
        n_questions=len(questions)
    )
    
    await run_combined_benchmark(
        questions=questions,
        results_dir=combined_dir,
        top_k=params["top_k"],
        n_questions=len(questions)
    )
    
    # Obtener accuracies para traditional RAG (exact e ideal)
    trad_exact_acc, trad_ideal_acc = get_accuracies(traditional_dir)
    
    # Obtener accuracies para combined RAG (exact e ideal)
    comb_exact_acc, comb_ideal_acc = get_accuracies(combined_dir)
    
    # Si alguno de los accuracies es 0, probablemente hubo un error
    if trad_exact_acc == 0:
        print("⚠️ Warning: Traditional RAG exact accuracy is 0")
    if trad_ideal_acc == 0:
        print("⚠️ Warning: Traditional RAG ideal accuracy is 0")
    if comb_exact_acc == 0:
        print("⚠️ Warning: Combined RAG exact accuracy is 0")
    if comb_ideal_acc == 0:
        print("⚠️ Warning: Combined RAG ideal accuracy is 0")
    
    return trad_exact_acc, trad_ideal_acc, comb_exact_acc, comb_ideal_acc

async def main():
    n_questions = 500  # Número de preguntas por ejecución
    max_attempts = 1
    param_sets = [
        {"top_k": 5},
        {"top_k": 8},
        {"top_k": 10},
        {"top_k": 13},
        {"top_k": 15},
        {"top_k": 17},
        {"top_k": 20}
    ]
    
    base_results_dir = Path("ablation_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    # Almacenar los mejores resultados
    best_results = {
        "params": None,
        "traditional_exact_acc": 0.0,
        "traditional_ideal_acc": 0.0,
        "combined_exact_acc": 0.0,
        "combined_ideal_acc": 0.0,
        "diff_exact": -float('inf'),  # Diferencia combinado - tradicional
        "diff_ideal": -float('inf')   # Diferencia combinado - tradicional
    }
    
    for param_idx, params in enumerate(param_sets):
        print(f"\n=== Evaluando conjunto de parámetros {param_idx+1}/{len(param_sets)} ===")
        print(f"Parámetros: {params}")
        
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
            trad_exact_acc, trad_ideal_acc, comb_exact_acc, comb_ideal_acc = await run_benchmarks(questions, run_dir, params)
            
            print(f"Traditional RAG exact accuracy:   {trad_exact_acc:.4f}")
            print(f"Traditional RAG ideal accuracy:   {trad_ideal_acc:.4f}")
            print(f"Combined RAG exact accuracy:      {comb_exact_acc:.4f}")
            print(f"Combined RAG ideal accuracy:      {comb_ideal_acc:.4f}")
            
            # Calcular diferencias para comparación
            diff_exact = comb_exact_acc - trad_exact_acc
            diff_ideal = comb_ideal_acc - trad_ideal_acc
            
            print(f"Difference in exact accuracy:     {diff_exact:.4f}")
            print(f"Difference in ideal accuracy:     {diff_ideal:.4f}")
            
            # Actualizar mejor resultado si hay mejora en la diferencia combinada
            combined_diff = diff_exact + diff_ideal
            best_combined_diff = best_results["diff_exact"] + best_results["diff_ideal"]
            
            if combined_diff > best_combined_diff:
                best_results["params"] = params.copy()
                best_results["traditional_exact_acc"] = trad_exact_acc
                best_results["traditional_ideal_acc"] = trad_ideal_acc
                best_results["combined_exact_acc"] = comb_exact_acc
                best_results["combined_ideal_acc"] = comb_ideal_acc
                best_results["diff_exact"] = diff_exact
                best_results["diff_ideal"] = diff_ideal
                print(f"✅ New best result found with params {params}")
            
            # Guardar resumen de la ejecución
            summary = {
                "run_dir": str(run_dir),
                "params": params,
                "n_questions": n_questions,
                "traditional_exact_accuracy": trad_exact_acc,
                "traditional_ideal_accuracy": trad_ideal_acc,
                "combined_exact_accuracy": comb_exact_acc,
                "combined_ideal_accuracy": comb_ideal_acc,
                "diff_exact": diff_exact,
                "diff_ideal": diff_ideal,
                "combined_diff": combined_diff,
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
            
            # Evaluar si Combined RAG supera a Traditional RAG
            if comb_exact_acc >= trad_exact_acc and comb_ideal_acc >= trad_ideal_acc:
                print("✅ Combined RAG is as good or better in both exact and ideal metrics.")
            else:
                print("⚠️ Combined RAG is worse in at least one metric.")
    
    # Al final, mostrar el mejor resultado encontrado
    print("\n=== RESULTS SUMMARY ===")
    print(f"Best parameters: {best_results['params']}")
    print(f"With these parameters:")
    print(f"Traditional RAG exact accuracy:   {best_results['traditional_exact_acc']:.4f}")
    print(f"Traditional RAG ideal accuracy:   {best_results['traditional_ideal_acc']:.4f}")
    print(f"Combined RAG exact accuracy:      {best_results['combined_exact_acc']:.4f}")
    print(f"Combined RAG ideal accuracy:      {best_results['combined_ideal_acc']:.4f}")
    print(f"Difference in exact accuracy:     {best_results['diff_exact']:.4f}")
    print(f"Difference in ideal accuracy:     {best_results['diff_ideal']:.4f}")
    
    # Guardar resumen final
    final_summary = {
        "timestamp": int(time.time()),
        "best_params": best_results["params"],
        "accuracies": {
            "traditional": {
                "exact": best_results["traditional_exact_acc"],
                "ideal": best_results["traditional_ideal_acc"]
            },
            "combined": {
                "exact": best_results["combined_exact_acc"],
                "ideal": best_results["combined_ideal_acc"]
            }
        },
        "diffs": {
            "exact": best_results["diff_exact"],
            "ideal": best_results["diff_ideal"],
            "combined": best_results["diff_exact"] + best_results["diff_ideal"]
        },
        "successful": best_results["diff_exact"] >= 0 and best_results["diff_ideal"] >= 0
    }
    
    with open(base_results_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    
    print("Finished all parameter sets. See final_summary.json for best results.")

if __name__ == "__main__":
    asyncio.run(main())
