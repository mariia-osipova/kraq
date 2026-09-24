import asyncio
import json
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from loguru import logger
import glob
from tqdm.asyncio import tqdm

# Añadir el directorio raíz del proyecto a sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configurar logger
logger.add("re_evaluation_parallel_logs.log", rotation="500 MB")

# Configuración
VLLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MAX_CONCURRENT_REQUESTS = 40  # Número de llamadas paralelas

def extract_answer_before_explanation(text: str) -> str:
    """
    Extrae la respuesta antes de "Explanation:" o variaciones similares
    EXACTA del archivo llm_evaluator_parallel.py
    """
    if not text:
        return ""
    
    # Lista de patrones que indican el inicio de una explicación
    explanation_patterns = [
        "Explanation:",
        "\nExplanation:",
        "\n\nExplanation:",
        "Explanation :",
        "\nExplanation :",
        "\n\nExplanation :",
        "EXPLANATION:",
        "\nEXPLANATION:",
        "\n\nEXPLANATION:"
    ]
    
    # Buscar el primer patrón que aparezca
    split_position = len(text)
    for pattern in explanation_patterns:
        pos = text.find(pattern)
        if pos != -1 and pos < split_position:
            split_position = pos
    
    # Extraer solo la parte antes de la explicación
    clean_answer = text[:split_position].strip()
    
    # Remover saltos de línea adicionales al final
    while clean_answer.endswith('\n'):
        clean_answer = clean_answer[:-1].strip()
    
    return clean_answer

def is_yes_no_question(question: str, reference_answer: str) -> bool:
    """
    Detecta si es una pregunta de yes/no basándose en la pregunta y la respuesta de referencia
    EXACTA del archivo llm_evaluator_parallel.py
    """
    question_lower = question.lower().strip()
    reference_lower = reference_answer.lower().strip()
    
    # Patrones de preguntas yes/no
    yes_no_patterns = [
        "is it", "are they", "does", "do", "can", "could", "would", "should", "will",
        "has", "have", "was", "were", "did"
    ]
    
    # Si la respuesta de referencia es claramente yes/no
    if reference_lower in ["yes", "no", "true", "false"]:
        return True
    
    # Si la pregunta empieza con patrones típicos de yes/no
    for pattern in yes_no_patterns:
        if question_lower.startswith(pattern):
            return True
    
    return False

def extract_yes_no_from_answer(answer: str) -> str:
    """
    Extrae la respuesta yes/no del inicio de una respuesta
    EXACTA del archivo llm_evaluator_parallel.py
    """
    answer_lower = answer.lower().strip()
    
    # Buscar yes/no al inicio
    if answer_lower.startswith("yes"):
        return "yes"
    elif answer_lower.startswith("no"):
        return "no"
    elif answer_lower.startswith("true"):
        return "yes"
    elif answer_lower.startswith("false"):
        return "no"
    
    return answer_lower

async def evaluate_correctness_llm_exact_same_as_solver(
    question: str,
    generated_answer: str,
    reference_answer: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> int:
    """
    Evalúa si la respuesta generada es correcta usando el LLM
    EXACTAMENTE LA MISMA LÓGICA Y PROMPTS que llm_evaluator_parallel.py
    """
    async with semaphore:
        try:
            # Extraer solo la respuesta antes de "Explanation:"
            clean_generated_answer = extract_answer_before_explanation(generated_answer)
            
            # Detectar si es pregunta yes/no
            if is_yes_no_question(question, reference_answer):
                # Para preguntas yes/no, usar evaluación estricta
                reference_yn = extract_yes_no_from_answer(reference_answer)
                generated_yn = extract_yes_no_from_answer(clean_generated_answer)
                
                # Comparación directa para yes/no
                if reference_yn in ["yes", "no"] and generated_yn in ["yes", "no"]:
                    return 1 if reference_yn == generated_yn else 0
                
                # Si no se puede extraer claramente, usar LLM con prompt estricto
                prompt = f"""This is a YES/NO question evaluation. You must be extremely strict.

Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {clean_generated_answer}

STRICT YES/NO RULES:
- If reference is "yes" and generated answer starts with "No" → INCORRECT (0)
- If reference is "no" and generated answer starts with "Yes" → INCORRECT (0)
- The generated answer must have the SAME yes/no stance as the reference
- Any contradiction in yes/no stance = INCORRECT

Examples:
- Reference: "yes", Generated: "No, because..." → 0 (INCORRECT)
- Reference: "no", Generated: "Yes, it is..." → 0 (INCORRECT)
- Reference: "yes", Generated: "Yes, that's correct..." → 1 (CORRECT)

Respond with ONLY a single digit:
1 - CORRECT: Same yes/no stance
0 - INCORRECT: Different yes/no stance

Your verdict (just the digit 1 or 0):"""
            else:
                # Para preguntas no yes/no, usar evaluación más flexible
                prompt = f"""You are an expert evaluator for question answering systems. Your task is to determine if the generated answer correctly responds to the question according to the reference answer.

Question: {question}
Generated Answer: {clean_generated_answer}
Reference Answer: {reference_answer}

EVALUATION RULES:
1. The generated answer is CORRECT if it contains the reference answer, even if it provides additional correct information
2. The generated answer is INCORRECT only if it contradicts the reference answer or provides wrong information
3. For factual questions: The reference answer must be present in the generated answer

Examples:
- Reference: "Paris", Generated: "The capital is Paris, France" → CORRECT
- Reference: "The Wonder Years", Generated: "He wrote for The Wonder Years and other shows" → CORRECT

Respond with ONLY a single digit:
1 - CORRECT: The generated answer aligns with the reference answer
0 - INCORRECT: The generated answer contradicts or doesn't match the reference answer

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

async def evaluate_single_result(
    result: Dict[str, Any],
    index: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Evalúa un solo resultado de forma asíncrona
    """
    try:
        question = result['question']
        generated_answer = result['generated_answer']
        reference_answer = result['reference_answer']
        original_llm_correct = result.get('llm_correct', 0)
        
        # Extraer respuesta limpia
        clean_generated_answer = extract_answer_before_explanation(generated_answer)
        
        # Detectar si es pregunta yes/no
        is_yn_question = is_yes_no_question(question, reference_answer)
        
        # Nueva evaluación con EXACTAMENTE el mismo prompt del solver
        new_llm_correct = await evaluate_correctness_llm_exact_same_as_solver(
            question=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            client=client,
            semaphore=semaphore
        )
        
        # Crear nuevo resultado
        new_result = result.copy()
        new_result['llm_correct_original_ablation'] = original_llm_correct
        new_result['llm_correct_solver_prompt'] = new_llm_correct
        new_result['evaluation_changed'] = (original_llm_correct != new_llm_correct)
        new_result['clean_generated_answer'] = clean_generated_answer
        new_result['is_yes_no_question'] = is_yn_question
        new_result['_index'] = index  # Para mantener el orden
        
        # Para preguntas yes/no, agregar análisis adicional como en el solver
        if is_yn_question:
            reference_yn = extract_yes_no_from_answer(reference_answer)
            generated_yn = extract_yes_no_from_answer(clean_generated_answer)
            new_result['reference_yes_no'] = reference_yn
            new_result['generated_yes_no'] = generated_yn
            new_result['yes_no_match'] = reference_yn == generated_yn if reference_yn in ["yes", "no"] and generated_yn in ["yes", "no"] else None
        
        return new_result
        
    except Exception as e:
        logger.error(f"Error procesando pregunta {index}: {e}")
        # Mantener resultado original en caso de error
        new_result = result.copy()
        new_result['llm_correct_original_ablation'] = result.get('llm_correct', 0)
        new_result['llm_correct_solver_prompt'] = 0
        new_result['evaluation_changed'] = False
        new_result['error'] = str(e)
        new_result['_index'] = index
        return new_result

async def re_evaluate_results_file_parallel(
    results_file_path: Path,
    output_file_path: Path,
    client: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Re-evalúa un archivo de resultados usando EXACTAMENTE el mismo prompt del solver EN PARALELO
    """
    logger.info(f"Re-evaluando (paralelo con prompt exacto del solver): {results_file_path}")
    
    # Cargar resultados originales
    with open(results_file_path, 'r', encoding='utf-8') as f:
        original_results = json.load(f)
    
    total_questions = len(original_results)
    logger.info(f"Procesando {total_questions} preguntas con {MAX_CONCURRENT_REQUESTS} llamadas paralelas")
    
    # Crear semáforo para limitar concurrencia
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Crear tareas para todas las evaluaciones
    tasks = []
    for i, result in enumerate(original_results):
        task = evaluate_single_result(result, i, client, semaphore)
        tasks.append(task)
    
    # Ejecutar todas las tareas en paralelo con barra de progreso
    logger.info(f"Iniciando evaluación paralela de {len(tasks)} preguntas...")
    start_time = time.time()
    
    # Usar tqdm para mostrar progreso
    re_evaluated_results_unordered = await tqdm.gather(
        *tasks, 
        desc=f"Evaluando {results_file_path.name}",
        total=len(tasks)
    )
    
    # Reordenar resultados según el índice original
    re_evaluated_results = sorted(re_evaluated_results_unordered, key=lambda x: x['_index'])
    
    # Remover el índice temporal
    for result in re_evaluated_results:
        del result['_index']
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluación paralela completada en {evaluation_time:.2f} segundos")
    
    # Calcular estadísticas
    original_correct = sum(1 for r in re_evaluated_results if r.get('llm_correct_original_ablation', 0) == 1)
    new_correct = sum(1 for r in re_evaluated_results if r.get('llm_correct_solver_prompt', 0) == 1)
    changed_evaluations = sum(1 for r in re_evaluated_results if r.get('evaluation_changed', False))
    
    # Estadísticas específicas para yes/no
    yes_no_questions = [r for r in re_evaluated_results if r.get('is_yes_no_question', False)]
    yes_no_stats = {}
    if yes_no_questions:
        total_yn = len(yes_no_questions)
        correct_yn_original = sum(1 for r in yes_no_questions if r.get('llm_correct_original_ablation', 0) == 1)
        correct_yn_new = sum(1 for r in yes_no_questions if r.get('llm_correct_solver_prompt', 0) == 1)
        correct_yn_direct_match = sum(1 for r in yes_no_questions if r.get('yes_no_match', False) == True)
        
        yes_no_stats = {
            'total_yes_no_questions': total_yn,
            'original_correct_yn': correct_yn_original,
            'new_correct_yn': correct_yn_new,
            'direct_match_correct_yn': correct_yn_direct_match,
            'original_yn_accuracy': correct_yn_original / total_yn if total_yn > 0 else 0,
            'new_yn_accuracy': correct_yn_new / total_yn if total_yn > 0 else 0,
            'direct_match_yn_accuracy': correct_yn_direct_match / total_yn if total_yn > 0 else 0
        }
    
    original_accuracy = original_correct / total_questions if total_questions > 0 else 0
    new_accuracy = new_correct / total_questions if total_questions > 0 else 0
    change_rate = changed_evaluations / total_questions if total_questions > 0 else 0
    
    stats = {
        'total_questions': total_questions,
        'original_correct': original_correct,
        'new_correct': new_correct,
        'changed_evaluations': changed_evaluations,
        'original_accuracy': original_accuracy,
        'new_accuracy': new_accuracy,
        'accuracy_change': new_accuracy - original_accuracy,
        'change_rate': change_rate,
        'evaluation_time_seconds': evaluation_time,
        'questions_per_second': total_questions / evaluation_time if evaluation_time > 0 else 0,
        'input_file': str(results_file_path),
        'output_file': str(output_file_path),
        'yes_no_stats': yes_no_stats
    }
    
    # Guardar resultados re-evaluados
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(re_evaluated_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Re-evaluación completada:")
    logger.info(f"  Original accuracy: {original_accuracy:.4f}")
    logger.info(f"  New accuracy: {new_accuracy:.4f}")
    logger.info(f"  Change: {new_accuracy - original_accuracy:+.4f}")
    logger.info(f"  Changed evaluations: {changed_evaluations}/{total_questions} ({change_rate:.2%})")
    logger.info(f"  Speed: {stats['questions_per_second']:.1f} preguntas/segundo")
    
    if yes_no_stats:
        logger.info(f"  Yes/No questions: {yes_no_stats['total_yes_no_questions']}")
        logger.info(f"  Yes/No original accuracy: {yes_no_stats['original_yn_accuracy']:.4f}")
        logger.info(f"  Yes/No new accuracy: {yes_no_stats['new_yn_accuracy']:.4f}")
    
    return stats

async def find_and_re_evaluate_all_results():
    """
    Encuentra y re-evalúa todos los archivos results.json en las carpetas de ablation_combined
    """
    logger.info("=== INICIANDO RE-EVALUACIÓN PARALELA CON PROMPT EXACTO DEL SOLVER ===")
    logger.info(f"Configuración: {MAX_CONCURRENT_REQUESTS} llamadas paralelas al LLM")
    
    # Crear directorio de salida
    timestamp = int(time.time())
    output_base_dir = Path("ablation_combined") / f"re_evaluated_with_exact_solver_prompt_{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar cliente LLM
    llm_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    
    # Buscar todos los archivos results.json en las carpetas de ablación
    ablation_base_dir = Path("ablation_combined")
    results_files = []
    
    # Buscar en todas las carpetas ablation_study_*
    for study_dir in ablation_base_dir.glob("ablation_study_*"):
        if study_dir.is_dir():
            # Buscar en todas las subcarpetas alpha_*_similar_*
            for exp_dir in study_dir.glob("alpha_*_similar_*"):
                if exp_dir.is_dir():
                    results_file = exp_dir / "results.json"
                    if results_file.exists():
                        results_files.append(results_file)
    
    logger.info(f"Encontrados {len(results_files)} archivos de resultados para re-evaluar")
    
    # Re-evaluar cada archivo
    all_stats = []
    total_start_time = time.time()
    
    for i, results_file in enumerate(results_files):
        logger.info(f"\n--- Re-evaluando archivo {i+1}/{len(results_files)} ---")
        
        # Crear estructura de directorios de salida que refleje la original
        relative_path = results_file.relative_to(ablation_base_dir)
        output_file = output_base_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            stats = await re_evaluate_results_file_parallel(results_file, output_file, llm_client)
            all_stats.append(stats)
            
        except Exception as e:
            logger.error(f"Error re-evaluando {results_file}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Crear resumen general
    if all_stats:
        total_questions = sum(s['total_questions'] for s in all_stats)
        total_original_correct = sum(s['original_correct'] for s in all_stats)
        total_new_correct = sum(s['new_correct'] for s in all_stats)
        total_changed = sum(s['changed_evaluations'] for s in all_stats)
        total_evaluation_time = sum(s['evaluation_time_seconds'] for s in all_stats)
        
        # Estadísticas agregadas de yes/no
        total_yn_questions = sum(s.get('yes_no_stats', {}).get('total_yes_no_questions', 0) for s in all_stats)
        total_yn_original_correct = sum(s.get('yes_no_stats', {}).get('original_correct_yn', 0) for s in all_stats)
        total_yn_new_correct = sum(s.get('yes_no_stats', {}).get('new_correct_yn', 0) for s in all_stats)
        
        overall_stats = {
            'summary': {
                'total_files_processed': len(all_stats),
                'total_questions': total_questions,
                'total_original_correct': total_original_correct,
                'total_new_correct': total_new_correct,
                'total_changed_evaluations': total_changed,
                'overall_original_accuracy': total_original_correct / total_questions if total_questions > 0 else 0,
                'overall_new_accuracy': total_new_correct / total_questions if total_questions > 0 else 0,
                'overall_accuracy_change': (total_new_correct - total_original_correct) / total_questions if total_questions > 0 else 0,
                'overall_change_rate': total_changed / total_questions if total_questions > 0 else 0,
                'total_time_seconds': total_time,
                'total_evaluation_time_seconds': total_evaluation_time,
                'overall_questions_per_second': total_questions / total_evaluation_time if total_evaluation_time > 0 else 0,
                'max_concurrent_requests': MAX_CONCURRENT_REQUESTS,
                'total_yes_no_questions': total_yn_questions,
                'total_yn_original_correct': total_yn_original_correct,
                'total_yn_new_correct': total_yn_new_correct,
                'overall_yn_original_accuracy': total_yn_original_correct / total_yn_questions if total_yn_questions > 0 else 0,
                'overall_yn_new_accuracy': total_yn_new_correct / total_yn_questions if total_yn_questions > 0 else 0
            },
            'individual_file_stats': all_stats,
            'timestamp': timestamp,
            'output_directory': str(output_base_dir)
        }
        
        # Guardar resumen
        with open(output_base_dir / "re_evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2, ensure_ascii=False)
        
        # Crear CSV resumen para fácil análisis
        import pandas as pd
        
        # Crear DataFrame con estadísticas por archivo
        df_stats = pd.DataFrame([
            {
                'file': Path(s['input_file']).name,
                'experiment': Path(s['input_file']).parent.name,
                'total_questions': s['total_questions'],
                'original_accuracy': s['original_accuracy'],
                'new_accuracy': s['new_accuracy'],
                'accuracy_change': s['accuracy_change'],
                'changed_evaluations': s['changed_evaluations'],
                'change_rate': s['change_rate'],
                'questions_per_second': s['questions_per_second'],
                'yes_no_questions': s.get('yes_no_stats', {}).get('total_yes_no_questions', 0),
                'yes_no_original_accuracy': s.get('yes_no_stats', {}).get('original_yn_accuracy', 0),
                'yes_no_new_accuracy': s.get('yes_no_stats', {}).get('new_yn_accuracy', 0)
            }
            for s in all_stats
        ])
        
        df_stats.to_csv(output_base_dir / "re_evaluation_stats.csv", index=False)
        
        # Log resumen final
        logger.info(f"\n=== RESUMEN FINAL ===")
        logger.info(f"Archivos procesados: {len(all_stats)}")
        logger.info(f"Total preguntas: {total_questions}")
        logger.info(f"Tiempo total: {total_time:.2f} segundos")
        logger.info(f"Tiempo de evaluación: {total_evaluation_time:.2f} segundos")
        logger.info(f"Velocidad promedio: {overall_stats['summary']['overall_questions_per_second']:.1f} preguntas/segundo")
        logger.info(f"Accuracy original: {overall_stats['summary']['overall_original_accuracy']:.4f}")
        logger.info(f"Accuracy nueva: {overall_stats['summary']['overall_new_accuracy']:.4f}")
        logger.info(f"Cambio en accuracy: {overall_stats['summary']['overall_accuracy_change']:+.4f}")
        logger.info(f"Evaluaciones cambiadas: {total_changed}/{total_questions} ({overall_stats['summary']['overall_change_rate']:.2%})")
        
        if total_yn_questions > 0:
            logger.info(f"\n=== ESTADÍSTICAS YES/NO ===")
            logger.info(f"Total preguntas yes/no: {total_yn_questions}")
            logger.info(f"Yes/No accuracy original: {overall_stats['summary']['overall_yn_original_accuracy']:.4f}")
            logger.info(f"Yes/No accuracy nueva: {overall_stats['summary']['overall_yn_new_accuracy']:.4f}")
        
        logger.info(f"\nResultados guardados en: {output_base_dir}")
    
    else:
        logger.warning("No se procesaron archivos exitosamente")

async def main():
    """Función principal"""
    await find_and_re_evaluate_all_results()

if __name__ == "__main__":
    asyncio.run(main()) 