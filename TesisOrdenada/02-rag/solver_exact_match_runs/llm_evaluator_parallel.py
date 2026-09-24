#!/usr/bin/env python3
"""
Script para evaluar en paralelo todas las respuestas usando el LLM evaluador
Procesa tanto combined/exact_answers como traditional/answers con 20 llamadas concurrentes
Genera reportes detallados por pregunta
"""

import asyncio
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MAX_CONCURRENT = 30  # Número de llamadas concurrentes

def extract_answer_before_explanation(text: str) -> str:
    """
    Extrae la respuesta antes de "Explanation:" o variaciones similares
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

async def evaluate_correctness_llm(
    question: str,
    generated_answer: str,
    reference_answer: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> int:
    """
    Evalúa si la respuesta generada es correcta usando el LLM
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

async def process_single_result(
    result: Dict[str, Any],
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int
) -> Dict[str, Any]:
    """
    Procesa un solo resultado y agrega la evaluación LLM
    """
    try:
        question = result.get('question', '')
        generated_answer = result.get('generated_answer', '')
        reference_answer = result.get('reference_answer', '')
        
        if not all([question, generated_answer, reference_answer]):
            logger.warning(f"Datos faltantes en resultado {index}")
            result['llm_is_correct'] = 0
            result['clean_generated_answer'] = ""
            result['is_yes_no_question'] = False
            return result
        
        # Extraer respuesta limpia
        clean_generated_answer = extract_answer_before_explanation(generated_answer)
        
        # Detectar si es pregunta yes/no
        is_yn_question = is_yes_no_question(question, reference_answer)
        
        # Evaluar con LLM
        llm_correctness = await evaluate_correctness_llm(
            question, generated_answer, reference_answer, client, semaphore
        )
        
        # REORGANIZAR EL RESULTADO PARA QUE exact_match Y llm_is_correct ESTÉN JUNTOS
        # Crear un nuevo diccionario con el orden deseado
        organized_result = {}
        
        # Mantener todos los campos originales en su orden original hasta exact_match
        for key, value in result.items():
            organized_result[key] = value
            # Cuando llegamos a exact_match, agregamos llm_is_correct inmediatamente después
            if key == 'exact_match':
                organized_result['llm_is_correct'] = llm_correctness
        
        # Agregar campos adicionales de análisis
        organized_result['clean_generated_answer'] = clean_generated_answer
        organized_result['is_yes_no_question'] = is_yn_question
        
        # Para preguntas yes/no, agregar análisis adicional
        if is_yn_question:
            reference_yn = extract_yes_no_from_answer(reference_answer)
            generated_yn = extract_yes_no_from_answer(clean_generated_answer)
            organized_result['reference_yes_no'] = reference_yn
            organized_result['generated_yes_no'] = generated_yn
            organized_result['yes_no_match'] = reference_yn == generated_yn if reference_yn in ["yes", "no"] and generated_yn in ["yes", "no"] else None
        
        # Log progreso cada 10 evaluaciones
        if (index + 1) % 10 == 0:
            logger.info(f"Procesado {index + 1}/{total} resultados")
        
        return organized_result
        
    except Exception as e:
        logger.error(f"Error procesando resultado {index}: {e}")
        result['llm_is_correct'] = 0
        result['clean_generated_answer'] = ""
        result['is_yes_no_question'] = False
        return result

def create_csv_report(results: List[Dict[str, Any]], output_file: Path, method_name: str):
    """
    Crea SOLO un reporte CSV para fácil visualización (sin JSON duplicado)
    """
    detailed_report = []
    
    for i, result in enumerate(results):
        question = result.get('question', '')
        generated_answer = result.get('generated_answer', '')
        reference_answer = result.get('reference_answer', '')
        clean_generated_answer = result.get('clean_generated_answer', '')
        exact_match = result.get('exact_match', 0)
        llm_is_correct = result.get('llm_is_correct', 0)
        original_is_correct = result.get('is_correct', 0)
        is_yes_no = result.get('is_yes_no_question', False)
        
        detailed_entry = {
            "question_id": i + 1,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer_full": generated_answer,
            "clean_answer_sent_to_llm": clean_generated_answer,
            "exact_match_score": exact_match,
            "llm_evaluation_score": llm_is_correct,
            "original_is_correct_score": original_is_correct,
            "answer_was_cleaned": len(clean_generated_answer) != len(generated_answer.strip()),
            "is_yes_no_question": is_yes_no,
            "method": method_name
        }
        
        # Agregar campos específicos para preguntas yes/no
        if is_yes_no:
            detailed_entry["reference_yes_no"] = result.get('reference_yes_no', '')
            detailed_entry["generated_yes_no"] = result.get('generated_yes_no', '')
            detailed_entry["yes_no_match"] = result.get('yes_no_match', None)
        
        detailed_report.append(detailed_entry)
    
    # SOLO crear CSV (no JSON duplicado)
    csv_output = output_file.parent / f"{output_file.stem}_detailed_report.csv"
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'question_id', 'question', 'reference_answer', 
            'generated_answer_full', 'clean_answer_sent_to_llm',
            'exact_match_score', 'llm_evaluation_score', 'original_is_correct_score',
            'answer_was_cleaned', 'is_yes_no_question', 'reference_yes_no', 
            'generated_yes_no', 'yes_no_match', 'method'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_report)
    
    logger.info(f"Reporte CSV guardado en: {csv_output}")
    
    # Estadísticas específicas para yes/no
    yes_no_questions = [r for r in detailed_report if r.get('is_yes_no_question', False)]
    if yes_no_questions:
        total_yn = len(yes_no_questions)
        correct_yn_llm = sum(1 for r in yes_no_questions if r.get('llm_evaluation_score', 0) == 1)
        correct_yn_exact = sum(1 for r in yes_no_questions if r.get('yes_no_match', False) == True)
        
        logger.info(f"\n=== ESTADÍSTICAS YES/NO ===")
        logger.info(f"Total preguntas yes/no: {total_yn}")
        logger.info(f"Correctas según LLM: {correct_yn_llm}/{total_yn} ({correct_yn_llm/total_yn:.2%})")
        logger.info(f"Correctas según match directo: {correct_yn_exact}/{total_yn} ({correct_yn_exact/total_yn:.2%})")
    
    return detailed_report

async def evaluate_results_file(
    results_file: Path,
    output_file: Path,
    method_name: str
) -> Dict[str, Any]:
    """
    Evalúa un archivo de resultados completo
    """
    logger.info(f"Iniciando evaluación LLM para {method_name}: {results_file}")
    
    # Cargar resultados
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Error cargando {results_file}: {e}")
        return {"error": str(e)}
    
    logger.info(f"Cargados {len(results)} resultados para {method_name}")
    
    # Configurar cliente LLM
    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"
    )
    
    # Semáforo para controlar concurrencia
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Procesar todos los resultados en paralelo
    start_time = time.time()
    
    tasks = [
        process_single_result(result, client, semaphore, i, len(results))
        for i, result in enumerate(results)
    ]
    
    # Ejecutar todas las tareas
    logger.info(f"Ejecutando {len(tasks)} evaluaciones con {MAX_CONCURRENT} llamadas concurrentes...")
    updated_results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calcular estadísticas
    total_questions = len(updated_results)
    llm_correct = sum(1 for r in updated_results if r.get('llm_is_correct', 0) == 1)
    exact_matches = sum(1 for r in updated_results if r.get('exact_match', 0) == 1)
    original_correct = sum(1 for r in updated_results if r.get('is_correct', 0) == 1)
    
    llm_accuracy = llm_correct / total_questions if total_questions > 0 else 0
    exact_match_rate = exact_matches / total_questions if total_questions > 0 else 0
    original_accuracy = original_correct / total_questions if total_questions > 0 else 0
    
    # SOLO guardar el archivo principal con toda la información
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_results, f, indent=2, ensure_ascii=False)
    
    # Crear SOLO el reporte CSV para visualización fácil
    create_csv_report(updated_results, output_file, method_name)
    
    # Crear reporte de estadísticas
    stats = {
        "method": method_name,
        "total_questions": total_questions,
        "llm_correct": llm_correct,
        "exact_matches": exact_matches,
        "original_correct": original_correct,
        "llm_accuracy": llm_accuracy,
        "exact_match_rate": exact_match_rate,
        "original_accuracy": original_accuracy,
        "evaluation_time": total_time,
        "questions_per_second": total_questions / total_time if total_time > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Evaluación completada para {method_name}:")
    logger.info(f"  - Total preguntas: {total_questions}")
    logger.info(f"  - LLM accuracy: {llm_accuracy:.4f}")
    logger.info(f"  - Exact match rate: {exact_match_rate:.4f}")
    logger.info(f"  - Original accuracy: {original_accuracy:.4f}")
    logger.info(f"  - Tiempo total: {total_time:.2f}s")
    logger.info(f"  - Velocidad: {total_questions / total_time:.2f} preguntas/segundo")
    
    return stats

async def process_run_directory(run_dir: Path) -> Dict[str, Any]:
    """
    Procesa un directorio de corrida completo (traditional + combined)
    """
    logger.info(f"Procesando directorio de corrida: {run_dir}")
    
    # Definir archivos de entrada y salida
    files_to_process = [
        {
            "input": run_dir / "traditional" / "answers" / "full_results.json",
            "output": run_dir / "traditional" / "answers" / "full_results_llm_evaluated.json",
            "method": "traditional"
        },
        {
            "input": run_dir / "combined" / "exact_answers" / "full_results.json",
            "output": run_dir / "combined" / "exact_answers" / "full_results_llm_evaluated.json",
            "method": "combined"
        }
    ]
    
    results = {}
    
    for file_info in files_to_process:
        input_file = file_info["input"]
        output_file = file_info["output"]
        method = file_info["method"]
        
        if not input_file.exists():
            logger.warning(f"Archivo no encontrado: {input_file}")
            continue
        
        # Evaluar archivo
        stats = await evaluate_results_file(input_file, output_file, method)
        results[method] = stats
    
    # Crear reporte comparativo
    if "traditional" in results and "combined" in results:
        comparison = {
            "run_directory": str(run_dir),
            "traditional_stats": results["traditional"],
            "combined_stats": results["combined"],
            "comparison": {
                "llm_accuracy_difference": results["combined"]["llm_accuracy"] - results["traditional"]["llm_accuracy"],
                "exact_match_difference": results["combined"]["exact_match_rate"] - results["traditional"]["exact_match_rate"],
                "original_accuracy_difference": results["combined"]["original_accuracy"] - results["traditional"]["original_accuracy"],
                "combined_better_llm": results["combined"]["llm_accuracy"] > results["traditional"]["llm_accuracy"],
                "combined_better_exact": results["combined"]["exact_match_rate"] > results["traditional"]["exact_match_rate"],
                "combined_better_original": results["combined"]["original_accuracy"] > results["traditional"]["original_accuracy"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar reporte comparativo
        comparison_file = run_dir / "llm_evaluation_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte comparativo guardado en: {comparison_file}")
        
        # Log comparación
        logger.info(f"\n=== COMPARACIÓN LLM EVALUATION ===")
        logger.info(f"Traditional LLM accuracy: {results['traditional']['llm_accuracy']:.4f}")
        logger.info(f"Combined LLM accuracy: {results['combined']['llm_accuracy']:.4f}")
        logger.info(f"Diferencia: {comparison['comparison']['llm_accuracy_difference']:.4f}")
        logger.info(f"Combined mejor: {comparison['comparison']['combined_better_llm']}")
        
        return comparison
    
    return results

async def main():
    """
    Función principal que procesa todas las corridas en solver_exact_match_runs
    """
    logger.info("Iniciando evaluación LLM en paralelo para todas las corridas")
    
    # Directorio base
    base_dir = Path("solver_exact_match_runs")
    
    if not base_dir.exists():
        logger.error(f"Directorio no encontrado: {base_dir}")
        return
    
    # Buscar todos los directorios de corridas
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("traditional_vs_combined")]
    
    if not run_dirs:
        logger.error("No se encontraron directorios de corridas")
        return
    
    logger.info(f"Encontrados {len(run_dirs)} directorios de corridas")
    
    all_results = []
    
    for run_dir in sorted(run_dirs):
        try:
            result = await process_run_directory(run_dir)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error procesando {run_dir}: {e}")
            continue
    
    # Crear resumen final
    final_summary = {
        "total_runs_processed": len(all_results),
        "runs": all_results,
        "overall_stats": {
            "runs_where_combined_better_llm": sum(1 for r in all_results if r.get("comparison", {}).get("combined_better_llm", False)),
            "runs_where_combined_better_exact": sum(1 for r in all_results if r.get("comparison", {}).get("combined_better_exact", False)),
            "runs_where_combined_better_original": sum(1 for r in all_results if r.get("comparison", {}).get("combined_better_original", False))
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Guardar resumen final
    summary_file = base_dir / "llm_evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n=== RESUMEN FINAL ===")
    logger.info(f"Corridas procesadas: {len(all_results)}")
    logger.info(f"Combined mejor (LLM): {final_summary['overall_stats']['runs_where_combined_better_llm']}")
    logger.info(f"Combined mejor (Exact): {final_summary['overall_stats']['runs_where_combined_better_exact']}")
    logger.info(f"Combined mejor (Original): {final_summary['overall_stats']['runs_where_combined_better_original']}")
    logger.info(f"Resumen guardado en: {summary_file}")

if __name__ == "__main__":
    asyncio.run(main())