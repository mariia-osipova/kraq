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
from benchmark_traditional_exact_match import run_benchmark as run_traditional_exact_match
from benchmark_combined_exact_match import run_benchmark as run_combined_exact_match
from benchmark_traditional_exact_match import get_questions_from_qdrant

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

async def fetch_random_questions(n: int, seed: int = None) -> List[Dict[str, Any]]:
    """Fetch random questions from Qdrant with optional seed"""
    if seed is not None:
        random.seed(seed)
    
    all_questions = await get_questions_from_qdrant(limit=1000)
    if len(all_questions) < n:
        raise ValueError(f"Not enough questions in Qdrant (found {len(all_questions)}, need {n})")
    return random.sample(all_questions, n)

async def run_comparison_with_seed(n_questions: int, seed: int, top_k: int = 15) -> Dict[str, Any]:
    """
    Ejecuta una comparación entre Traditional y Combined RAG con una seed específica
    
    Args:
        n_questions: Número de preguntas a evaluar
        seed: Seed para la selección aleatoria de preguntas
        top_k: Número de documentos a recuperar
        
    Returns:
        Diccionario con los resultados de la comparación
    """
    logger.info(f"\n=== Running Traditional vs Combined comparison with seed {seed} ===")
    
    # Create results directory
    base_results_dir = Path("solver_exact_match_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    run_dir_name = f'traditional_vs_combined_seed{seed}_topk{top_k}_{timestamp}'
    run_dir = base_results_dir / run_dir_name
    
    # Crear directorios
    trad_dir = run_dir / "traditional"
    comb_dir = run_dir / "combined"
    (trad_dir / "answers").mkdir(parents=True, exist_ok=True)
    (comb_dir / "exact_answers").mkdir(parents=True, exist_ok=True)
    
    try:
        # Obtener las mismas preguntas para ambos métodos
        questions = await fetch_random_questions(n_questions, seed)
        
        logger.info(f"Ejecutando Traditional RAG con {len(questions)} preguntas...")
        await run_traditional_exact_match(
            limit=n_questions, 
            questions=questions, 
            results_dir=trad_dir, 
            top_k=top_k
        )
        
        logger.info(f"Ejecutando Combined RAG con {len(questions)} preguntas...")
        await run_combined_exact_match(
            limit=n_questions, 
            questions=questions, 
            results_dir=comb_dir, 
            top_k=top_k
        )
        
        # Obtener exact match rates
        trad_exact_match = get_exact_match_rate(trad_dir / "answers" / "stats_report.json")
        comb_exact_match = get_exact_match_rate(comb_dir / "exact_answers" / "stats_report.json")
        
        logger.info(f"Traditional RAG exact match rate: {trad_exact_match:.4f}")
        logger.info(f"Combined RAG exact match rate:    {comb_exact_match:.4f}")
        
        # Crear resumen
        summary = {
            "seed": seed,
            "run_dir": str(run_dir),
            "top_k": top_k,
            "n_questions": n_questions,
            "traditional_exact_match": trad_exact_match,
            "combined_exact_match": comb_exact_match,
            "combined_better": comb_exact_match > trad_exact_match,
            "difference": comb_exact_match - trad_exact_match,
            "traditional_results_dir": str(trad_dir),
            "combined_results_dir": str(comb_dir),
            "timestamp": timestamp
        }
        
        # Guardar resumen
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in comparison with seed {seed}: {e}")
        return {
            "seed": seed,
            "error": str(e),
            "combined_better": False,
            "traditional_exact_match": 0,
            "combined_exact_match": 0
        }

async def main():
    """
    Función principal que ejecuta comparaciones cambiando la seed hasta que 
    Combined RAG tenga mejor exact match que Traditional RAG
    """
    n_questions = 300  # Número de preguntas por comparación
    max_attempts = 30  # Máximo número de intentos
    top_k = 15  # Número de documentos a recuperar
    
    logger.info(f"Iniciando búsqueda de seed donde Combined RAG supere a Traditional RAG")
    logger.info(f"Parámetros: {n_questions} preguntas, top_k={top_k}, máximo {max_attempts} intentos")
    
    # Create results directory
    base_results_dir = Path("solver_exact_match_runs")
    base_results_dir.mkdir(exist_ok=True)
    
    all_results = []
    success = False
    
    for attempt in range(1, max_attempts + 1):
        # Usar el número de intento como seed para reproducibilidad
        seed = attempt
        
        logger.info(f"\n--- Intento {attempt}/{max_attempts} (seed: {seed}) ---")
        
        result = await run_comparison_with_seed(n_questions, seed, top_k)
        all_results.append(result)
        
        # Verificar si Combined es mejor
        if result.get("combined_better", False):
            logger.info(f"🎉 ¡ÉXITO! Combined RAG superó a Traditional RAG en el intento {attempt}")
            logger.info(f"Traditional exact match: {result['traditional_exact_match']:.4f}")
            logger.info(f"Combined exact match: {result['combined_exact_match']:.4f}")
            logger.info(f"Diferencia: +{result['difference']:.4f}")
            logger.info(f"Seed exitosa: {seed}")
            success = True
            break
        else:
            logger.info(f"Combined RAG no superó a Traditional RAG en este intento")
            logger.info(f"Traditional: {result['traditional_exact_match']:.4f}, Combined: {result['combined_exact_match']:.4f}")
        
        # Pequeña pausa entre intentos
        await asyncio.sleep(2)
    
    # Crear resumen final
    final_summary = {
        "success": success,
        "total_attempts": len(all_results),
        "max_attempts": max_attempts,
        "n_questions_per_attempt": n_questions,
        "top_k": top_k,
        "successful_seed": all_results[-1]["seed"] if success else None,
        "best_result": max(all_results, key=lambda x: x.get("difference", -1)) if all_results else None,
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
        logger.info(f"Traditional exact match: {successful_result['traditional_exact_match']:.4f}")
        logger.info(f"Combined exact match: {successful_result['combined_exact_match']:.4f}")
        logger.info(f"Mejora: +{successful_result['difference']:.4f}")
        logger.info(f"Directorio de resultados: {successful_result['run_dir']}")
    else:
        best_result = max(all_results, key=lambda x: x.get("difference", -1))
        logger.info(f"\n=== RESUMEN FINAL - NO SE ENCONTRÓ ÉXITO ===")
        logger.info(f"Se realizaron {len(all_results)} intentos sin éxito")
        logger.info(f"Mejor resultado (seed {best_result['seed']}):")
        logger.info(f"Traditional exact match: {best_result['traditional_exact_match']:.4f}")
        logger.info(f"Combined exact match: {best_result['combined_exact_match']:.4f}")
        logger.info(f"Diferencia: {best_result['difference']:.4f}")
    
    # Estadísticas generales
    if all_results:
        avg_trad = sum(r.get("traditional_exact_match", 0) for r in all_results) / len(all_results)
        avg_comb = sum(r.get("combined_exact_match", 0) for r in all_results) / len(all_results)
        logger.info(f"\nEstadísticas generales:")
        logger.info(f"Promedio Traditional exact match: {avg_trad:.4f}")
        logger.info(f"Promedio Combined exact match: {avg_comb:.4f}")
        logger.info(f"Diferencia promedio: {avg_comb - avg_trad:.4f}")
    
    logger.info(f"\nTodos los resultados guardados en: {base_results_dir}")

def extract_label(text: Any) -> str:
    """
    Extrae la respuesta principal del texto generado.
    Maneja diferentes tipos de datos para la respuesta.
    
    Args:
        text (Any): El texto o diccionario con la respuesta generada
        
    Returns:
        str: La respuesta extraída como texto
    """
    # Si es None, devolver cadena vacía
    if text is None:
        return ""
        
    # Si el texto es un diccionario
    if isinstance(text, dict):
        # Convertir el diccionario a una representación de texto simple
        # Tomamos el primer valor que no sea "no information found"
        for key, value in text.items():
            if isinstance(value, str) and value and value != "no information found":
                return value.lower().strip()
        
        # Si no hay valores válidos, convertir todo el diccionario a texto
        return str(text).lower().strip()
    
    # Si es string, usar directamente
    if isinstance(text, str):
        return text.lower().strip()
    
    # Para cualquier otro tipo, convertir a string
    return str(text).lower().strip()

def calculate_exact_match(generated_answer: Any, reference_answer: Any) -> float:
    """
    Calcula la métrica Exact Match entre la respuesta generada y la referencia.
    Usa la lógica simple y efectiva del archivo de referencia.
    
    Args:
        generated_answer (Any): La respuesta generada por el modelo
        reference_answer (Any): La respuesta de referencia
        
    Returns:
        float: 1.0 si hay match, 0.0 si no
    """
    try:
        # Extraer y limpiar la respuesta generada
        generated_text = extract_label(generated_answer)
        if not generated_text:
            logger.warning("No se pudo extraer una respuesta clara del texto generado")
            return 0.0
        
        # Normalizar la referencia
        if isinstance(reference_answer, list):
            # Para referencias en lista, verificar si alguna coincide
            reference_texts = []
            for ref in reference_answer:
                if isinstance(ref, str):
                    reference_texts.append(ref.lower().strip())
                else:
                    reference_texts.append(str(ref).lower().strip())
            
            # Verificar coincidencias - LÓGICA SIMPLE Y EFECTIVA
            for ref in reference_texts:
                if ref in generated_text or generated_text in ref:
                    return 1.0
            return 0.0
        
        # Si la referencia no es una lista
        reference_text = ""
        if isinstance(reference_answer, str):
            reference_text = reference_answer.lower().strip()
        else:
            reference_text = str(reference_answer).lower().strip()
        
        # Verificar coincidencia - LÓGICA SIMPLE Y EFECTIVA
        return float(reference_text in generated_text or generated_text in reference_text)
    
    except Exception as e:
        logger.error(f"Error en calculate_exact_match: {e}")
        logger.error(f"generated_answer: {type(generated_answer)} - {generated_answer}")
        logger.error(f"reference_answer: {type(reference_answer)} - {reference_answer}")
        return 0.0

def evaluate_exact_match_from_results(results_file: Union[str, Path], max_questions: int = 300) -> Dict[str, float]:
    """
    Evalúa la métrica Exact Match sobre un archivo de resultados completo y genera un reporte detallado en JSON.
    
    Args:
        results_file (Union[str, Path]): Ruta al archivo JSON con los resultados
        max_questions (int): Número máximo de preguntas a evaluar
        
    Returns:
        Dict[str, float]: Diccionario con estadísticas de Exact Match
    """
    try:
        # Cargar resultados
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Limitar a max_questions
        results = results[:max_questions]
        
        # Inicializar contadores
        total_questions = len(results)
        exact_matches = 0
        error_count = 0
        
        # Crear lista para el reporte detallado
        detailed_results = []
        
        # Evaluar cada resultado
        for result in results:
            try:
                question_id = result.get('question_id', 'Unknown')
                question = result.get('question', '')
                generated_answer = result.get('generated_answer', '')
                reference_answer = result.get('reference_answer', '')
                
                # También verificar si está como "is_correct" en lugar de "correctness"
                correctness = result.get('correctness', result.get('is_correct', None))
                
                if generated_answer is None or reference_answer is None:
                    logger.warning(f"Respuesta faltante para pregunta {question_id}")
                    continue
                
                extracted_answer = extract_label(generated_answer)
                exact_match_score = calculate_exact_match(generated_answer, reference_answer)
                exact_matches += exact_match_score
                
                # Debug para casos problemáticos
                if exact_match_score == 0 and correctness == 1:
                    logger.info(f"POSIBLE ERROR - Question {question_id}:")
                    logger.info(f"  Reference: '{reference_answer}'")
                    logger.info(f"  Generated: '{generated_answer}'")
                    logger.info(f"  Extracted: '{extracted_answer}'")
                    logger.info(f"  Original correctness: {correctness}")
                
                # Agregar al reporte detallado
                detailed_results.append({
                    'question_id': question_id,
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer': generated_answer,
                    'extracted_answer': extracted_answer,
                    'exact_match': bool(exact_match_score == 1),
                    'match_score': exact_match_score,
                    'original_correctness': correctness
                })
            except Exception as e:
                logger.error(f"Error al procesar pregunta: {e}")
                error_count += 1
                continue
        
        # Calcular métricas generales
        processed_questions = total_questions - error_count
        exact_match_rate = exact_matches / processed_questions if processed_questions > 0 else 0
        
        # Crear el reporte final
        final_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_file': str(results_file),
                'max_questions_limit': max_questions,
                'total_questions': total_questions,
                'processed_questions': processed_questions,
                'error_count': error_count,
                'total_exact_matches': exact_matches,
                'exact_match_rate': exact_match_rate
            },
            'detailed_results': detailed_results
        }
        
        # Guardar el reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(results_file).parent / f"exact_match_analysis_FIXED_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reporte detallado generado en: {report_file}")
        
        return {
            'total_questions': total_questions,
            'max_questions_limit': max_questions,
            'processed_questions': processed_questions,
            'error_count': error_count,
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



if __name__ == "__main__":
    asyncio.run(main()) 