import json
from pathlib import Path
from typing import Dict, List, Union, Any
from loguru import logger
import re
from datetime import datetime
import traceback

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
            
            # Verificar coincidencias
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
        
        # Verificar coincidencia
        return float(reference_text in generated_text or generated_text in reference_text)
    
    except Exception as e:
        logger.error(f"Error en calculate_exact_match: {e}")
        logger.error(f"generated_answer: {type(generated_answer)} - {generated_answer}")
        logger.error(f"reference_answer: {type(reference_answer)} - {reference_answer}")
        return 0.0

def evaluate_exact_match_from_results(results_file: Union[str, Path]) -> Dict[str, float]:
    """
    Evalúa la métrica Exact Match sobre un archivo de resultados completo y genera un reporte detallado en JSON.
    
    Args:
        results_file (Union[str, Path]): Ruta al archivo JSON con los resultados
        
    Returns:
        Dict[str, float]: Diccionario con estadísticas de Exact Match
    """
    try:
        # Cargar resultados
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Inicializar contadores
        total_questions = len(results)
        exact_matches = 0
        
        # Crear lista para el reporte detallado
        detailed_results = []
        
        # Evaluar cada resultado
        for i, result in enumerate(results):
            try:
                question_id = result.get('question_id', f'Unknown-{i}')
                question = result.get('question', '')
                generated_answer = result.get('generated_answer', '')
                reference_answer = result.get('reference_answer', '')
                
                if generated_answer is None or reference_answer is None:
                    logger.warning(f"Respuesta faltante para pregunta {question_id}")
                    continue
                
                # Extraer la respuesta y calcular coincidencia
                extracted_answer = extract_label(generated_answer)
                exact_match_score = calculate_exact_match(generated_answer, reference_answer)
                exact_matches += exact_match_score
                
                # Agregar al reporte detallado
                detailed_results.append({
                    'question_id': question_id,
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer': generated_answer,
                    'extracted_answer': extracted_answer,
                    'exact_match': bool(exact_match_score == 1),
                    'match_score': exact_match_score
                })
            except Exception as e:
                logger.error(f"Error procesando pregunta {i}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Calcular métricas generales
        exact_match_rate = exact_matches / total_questions if total_questions > 0 else 0
        
        # Crear el reporte final
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
        
        # Guardar el reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(results_file).parent / f"exact_match_analysis_{timestamp}.json"
        
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
        logger.error(traceback.format_exc())
        return {
            'total_questions': 0,
            'exact_matches': 0,
            'exact_match_rate': 0.0,
            'error': str(e)
        }

def main():
    """
    Función principal para ejecutar la evaluación de Exact Match sobre los resultados.
    """
    # Definir la ruta al archivo de resultados
    results_dir = Path(__file__).parent.parent / "BenchMarks" / "results"
    
    # Evaluar tanto resultados completos como parciales
    result_files = [
        results_dir / "answers" / "full_results.json"
    ]
    
    for results_file in result_files:
        if not results_file.exists():
            logger.warning(f"No se encontró el archivo de resultados en {results_file}")
            continue
            
        print(f"\n=== Evaluando {results_file.name} ===")
        metrics = evaluate_exact_match_from_results(results_file)
        
        # Imprimir resultados
        print(f"Total de preguntas evaluadas: {metrics['total_questions']}")
        print(f"Número de Exact Matches: {metrics['exact_matches']}")
        print(f"Tasa de Exact Match: {metrics['exact_match_rate']:.2%}")
        if 'report_file' in metrics:
            print(f"Reporte detallado generado en: {metrics['report_file']}")

if __name__ == "__main__":
    main()