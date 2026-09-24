import json
from pathlib import Path
from typing import Dict, List, Union
from loguru import logger
import re
from datetime import datetime

def extract_label(text: str) -> str:
    """
    Extrae la respuesta principal del texto generado.
    Para HotPot, la respuesta suele estar en el primer párrafo o ser la primera afirmación.
    
    Args:
        text (str): El texto completo de la respuesta
        
    Returns:
        str: La respuesta extraída
    """
    # Dividir por párrafos
    paragraphs = text.split('\n')
    
    # Tomar el primer párrafo no vacío
    for paragraph in paragraphs:
        if paragraph.strip():
            # Limpiar la respuesta de cualquier texto explicativo
            # Buscar patrones comunes en respuestas de HotPot
            cleaned = paragraph.strip()
            # Si contiene "Based on", tomar solo la parte después
            if "Based on" in cleaned:
                parts = cleaned.split(",", 1)
                if len(parts) > 1:
                    cleaned = parts[1].strip()
            
            return cleaned.lower()
    return ""

def calculate_exact_match(generated_answer: str, reference_answer: Union[str, List[str]]) -> float:
    """
    Calcula la métrica Exact Match entre la respuesta generada y la referencia.
    Para HotPot, compara si la respuesta de referencia está contenida en la respuesta generada.
    
    Args:
        generated_answer (str): La respuesta generada por el modelo
        reference_answer (Union[str, List[str]]): La respuesta de referencia
        
    Returns:
        float: 1.0 si hay match, 0.0 si no
    """
    # Extraer y limpiar la respuesta generada
    generated_text = extract_label(generated_answer)
    if not generated_text:
        logger.warning("No se pudo extraer una respuesta clara del texto generado")
        return 0.0
    
    # Normalizar la referencia
    if isinstance(reference_answer, list):
        reference_texts = [ref.lower().strip() for ref in reference_answer if isinstance(ref, str)]
        # Verificar si alguna de las referencias está contenida en la respuesta generada
        return float(any(ref in generated_text for ref in reference_texts))
    
    # Si la referencia es un string
    reference_text = reference_answer.lower().strip()
    # Verificar si la referencia está contenida en la respuesta generada
    return float(reference_text in generated_text)

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
        for result in results:
            question_id = result.get('question_id', 'Unknown')
            question = result.get('question', '')
            generated_answer = result.get('generated_answer', '')
            reference_answer = result.get('reference_answer', '')
            
            if not generated_answer or not reference_answer:
                logger.warning(f"Respuesta faltante para pregunta {question_id}")
                continue
            
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
    results_dir = Path(__file__).parent.parent / "Benchmark" / "results_bien" / "combined"
    
    # Evaluar tanto resultados completos como parciales
    result_files = [
        results_dir / "exact_answers" / "full_results.json"
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