import json
from pathlib import Path
from typing import Dict, List, Union, Tuple
from loguru import logger
import re

def extract_label(text: str) -> str:
    """
    Extrae la etiqueta (label) de la respuesta generada.
    Busca después de 'Label:' o 'LABEL:' y antes de 'Explanation' o nueva línea.
    
    Args:
        text (str): El texto completo de la respuesta
        
    Returns:
        str: La etiqueta encontrada o texto vacío si no se encuentra
    """
    # Buscar el patrón "Label:" o "LABEL:" seguido por el valor
    label_pattern = re.compile(r'(?:Label:|LABEL:)\s*([^:\n]+?)(?:\s*(?:Explanation:|$)|\n)', re.IGNORECASE)
    match = label_pattern.search(text)
    
    if match:
        return match.group(1).strip().lower()
    return ""

def calculate_exact_match(generated_answer: str, reference_answer: Union[str, List[str]]) -> float:
    """
    Calcula la métrica Exact Match entre la etiqueta de la respuesta generada y la referencia.
    
    Args:
        generated_answer (str): La respuesta generada por el modelo
        reference_answer (Union[str, List[str]]): La respuesta de referencia (puede ser string o lista)
        
    Returns:
        float: 1.0 si hay exact match, 0.0 si no
    """
    # Extraer la etiqueta de la respuesta generada
    generated_label = extract_label(generated_answer)
    if not generated_label:
        logger.warning("No se encontró etiqueta en la respuesta generada")
        return 0.0
    
    # Normalizar la referencia
    if isinstance(reference_answer, list):
        # Convertir cada elemento de la lista a minúsculas
        reference_labels = [ref.lower().strip() for ref in reference_answer if isinstance(ref, str)]
        return float(generated_label in reference_labels)
    
    # Si la referencia es un string
    reference_label = reference_answer.lower().strip()
    return float(generated_label == reference_label)

def evaluate_exact_match_from_results(results_file: Union[str, Path]) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    """
    Evalúa la métrica Exact Match sobre un archivo de resultados completo.
    
    Args:
        results_file (Union[str, Path]): Ruta al archivo JSON con los resultados
        
    Returns:
        Tuple[Dict[str, float], List[Dict[str, str]]]: 
            - Diccionario con estadísticas de Exact Match
            - Lista de discrepancias (casos donde is_correct=1 pero no hay exact match)
    """
    try:
        # Cargar resultados
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Inicializar contadores y lista de discrepancias
        total_questions = len(results)
        exact_matches = 0
        discrepancies = []
        
        # Evaluar cada resultado
        for result in results:
            generated_answer = result.get('generated_answer', '')
            reference_answer = result.get('reference_answer', '')
            question = result.get('question', 'Unknown')
            is_correct = result.get('is_correct', 0)
            
            if not generated_answer or not reference_answer:
                logger.warning(f"Respuesta faltante para pregunta {question}")
                continue
                
            exact_match_score = calculate_exact_match(generated_answer, reference_answer)
            exact_matches += exact_match_score
            
            # Verificar discrepancias (is_correct=1 pero no hay exact match)
            if is_correct == 1 and exact_match_score == 0:
                discrepancies.append({
                    'question': question,
                    'generated_label': extract_label(generated_answer),
                    'reference_answer': reference_answer,
                    'question_id': result.get('question_id', 'Unknown')
                })
        
        # Calcular métricas
        exact_match_rate = exact_matches / total_questions if total_questions > 0 else 0
        
        return {
            'total_questions': total_questions,
            'exact_matches': exact_matches,
            'exact_match_rate': exact_match_rate
        }, discrepancies
        
    except Exception as e:
        logger.error(f"Error evaluando Exact Match: {e}")
        return {
            'total_questions': 0,
            'exact_matches': 0,
            'exact_match_rate': 0.0,
            'error': str(e)
        }, []

def main():
    """
    Función principal para ejecutar la evaluación de Exact Match sobre los resultados.
    """
    # Definir la ruta al archivo de resultados
    results_dir = Path(__file__).parent.parent / "Benchmark" / "results_finetuned"
    
    # Evaluar tanto resultados completos como parciales
    result_files = [
        results_dir / "exact_answers" / "full_results.json",
        results_dir / "exact_answers" / "partial_results.json",
        results_dir / "ideal_answers" / "full_results.json",
        results_dir / "ideal_answers" / "partial_results.json"
    ]
    
    for results_file in result_files:
        if not results_file.exists():
            logger.warning(f"No se encontró el archivo de resultados en {results_file}")
            continue
            
        print(f"\n=== Evaluando {results_file.name} ===")
        metrics, discrepancies = evaluate_exact_match_from_results(results_file)
        
        # Imprimir resultados generales
        print(f"Total de preguntas evaluadas: {metrics['total_questions']}")
        print(f"Número de Exact Matches: {metrics['exact_matches']}")
        print(f"Tasa de Exact Match: {metrics['exact_match_rate']:.2%}")
        
        # Imprimir discrepancias
        if discrepancies:
            print("\nDiscrepancias encontradas (is_correct=1 pero no hay exact match):")
            print("-" * 80)
            for disc in discrepancies:
                print(f"Question ID: {disc['question_id']}")
                print(f"Pregunta: {disc['question']}")
                print(f"Etiqueta generada: {disc['generated_label']}")
                print(f"Referencia: {disc['reference_answer']}")
                print("-" * 80)
            print(f"\nTotal de discrepancias: {len(discrepancies)}")

if __name__ == "__main__":
    main()