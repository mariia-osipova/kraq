import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import sys

# Añadir src al path para importar módulos correctamente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar los módulos para calcular BERTScore
from src.utils.bert_score_utils import calculate_bert_score

def load_benchmark_results(model_type: str) -> Dict:
    """
    Carga los resultados del benchmark para un tipo de modelo específico.
    """
    filename = f"similitudes_{model_type}.json"
    filepath = os.path.join(project_root, "output", filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No se encontró el archivo de resultados para el modelo {model_type}: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error al decodificar el archivo JSON para el modelo {model_type}")
        return None

def analyze_threshold_relevance(results: Dict, threshold: float) -> Dict:
    """
    Analiza los resultados de un modelo específico según el threshold dado.
    Calcula BERTScore si no está presente en los resultados.
    """
    if not results:
        return None
    
    # Extraer pares de preguntas
    question_pairs = results.get('resultados', [])
    if not question_pairs:
        print("No se encontraron pares de preguntas en los resultados")
        return None
    
    # Preparar listas de preguntas para calcular BERTScore
    benchmark_questions = [pair.get('pregunta_original', '') for pair in question_pairs]
    representative_questions = [pair.get('pregunta_cercana', '') for pair in question_pairs]
    
    # Calcular BERTScore directamente
    print(f"Calculando BERTScore para {len(benchmark_questions)} pares de preguntas...")
    precision, recall, f1_scores = calculate_bert_score(benchmark_questions, representative_questions)
    
    # Convertir a lista Python (desde tensor)
    bert_scores = [float(score) for score in f1_scores]
    
    # Contar cuántos pares superan el threshold
    scores_array = np.array(bert_scores)
    above_threshold = scores_array >= threshold
    pairs_above_threshold = int(np.sum(above_threshold))
    
    # Calcular accuracy (proporción de pares que superan el threshold)
    total_pairs = len(question_pairs)
    threshold_accuracy = pairs_above_threshold / total_pairs if total_pairs > 0 else 0
    
    # Crear lista detallada de pares con sus scores
    detailed_pairs = []
    for i, (pair, score, is_above) in enumerate(zip(question_pairs, bert_scores, above_threshold)):
        detailed_pair = {
            'benchmark_question': pair.get('pregunta_original', ''),
            'representative_question': pair.get('pregunta_cercana', ''),
            'cosine_similarity': pair.get('similitud_coseno', 0),
            'bert_score': score,
            'above_threshold': bool(is_above)
        }
        detailed_pairs.append(detailed_pair)
    
    # Ordenar por BERTScore descendente
    detailed_pairs.sort(key=lambda x: x['bert_score'], reverse=True)
    
    return {
        'total_pairs': total_pairs,
        'pairs_above_threshold': pairs_above_threshold,
        'threshold_accuracy': threshold_accuracy,
        'model_type': results['info_coleccion']['tipo_modelo'],
        'collection_name': results['info_coleccion']['nombre_coleccion'],
        'bert_score_stats': {
            'mean': float(np.mean(bert_scores)),
            'median': float(np.median(bert_scores)),
            'max': float(np.max(bert_scores)),
            'min': float(np.min(bert_scores))
        },
        'question_pairs': detailed_pairs
    }

def calculate_and_save_threshold_analysis(threshold: float = 0.75):
    """
    Analiza los resultados de los tres modelos y guarda el análisis de threshold.
    """
    model_types = ['random', 'base', 'finetuned']
    results = {}
    
    print(f"=== Analizando relevancia con threshold {threshold} ===")
    
    # Cargar y analizar resultados para cada modelo
    for model_type in model_types:
        print(f"\nProcesando modelo: {model_type}")
        benchmark_results = load_benchmark_results(model_type)
        
        if benchmark_results:
            analysis = analyze_threshold_relevance(benchmark_results, threshold)
            if analysis:
                results[model_type] = analysis
                print(f"- BERTScore medio: {analysis['bert_score_stats']['mean']:.4f}")
                print(f"- Pares que superan threshold: {analysis['pairs_above_threshold']}/{analysis['total_pairs']}")
                print(f"- Accuracy de BERTScore: {analysis['threshold_accuracy']:.4f}")
    
    if results:
        # Preparar resultado final con detalle de todos los pares
        final_result = {
            'threshold_value': threshold,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                model_type: {
                    'total_pairs': results[model_type]['total_pairs'],
                    'pairs_above_threshold': results[model_type]['pairs_above_threshold'],
                    'threshold_accuracy': results[model_type]['threshold_accuracy'],
                    'bert_score_stats': results[model_type]['bert_score_stats'],
                    'collection_name': results[model_type]['collection_name']
                } for model_type in results
            },
            'detailed_results': {
                model_type: {
                    'question_pairs': results[model_type]['question_pairs']
                } for model_type in results
            }
        }
        
        # Guardar resultados
        output_path = os.path.join(project_root, "output", f"threshold_analysis_{int(threshold*100)}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\nResultados guardados en: {output_path}")
        
        # Imprimir comparativa final
        print("\n=== Comparativa de Modelos ===")
        print(f"Threshold utilizado: {threshold}")
        print("\nAccuracy de BERTScore por modelo:")
        for model_type in results:
            accuracy = results[model_type]['threshold_accuracy']
            pairs = results[model_type]['pairs_above_threshold']
            total = results[model_type]['total_pairs']
            mean_score = results[model_type]['bert_score_stats']['mean']
            print(f"{model_type.upper()}: {accuracy:.4f} ({pairs}/{total} pares) - BERTScore medio: {mean_score:.4f}")
        
        # Mostrar algunos ejemplos de pares que superan el threshold
        print("\n=== Ejemplos de Pares que Superan el Threshold ===")
        for model_type in results:
            print(f"\nModelo {model_type.upper()}:")
            above_threshold_pairs = [p for p in results[model_type]['question_pairs'] if p['above_threshold']][:3]
            for i, pair in enumerate(above_threshold_pairs):
                print(f"\nPar {i+1}:")
                print(f"Benchmark: {pair['benchmark_question']}")
                print(f"Representativa: {pair['representative_question']}")
                print(f"BERTScore: {pair['bert_score']:.4f}")
                print(f"Similitud Coseno: {pair['cosine_similarity']:.4f}")
                print("-" * 80)
        
        return final_result
    
    return None

if __name__ == "__main__":
    import sys
    import time
    
    # Puedes ajustar el threshold aquí
    THRESHOLD = 0.73
    
    print("Comenzando análisis de threshold de relevancia")
    results = calculate_and_save_threshold_analysis(THRESHOLD)
    
    if results:
        print("\nAnálisis completado exitosamente")
