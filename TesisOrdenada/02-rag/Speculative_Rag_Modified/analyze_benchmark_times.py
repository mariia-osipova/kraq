import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def analyze_benchmark_results(results_path):
    """
    Analiza los tiempos de ejecución de los resultados del benchmark y genera estadísticas.
    
    Args:
        results_path: Ruta al archivo JSON con los resultados del benchmark
    """
    # Cargar los resultados
    print(f"Analizando resultados de: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Extraer todos los tiempos totales
    total_times = [result.get('total_time', 0) for result in results]
    
    # Calcular estadísticas
    median_time = np.median(total_times)
    mean_time = np.mean(total_times)
    min_time = np.min(total_times)
    max_time = np.max(total_times)
    percentile_25 = np.percentile(total_times, 25)
    percentile_75 = np.percentile(total_times, 75)
    std_time = np.std(total_times)
    
    # Extraer también otros tiempos importantes
    inbedder_times = [result.get('inbedder_time', 0) for result in results]
    retrieval_times = [result.get('retrieval_time', 0) for result in results]
    similar_question_times = [result.get('similar_question_time', 0) for result in results]
    clustering_times = [result.get('clustering_time', 0) for result in results]
    draft_sequential_times = [result.get('draft_sequential_time', 0) for result in results]
    draft_parallel_times = [result.get('draft_parallel_estimate', 0) for result in results]
    verify_sequential_times = [result.get('verify_sequential_time', 0) for result in results]
    verify_parallel_times = [result.get('verify_parallel_estimate', 0) for result in results]
    
    # Calcular medianas para cada tipo de tiempo
    median_inbedder = np.median(inbedder_times)
    median_retrieval = np.median(retrieval_times)
    median_similar_question = np.median(similar_question_times)
    median_clustering = np.median(clustering_times)
    median_draft_sequential = np.median(draft_sequential_times)
    median_draft_parallel = np.median(draft_parallel_times)
    median_verify_sequential = np.median(verify_sequential_times)
    median_verify_parallel = np.median(verify_parallel_times)
    
    # Calcular estadísticas de precisión
    correct_answers = sum(result.get('is_correct', 0) for result in results)
    total_questions = len(results)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Crear directorio para guardar resultados del análisis
    output_dir = Path(results_path).parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Guardar estadísticas en un archivo JSON
    stats = {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "time_statistics": {
            "median_time": median_time,
            "mean_time": mean_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_time": std_time,
            "percentile_25": percentile_25,
            "percentile_75": percentile_75
        },
        "median_component_times": {
            "inbedder_time": median_inbedder,
            "retrieval_time": median_retrieval,
            "similar_question_time": median_similar_question,
            "clustering_time": median_clustering,
            "draft_sequential_time": median_draft_sequential,
            "draft_parallel_estimate": median_draft_parallel,
            "verify_sequential_time": median_verify_sequential,
            "verify_parallel_estimate": median_verify_parallel
        }
    }
    
    with open(output_dir / "time_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Guardar también en formato legible
    with open(output_dir / "time_stats.txt", 'w', encoding='utf-8') as f:
        f.write("ESTADÍSTICAS DE TIEMPOS DE EJECUCIÓN\n")
        f.write("==================================\n\n")
        
        f.write(f"Total de preguntas: {total_questions}\n")
        f.write(f"Respuestas correctas: {correct_answers}\n")
        f.write(f"Precisión: {accuracy:.2%}\n\n")
        
        f.write("Estadísticas de tiempo total (segundos):\n")
        f.write(f"- Mediana: {median_time:.2f}s\n")
        f.write(f"- Media: {mean_time:.2f}s\n")
        f.write(f"- Mínimo: {min_time:.2f}s\n")
        f.write(f"- Máximo: {max_time:.2f}s\n")
        f.write(f"- Desviación estándar: {std_time:.2f}s\n")
        f.write(f"- Percentil 25: {percentile_25:.2f}s\n")
        f.write(f"- Percentil 75: {percentile_75:.2f}s\n\n")
        
        f.write("Medianas por componente (segundos):\n")
        f.write(f"- InBedder: {median_inbedder:.2f}s\n")
        f.write(f"- Recuperación de documentos: {median_retrieval:.2f}s\n")
        f.write(f"- Búsqueda de pregunta similar: {median_similar_question:.2f}s\n")
        f.write(f"- Clustering: {median_clustering:.2f}s\n")
        f.write(f"- Drafting secuencial: {median_draft_sequential:.2f}s\n")
        f.write(f"- Drafting paralelo (estimado): {median_draft_parallel:.2f}s\n")
        f.write(f"- Verificación secuencial: {median_verify_sequential:.2f}s\n")
        f.write(f"- Verificación paralela (estimada): {median_verify_parallel:.2f}s\n")
    
    # Crear un histograma de los tiempos totales
    plt.figure(figsize=(10, 6))
    plt.hist(total_times, bins=30, alpha=0.7, color='blue')
    plt.axvline(median_time, color='red', linestyle='dashed', linewidth=2, label=f'Mediana: {median_time:.2f}s')
    plt.axvline(mean_time, color='green', linestyle='dashed', linewidth=2, label=f'Media: {mean_time:.2f}s')
    plt.xlabel('Tiempo de ejecución (segundos)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de tiempos de ejecución')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_distribution.png", dpi=300)
    
    # Crear un gráfico de barras para las medianas de los componentes
    component_names = [
        'InBedder', 'Recuperación', 'Pregunta Similar', 'Clustering',
        'Drafting Sec.', 'Drafting Par.', 'Verificación Sec.', 'Verificación Par.'
    ]
    component_times = [
        median_inbedder, median_retrieval, median_similar_question, median_clustering,
        median_draft_sequential, median_draft_parallel, median_verify_sequential, median_verify_parallel
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(component_names, component_times, color='skyblue')
    
    # Añadir etiquetas con los valores
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Componente')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Mediana de tiempos por componente')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "component_times.png", dpi=300)
    
    # Mostrar resumen
    print("\nRESUMEN DE ESTADÍSTICAS:")
    print(f"Mediana de tiempo total: {median_time:.2f} segundos")
    print(f"Media de tiempo total: {mean_time:.2f} segundos")
    print(f"Precisión: {accuracy:.2%} ({correct_answers}/{total_questions})")
    print(f"\nResultados guardados en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza tiempos de ejecución de resultados de benchmark")
    parser.add_argument("--results", type=str, 
                        default="Speculative_Rag_Modified/BenchMarks/benchmark_results_modified_qdrant_qfinetuned/answers/full_results.json",  # antes: ruta absoluta de la maquina original
                        help="Ruta al archivo JSON con los resultados del benchmark")
    
    args = parser.parse_args()
    analyze_benchmark_results(args.results)
