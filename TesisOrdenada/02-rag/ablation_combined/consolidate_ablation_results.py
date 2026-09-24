import os
import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def load_results_from_folder(folder_path: Path) -> List[Dict[str, Any]]:
    """
    Carga resultados de una carpeta de ablation study.
    Puede ser desde un archivo summary.json o desde carpetas individuales.
    """
    results = []
    
    # Primero intentar cargar desde ablation_summary.json si existe
    summary_file = folder_path / "ablation_summary.json"
    if summary_file.exists():
        print(f"Cargando desde summary file: {summary_file}")
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'results' in data:
                results.extend(data['results'])
                return results
    
    # Si no hay summary, buscar en carpetas individuales
    print(f"Buscando experimentos individuales en: {folder_path}")
    for item in folder_path.iterdir():
        if item.is_dir() and item.name.startswith('alpha_'):
            metrics_file = item / "metrics.json"
            if metrics_file.exists():
                print(f"  Cargando: {metrics_file}")
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    results.append(metrics)
    
    return results

def get_experiment_description(alpha: float, num_similar: int) -> str:
    """
    Genera una descripción del experimento basada en los parámetros.
    """
    if alpha == 0.0:
        return f"Traditional RAG (alpha=0, similar={num_similar})"
    else:
        return f"Combined RAG (alpha={alpha}, similar={num_similar})"

def create_consolidated_csv(ablation_folders: List[Path], output_file: str):
    """
    Crea un CSV consolidado con todos los resultados de ablación Combined RAG.
    """
    all_results = []
    
    # Cargar resultados de todas las carpetas
    for folder in ablation_folders:
        if folder.exists():
            print(f"\nProcesando carpeta: {folder}")
            folder_results = load_results_from_folder(folder)
            
            # Añadir información de la carpeta fuente y descripción del experimento
            for result in folder_results:
                result['source_folder'] = folder.name
                result['experiment_description'] = get_experiment_description(
                    result.get('alpha', 0), 
                    result.get('num_similar_questions', 0)
                )
                all_results.append(result)
            
            print(f"  Cargados {len(folder_results)} experimentos")
        else:
            print(f"Carpeta no encontrada: {folder}")
    
    if not all_results:
        print("No se encontraron resultados para consolidar.")
        return
    
    # Crear DataFrame
    df = pd.DataFrame(all_results)
    
    # Ordenar por alpha y num_similar_questions
    df = df.sort_values(['alpha', 'num_similar_questions'])
    
    # Seleccionar y reordenar columnas para el CSV final
    columns_order = [
        'experiment_description',
        'alpha', 
        'num_similar_questions',
        'exact_match_rate',
        'llm_accuracy',
        'top_k',
        'total_questions',
        'avg_cosine_similarity',
        'avg_bert_f1',
        'avg_response_time',
        'avg_retrieval_time',
        'seed',
        'source_folder'
    ]
    
    # Filtrar solo las columnas que existen
    available_columns = [col for col in columns_order if col in df.columns]
    df_final = df[available_columns]
    
    # Guardar CSV
    df_final.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"\n✅ CSV consolidado guardado en: {output_file}")
    print(f"Total de experimentos Combined RAG: {len(df_final)}")
    
    # Mostrar resumen por valor de alpha
    print("\n📊 Resumen por valor de alpha:")
    alpha_summary = df_final.groupby('alpha').agg({
        'exact_match_rate': ['count', 'mean', 'std'],
        'llm_accuracy': ['mean', 'std'],
        'num_similar_questions': ['min', 'max']
    }).round(4)
    print(alpha_summary)
    
    # Mostrar resumen por número de preguntas similares
    print("\n📊 Resumen por número de preguntas similares:")
    similar_summary = df_final.groupby('num_similar_questions').agg({
        'exact_match_rate': ['count', 'mean', 'std'],
        'llm_accuracy': ['mean', 'std'],
        'alpha': ['min', 'max']
    }).round(4)
    print(similar_summary)
    
    # Mostrar los mejores resultados
    print("\n🏆 Top 5 experimentos por Exact Match:")
    top_exact = df_final.nlargest(5, 'exact_match_rate')[['experiment_description', 'exact_match_rate', 'llm_accuracy']]
    print(top_exact.to_string(index=False))
    
    print("\n🏆 Top 5 experimentos por LLM Accuracy:")
    top_llm = df_final.nlargest(5, 'llm_accuracy')[['experiment_description', 'exact_match_rate', 'llm_accuracy']]
    print(top_llm.to_string(index=False))
    
    return df_final

def create_analysis_summary(df: pd.DataFrame, output_file: str):
    """
    Crea un resumen de análisis con estadísticas detalladas.
    """
    analysis = []
    
    # Análisis por alpha
    for alpha in sorted(df['alpha'].unique()):
        alpha_data = df[df['alpha'] == alpha]
        
        analysis.append({
            'parameter': 'alpha',
            'value': alpha,
            'count': len(alpha_data),
            'avg_exact_match': alpha_data['exact_match_rate'].mean(),
            'std_exact_match': alpha_data['exact_match_rate'].std(),
            'avg_llm_accuracy': alpha_data['llm_accuracy'].mean(),
            'std_llm_accuracy': alpha_data['llm_accuracy'].std(),
            'best_exact_match': alpha_data['exact_match_rate'].max(),
            'best_llm_accuracy': alpha_data['llm_accuracy'].max()
        })
    
    # Análisis por num_similar_questions
    for num_similar in sorted(df['num_similar_questions'].unique()):
        similar_data = df[df['num_similar_questions'] == num_similar]
        
        analysis.append({
            'parameter': 'num_similar_questions',
            'value': num_similar,
            'count': len(similar_data),
            'avg_exact_match': similar_data['exact_match_rate'].mean(),
            'std_exact_match': similar_data['exact_match_rate'].std(),
            'avg_llm_accuracy': similar_data['llm_accuracy'].mean(),
            'std_llm_accuracy': similar_data['llm_accuracy'].std(),
            'best_exact_match': similar_data['exact_match_rate'].max(),
            'best_llm_accuracy': similar_data['llm_accuracy'].max()
        })
    
    analysis_df = pd.DataFrame(analysis)
    analysis_df.to_csv(output_file, index=False, float_format='%.6f')
    
    print(f"\n✅ Análisis detallado guardado en: {output_file}")

def main():
    """
    Función principal para consolidar todos los resultados de ablación Combined RAG.
    """
    print("🔄 Consolidando resultados de ablación Combined RAG...")
    
    # Definir las carpetas de ablation study
    base_dir = Path("ablation_combined")
    ablation_folders = [
        base_dir / "ablation_study_seed23_topk15_1748290709",
        base_dir / "ablation_study_seed23_topk15_1748271806"
    ]
    
    # Crear CSV consolidado
    output_file = "ablation_combined/consolidated_combined_rag_results.csv"
    df = create_consolidated_csv(ablation_folders, output_file)
    
    if df is not None:
        # Crear análisis detallado
        analysis_file = "ablation_combined/combined_rag_analysis_summary.csv"
        create_analysis_summary(df, analysis_file)
        
        print(f"\n🎉 Consolidación completada!")
        print(f"📁 Archivos generados:")
        print(f"  - {output_file}")
        print(f"  - {analysis_file}")
        
        print(f"\n📋 Resumen general:")
        print(f"  Total experimentos: {len(df)}")
        print(f"  Valores de alpha: {sorted(df['alpha'].unique())}")
        print(f"  Números de preguntas similares: {sorted(df['num_similar_questions'].unique())}")
        print(f"  Rango Exact Match: {df['exact_match_rate'].min():.4f} - {df['exact_match_rate'].max():.4f}")
        print(f"  Rango LLM Accuracy: {df['llm_accuracy'].min():.4f} - {df['llm_accuracy'].max():.4f}")

if __name__ == "__main__":
    main() 