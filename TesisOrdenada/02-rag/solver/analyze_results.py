#!/usr/bin/env python3
"""
Script para analizar los resultados de las comparaciones de exact match
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_final_summary(results_dir: Path = None) -> Dict[str, Any]:
    """Carga el resumen final de los resultados"""
    if results_dir is None:
        results_dir = Path("solver_exact_match_runs")
    
    summary_path = results_dir / "final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No se encontró {summary_path}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)

def analyze_results(summary: Dict[str, Any]) -> pd.DataFrame:
    """Analiza los resultados y crea un DataFrame"""
    results = summary.get("all_results", [])
    
    df_data = []
    for result in results:
        if "error" not in result:
            df_data.append({
                "seed": result["seed"],
                "traditional_exact_match": result["traditional_exact_match"],
                "combined_exact_match": result["combined_exact_match"],
                "difference": result["difference"],
                "combined_better": result["combined_better"]
            })
    
    return pd.DataFrame(df_data)

def generate_report(results_dir: Path = None):
    """Genera un reporte completo de los resultados"""
    if results_dir is None:
        results_dir = Path("solver_exact_match_runs")
    
    # Cargar datos
    summary = load_final_summary(results_dir)
    df = analyze_results(summary)
    
    if df.empty:
        print("No hay datos para analizar")
        return
    
    # Crear reporte
    report_path = results_dir / "analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Análisis de Resultados - Traditional vs Combined RAG\n\n")
        
        # Información general
        f.write("## Información General\n\n")
        f.write(f"- Total de intentos: {summary['total_attempts']}\n")
        f.write(f"- Preguntas por intento: {summary['n_questions_per_attempt']}\n")
        f.write(f"- Top-K utilizado: {summary['top_k']}\n")
        f.write(f"- Éxito encontrado: {'Sí' if summary['success'] else 'No'}\n")
        
        if summary['success']:
            f.write(f"- Seed exitosa: {summary['successful_seed']}\n")
        
        f.write("\n")
        
        # Estadísticas descriptivas
        f.write("## Estadísticas Descriptivas\n\n")
        f.write("### Traditional RAG Exact Match\n")
        f.write(f"- Promedio: {df['traditional_exact_match'].mean():.4f}\n")
        f.write(f"- Mediana: {df['traditional_exact_match'].median():.4f}\n")
        f.write(f"- Desviación estándar: {df['traditional_exact_match'].std():.4f}\n")
        f.write(f"- Mínimo: {df['traditional_exact_match'].min():.4f}\n")
        f.write(f"- Máximo: {df['traditional_exact_match'].max():.4f}\n\n")
        
        f.write("### Combined RAG Exact Match\n")
        f.write(f"- Promedio: {df['combined_exact_match'].mean():.4f}\n")
        f.write(f"- Mediana: {df['combined_exact_match'].median():.4f}\n")
        f.write(f"- Desviación estándar: {df['combined_exact_match'].std():.4f}\n")
        f.write(f"- Mínimo: {df['combined_exact_match'].min():.4f}\n")
        f.write(f"- Máximo: {df['combined_exact_match'].max():.4f}\n\n")
        
        f.write("### Diferencias (Combined - Traditional)\n")
        f.write(f"- Promedio: {df['difference'].mean():.4f}\n")
        f.write(f"- Mediana: {df['difference'].median():.4f}\n")
        f.write(f"- Desviación estándar: {df['difference'].std():.4f}\n")
        f.write(f"- Mínimo: {df['difference'].min():.4f}\n")
        f.write(f"- Máximo: {df['difference'].max():.4f}\n\n")
        
        # Casos donde Combined es mejor
        better_cases = df[df['combined_better']].shape[0]
        f.write(f"### Casos donde Combined RAG es mejor\n")
        f.write(f"- Número de casos: {better_cases}\n")
        f.write(f"- Porcentaje: {(better_cases / len(df)) * 100:.2f}%\n\n")
        
        # Top 5 mejores y peores casos
        f.write("## Top 5 Mejores Casos (mayor diferencia)\n\n")
        top_5_best = df.nlargest(5, 'difference')
        for idx, row in top_5_best.iterrows():
            f.write(f"- Seed {row['seed']}: Traditional={row['traditional_exact_match']:.4f}, Combined={row['combined_exact_match']:.4f}, Diff=+{row['difference']:.4f}\n")
        
        f.write("\n## Top 5 Peores Casos (menor diferencia)\n\n")
        top_5_worst = df.nsmallest(5, 'difference')
        for idx, row in top_5_worst.iterrows():
            f.write(f"- Seed {row['seed']}: Traditional={row['traditional_exact_match']:.4f}, Combined={row['combined_exact_match']:.4f}, Diff={row['difference']:.4f}\n")
    
    print(f"Reporte generado: {report_path}")
    
    # Generar gráficos si matplotlib está disponible
    try:
        generate_plots(df, results_dir)
    except ImportError:
        print("Matplotlib no disponible, saltando generación de gráficos")

def generate_plots(df: pd.DataFrame, results_dir: Path):
    """Genera gráficos de análisis"""
    plt.style.use('default')
    
    # Configurar el tamaño de la figura
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis Traditional vs Combined RAG - Exact Match', fontsize=16)
    
    # Gráfico 1: Distribución de exact match rates
    axes[0, 0].hist(df['traditional_exact_match'], alpha=0.7, label='Traditional RAG', bins=20)
    axes[0, 0].hist(df['combined_exact_match'], alpha=0.7, label='Combined RAG', bins=20)
    axes[0, 0].set_xlabel('Exact Match Rate')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Exact Match Rates')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Scatter plot
    axes[0, 1].scatter(df['traditional_exact_match'], df['combined_exact_match'], alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Línea de igualdad')
    axes[0, 1].set_xlabel('Traditional RAG Exact Match')
    axes[0, 1].set_ylabel('Combined RAG Exact Match')
    axes[0, 1].set_title('Traditional vs Combined RAG')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Distribución de diferencias
    axes[1, 0].hist(df['difference'], bins=20, alpha=0.7, color='green')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='Sin diferencia')
    axes[1, 0].set_xlabel('Diferencia (Combined - Traditional)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].set_title('Distribución de Diferencias')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Serie temporal de diferencias
    axes[1, 1].plot(df['seed'], df['difference'], marker='o', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', label='Sin diferencia')
    axes[1, 1].set_xlabel('Seed')
    axes[1, 1].set_ylabel('Diferencia (Combined - Traditional)')
    axes[1, 1].set_title('Diferencias por Seed')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "analysis_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficos guardados: {results_dir / 'analysis_plots.png'}")

def main():
    """Función principal"""
    results_dir = Path("solver_exact_match_runs")
    
    if not results_dir.exists():
        print(f"Directorio {results_dir} no existe")
        return
    
    try:
        generate_report(results_dir)
        print("Análisis completado")
    except Exception as e:
        print(f"Error durante el análisis: {e}")

if __name__ == "__main__":
    main() 