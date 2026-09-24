import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_results_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Carga los resultados desde un archivo JSON.
    
    Args:
        file_path: Ruta al archivo JSON
        
    Returns:
        Lista de resultados
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando archivo {file_path}: {e}")
        return []

def extract_speculative_times(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extrae los tiempos del Speculative RAG.
    
    Args:
        results: Lista de resultados del benchmark
        
    Returns:
        Diccionario con listas de tiempos
    """
    times = {
        'total_time': [],
        'sequential_time': [],
        'parallel_estimate': []
    }
    
    for result in results:
        if 'total_time' in result and 'execution_times' in result:
            exec_times = result['execution_times']
            
            # Tiempo total
            times['total_time'].append(result['total_time'])
            
            # Tiempo secuencial (drafting + verificación)
            draft_seq = exec_times.get('draft_sequential_time', 0)
            verify_seq = exec_times.get('verify_sequential_time', 0)
            times['sequential_time'].append(draft_seq + verify_seq)
            
            # Tiempo paralelo estimado (drafting + verificación)
            draft_par = exec_times.get('draft_parallel_estimate', 0)
            verify_par = exec_times.get('verify_parallel_estimate', 0)
            times['parallel_estimate'].append(draft_par + verify_par)
    
    return times

def extract_modified_times(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Extrae los tiempos del Modified RAG.
    
    Args:
        results: Lista de resultados del benchmark
        
    Returns:
        Diccionario con listas de tiempos
    """
    times = {
        'total_time': [],
        'sequential_time': [],
        'parallel_estimate': [],
        'inbedder_time': [],
        'total_time_minus_inbedder': [],
        'sequential_time_minus_inbedder': [],
        'parallel_estimate_minus_inbedder': []
    }
    
    for result in results:
        if 'total_time' in result:
            # Tiempos básicos
            total_time = result['total_time']
            draft_seq = result.get('draft_sequential_time', 0)
            verify_seq = result.get('verify_sequential_time', 0)
            draft_par = result.get('draft_parallel_estimate', 0)
            verify_par = result.get('verify_parallel_estimate', 0)
            inbedder_time = result.get('inbedder_time', 0)
            
            # Tiempos originales
            times['total_time'].append(total_time)
            times['sequential_time'].append(draft_seq + verify_seq)
            times['parallel_estimate'].append(draft_par + verify_par)
            times['inbedder_time'].append(inbedder_time)
            
            # Tiempos menos inbedder
            times['total_time_minus_inbedder'].append(total_time - inbedder_time)
            times['sequential_time_minus_inbedder'].append((draft_seq + verify_seq) - inbedder_time)
            times['parallel_estimate_minus_inbedder'].append((draft_par + verify_par) - inbedder_time)
    
    return times

def calculate_statistics(times: List[float]) -> Dict[str, float]:
    """
    Calcula estadísticas básicas para una lista de tiempos.
    
    Args:
        times: Lista de tiempos
        
    Returns:
        Diccionario con estadísticas
    """
    if not times:
        return {'median': 0, 'mean': 0, 'min': 0, 'max': 0, 'std': 0}
    
    return {
        'median': statistics.median(times),
        'mean': statistics.mean(times),
        'min': min(times),
        'max': max(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0
    }

def generate_time_comparison_report() -> Dict[str, Any]:
    """
    Genera un reporte completo de comparación de tiempos entre Speculative RAG y Modified RAG
    usando los datos de ablation runs.
    
    Returns:
        Diccionario con el reporte completo
    """
    # Rutas a los archivos de resultados en ablation runs
    base_path = Path("ablation_runs_vanterior/spec_k2_m5_topk10_1747853341")
    speculative_path = base_path / "speculative/exact_answers/full_results.json"
    modified_path = base_path / "modified/answers/full_results.json"
    
    # Cargar resultados
    logger.info("Cargando resultados del Speculative RAG desde ablation runs...")
    speculative_results = load_results_from_file(speculative_path)
    
    logger.info("Cargando resultados del Modified RAG desde ablation runs...")
    modified_results = load_results_from_file(modified_path)
    
    if not speculative_results or not modified_results:
        logger.error("No se pudieron cargar los resultados")
        return {}
    
    # Extraer tiempos
    logger.info("Extrayendo tiempos del Speculative RAG...")
    spec_times = extract_speculative_times(speculative_results)
    
    logger.info("Extrayendo tiempos del Modified RAG...")
    mod_times = extract_modified_times(modified_results)
    
    # Calcular estadísticas
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'speculative_questions': len(speculative_results),
            'modified_questions': len(modified_results),
            'speculative_file': str(speculative_path),
            'modified_file': str(modified_path),
            'source': 'ablation_runs_vanterior/spec_k2_m5_topk10_1747853341'
        },
        'speculative_rag': {
            'total_time': calculate_statistics(spec_times['total_time']),
            'sequential_time': calculate_statistics(spec_times['sequential_time']),
            'parallel_estimate': calculate_statistics(spec_times['parallel_estimate'])
        },
        'modified_rag': {
            'total_time': calculate_statistics(mod_times['total_time']),
            'sequential_time': calculate_statistics(mod_times['sequential_time']),
            'parallel_estimate': calculate_statistics(mod_times['parallel_estimate']),
            'inbedder_time': calculate_statistics(mod_times['inbedder_time']),
            'total_time_minus_inbedder': calculate_statistics(mod_times['total_time_minus_inbedder']),
            'sequential_time_minus_inbedder': calculate_statistics(mod_times['sequential_time_minus_inbedder']),
            'parallel_estimate_minus_inbedder': calculate_statistics(mod_times['parallel_estimate_minus_inbedder'])
        }
    }
    
    # Calcular comparaciones basadas en medianas
    spec_median_total = report['speculative_rag']['total_time']['mean']
    spec_median_sequential = report['speculative_rag']['sequential_time']['mean']
    spec_median_parallel = report['speculative_rag']['parallel_estimate']['mean']
    
    mod_median_total = report['modified_rag']['total_time']['mean']
    mod_median_total_minus_inbedder = report['modified_rag']['total_time_minus_inbedder']['mean']
    mod_median_sequential = report['modified_rag']['sequential_time']['mean']
    mod_median_sequential_minus_inbedder = report['modified_rag']['sequential_time_minus_inbedder']['mean']
    mod_median_parallel = report['modified_rag']['parallel_estimate']['mean']
    mod_median_parallel_minus_inbedder = report['modified_rag']['parallel_estimate_minus_inbedder']['mean']
    
    # Cálculo del tiempo estimado según la fórmula del usuario
    # Para Speculative: tiempo total - tiempo secuencial + tiempo paralelo estimado
    spec_estimated_time = spec_median_total - spec_median_sequential + spec_median_parallel
    
    # Para Modified: tiempo total - tiempo secuencial + tiempo paralelo estimado (todo menos inbedder)
    mod_estimated_time = mod_median_total - mod_median_sequential + mod_median_parallel_minus_inbedder
    
    report['comparison'] = {
        'speculative_estimated_time': spec_estimated_time,
        'modified_estimated_time': mod_estimated_time,
        'time_difference': spec_estimated_time - mod_estimated_time,
        'speedup_factor': spec_estimated_time / mod_estimated_time if mod_estimated_time > 0 else 0,
        'percentage_improvement': ((spec_estimated_time - mod_estimated_time) / spec_estimated_time * 100) if spec_estimated_time > 0 else 0
    }
    
    # Comparaciones adicionales
    report['detailed_comparisons'] = {
        'total_time': {
            'speculative_median': spec_median_total,
            'modified_median': mod_median_total,
            'modified_median_minus_inbedder': mod_median_total_minus_inbedder,
            'improvement_vs_total': ((spec_median_total - mod_median_total) / spec_median_total * 100) if spec_median_total > 0 else 0,
            'improvement_vs_minus_inbedder': ((spec_median_total - mod_median_total_minus_inbedder) / spec_median_total * 100) if spec_median_total > 0 else 0
        },
        'sequential_time': {
            'speculative_median': spec_median_sequential,
            'modified_median': mod_median_sequential,
            'modified_median_minus_inbedder': mod_median_sequential_minus_inbedder,
            'improvement_vs_total': ((spec_median_sequential - mod_median_sequential) / spec_median_sequential * 100) if spec_median_sequential > 0 else 0,
            'improvement_vs_minus_inbedder': ((spec_median_sequential - mod_median_sequential_minus_inbedder) / spec_median_sequential * 100) if spec_median_sequential > 0 else 0
        },
        'parallel_estimate': {
            'speculative_median': spec_median_parallel,
            'modified_median': mod_median_parallel,
            'modified_median_minus_inbedder': mod_median_parallel_minus_inbedder,
            'improvement_vs_total': ((spec_median_parallel - mod_median_parallel) / spec_median_parallel * 100) if spec_median_parallel > 0 else 0,
            'improvement_vs_minus_inbedder': ((spec_median_parallel - mod_median_parallel_minus_inbedder) / spec_median_parallel * 100) if spec_median_parallel > 0 else 0
        }
    }
    
    return report

def print_report_summary(report: Dict[str, Any]):
    """
    Imprime un resumen del reporte de comparación de tiempos.
    
    Args:
        report: Diccionario con el reporte completo
    """
    if not report:
        print("No se pudo generar el reporte")
        return
    
    print("\n" + "="*80)
    print("REPORTE DE COMPARACIÓN DE TIEMPOS - ABLATION RUNS")
    print("="*80)
    
    metadata = report['metadata']
    print(f"\nMetadatos:")
    print(f"- Timestamp: {metadata['timestamp']}")
    print(f"- Fuente: {metadata['source']}")
    print(f"- Preguntas Speculative RAG: {metadata['speculative_questions']}")
    print(f"- Preguntas Modified RAG: {metadata['modified_questions']}")
    
    print(f"\n" + "-"*50)
    print("TIEMPO ESTIMADO SEGÚN FÓRMULA DEL USUARIO")
    print("-"*50)
    
    comparison = report['comparison']
    print(f"Speculative RAG (mediana total - secuencial + paralelo): {comparison['speculative_estimated_time']:.4f}s")
    print(f"Modified RAG (mediana total - secuencial + paralelo, sin inbedder): {comparison['modified_estimated_time']:.4f}s")
    print(f"Diferencia de tiempo: {comparison['time_difference']:.4f}s")
    print(f"Factor de speedup: {comparison['speedup_factor']:.2f}x")
    print(f"Mejora porcentual: {comparison['percentage_improvement']:.2f}%")
    
    print(f"\n" + "-"*50)
    print("COMPARACIONES DETALLADAS (MEDIANAS)")
    print("-"*50)
    
    detailed = report['detailed_comparisons']
    
    print(f"\nTiempo Total:")
    print(f"- Speculative RAG: {detailed['total_time']['speculative_median']:.4f}s")
    print(f"- Modified RAG: {detailed['total_time']['modified_median']:.4f}s")
    print(f"- Modified RAG (sin inbedder): {detailed['total_time']['modified_median_minus_inbedder']:.4f}s")
    print(f"- Mejora vs total: {detailed['total_time']['improvement_vs_total']:.2f}%")
    print(f"- Mejora vs sin inbedder: {detailed['total_time']['improvement_vs_minus_inbedder']:.2f}%")
    
    print(f"\nTiempo Secuencial (Drafting + Verificación):")
    print(f"- Speculative RAG: {detailed['sequential_time']['speculative_median']:.4f}s")
    print(f"- Modified RAG: {detailed['sequential_time']['modified_median']:.4f}s")
    print(f"- Modified RAG (sin inbedder): {detailed['sequential_time']['modified_median_minus_inbedder']:.4f}s")
    print(f"- Mejora vs total: {detailed['sequential_time']['improvement_vs_total']:.2f}%")
    print(f"- Mejora vs sin inbedder: {detailed['sequential_time']['improvement_vs_minus_inbedder']:.2f}%")
    
    print(f"\nTiempo Paralelo Estimado (Drafting + Verificación):")
    print(f"- Speculative RAG: {detailed['parallel_estimate']['speculative_median']:.4f}s")
    print(f"- Modified RAG: {detailed['parallel_estimate']['modified_median']:.4f}s")
    print(f"- Modified RAG (sin inbedder): {detailed['parallel_estimate']['modified_median_minus_inbedder']:.4f}s")
    print(f"- Mejora vs total: {detailed['parallel_estimate']['improvement_vs_total']:.2f}%")
    print(f"- Mejora vs sin inbedder: {detailed['parallel_estimate']['improvement_vs_minus_inbedder']:.2f}%")

def main():
    """
    Función principal para generar el reporte de comparación de tiempos usando ablation runs.
    """
    logger.info("Iniciando generación del reporte de comparación de tiempos desde ablation runs...")
    
    # Generar reporte
    report = generate_time_comparison_report()
    
    if not report:
        logger.error("No se pudo generar el reporte")
        return
    
    # Guardar reporte en JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"time_comparison_ablation_report_{timestamp}.json")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Reporte guardado en: {report_file}")
    
    # Imprimir resumen
    print_report_summary(report)
    
    print(f"\n" + "="*80)
    print(f"Reporte completo guardado en: {report_file}")
    print("="*80)

if __name__ == "__main__":
    main() 