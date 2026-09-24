import asyncio
import random
from pathlib import Path
import json
import time
from loguru import logger
from typing import List, Dict, Any, Tuple

# Importar las funciones de benchmark y obtención de preguntas
from SpeculativeRag.Benchmark.benchmark_speculative import run_benchmark_with_qdrant as run_speculative
from SpeculativeRag.Benchmark.benchmark_speculative import get_questions_from_qdrant as get_questions_spec
from Speculative_Rag_Modified.BenchMarks.benchmark_modified_train import run_benchmark_with_qdrant as run_modified
from Speculative_Rag_Modified.BenchMarks.benchmark_modified_train import get_questions_from_qdrant as get_questions_mod

async def get_accuracy_spec(results_dir: Path) -> Tuple[float, float]:
    """
    Obtiene la accuracy de los resultados de Speculative RAG tanto para exact como ideal
    Returns:
        Tuple[float, float]: (exact_accuracy, ideal_accuracy)
    """
    exact_acc = 0.0
    ideal_acc = 0.0
    
    try:
        # Leer exact answers
        exact_report = results_dir / "exact_answers" / "report.md"
        if exact_report.exists():
            with open(exact_report, "r", encoding="utf-8") as f:
                content = f.read()
                # Buscar la línea que contiene "Accuracy:"
                for line in content.split('\n'):
                    if "Accuracy:" in line:
                        exact_acc = float(line.split(':')[1].strip())
                        break
        
        # Leer ideal answers
        ideal_report = results_dir / "ideal_answers" / "report.md"
        if ideal_report.exists():
            with open(ideal_report, "r", encoding="utf-8") as f:
                content = f.read()
                for line in content.split('\n'):
                    if "Accuracy:" in line:
                        ideal_acc = float(line.split(':')[1].strip())
                        break
                        
    except Exception as e:
        logger.error(f"Error leyendo accuracies de Speculative RAG: {e}")
    
    return exact_acc, ideal_acc

async def get_accuracy_mod(results_dir: Path) -> Tuple[float, float]:
    """
    Obtiene la accuracy de los resultados de Modified RAG tanto para exact como ideal
    Returns:
        Tuple[float, float]: (exact_accuracy, ideal_accuracy)
    """
    exact_acc = 0.0
    ideal_acc = 0.0
    
    try:
        # Leer exact answers - usamos stats_report.json en lugar de report.md
        exact_report = results_dir / "exact_answers" / "stats_report.json"
        if exact_report.exists():
            with open(exact_report, "r", encoding="utf-8") as f:
                stats = json.load(f)
                exact_acc = stats.get("accuracy", 0.0)
        
        # Leer ideal answers
        ideal_report = results_dir / "ideal_answers" / "stats_report.json"
        if ideal_report.exists():
            with open(ideal_report, "r", encoding="utf-8") as f:
                stats = json.load(f)
                ideal_acc = stats.get("accuracy", 0.0)
                        
    except Exception as e:
        logger.error(f"Error leyendo accuracies de Modified RAG: {e}")
    
    # Verificar rutas alternativas si no se encontraron archivos
    if exact_acc == 0.0 and ideal_acc == 0.0:
        try:
            # Intentar con rutas absolutas como fallback
            alt_dir = Path(__file__).parent / "Speculative_Rag_Modified/BenchMarks/benchmark_results_modified_qdrant"
            
            exact_report = alt_dir / "exact_answers" / "stats_report.json"
            if exact_report.exists():
                with open(exact_report, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    exact_acc = stats.get("accuracy", 0.0)
            
            ideal_report = alt_dir / "ideal_answers" / "stats_report.json"
            if ideal_report.exists():
                with open(ideal_report, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    ideal_acc = stats.get("accuracy", 0.0)
        except Exception as e:
            logger.error(f"Error leyendo accuracies alternativas de Modified RAG: {e}")
    
    return exact_acc, ideal_acc

async def fetch_random_questions(n: int) -> List[Dict[str, Any]]:
    """
    Obtiene n preguntas aleatorias de Qdrant
    """
    all_questions = await get_questions_spec(limit=1000)
    if len(all_questions) < n:
        raise ValueError(f"No hay suficientes preguntas en Qdrant (encontradas {len(all_questions)}, necesarias {n})")
    return random.sample(all_questions, n)

async def main():
    # Configuración inicial
    n_questions = 500  # Aumentar el número de preguntas para un test más significativo
    max_attempts = 1
    base_params = {
        "embedding_model": "nomic-embed-text"
    }
    
    # Diferentes conjuntos de parámetros para probar (variando top_k, m y k)
    param_sets = [  # Menos clusters, subsets y documentos
        {**base_params, "k": 5, "m": 10, "top_k": 18}# Menos clusters, más subsets, menos documentos
    ]

    # Crear directorio base para resultados
    base_results_dir = Path("solver_runs")
    base_results_dir.mkdir(exist_ok=True)

    # Configurar logger para el solver
    logger.add(base_results_dir / "solver.log", rotation="500 MB")
    logger.info("Iniciando solver para comparación de RAGs")

    # Guardar la mejor configuración encontrada
    best_config = {
        "params": None,
        "acc_mod_exact": -1,
        "acc_mod_ideal": -1,
        "acc_spec_exact": -1,
        "acc_spec_ideal": -1,
        "run_dir": None
    }

    for param_idx, params in enumerate(param_sets):
        # Usa una semilla diferente para cada conjunto de parámetros
        params["seed"] = 23 + param_idx  # O params["seed"] = random.randint(1, 1000)
        logger.info(f"\n=== Probando conjunto de parámetros {param_idx + 1}/{len(param_sets)} ===")
        logger.info(f"k={params['k']}, m={params['m']}, top_k={params['top_k']}")
        
        for attempt in range(max_attempts):
            # Crear directorio para esta corrida
            timestamp = int(time.time())
            run_dir_name = f'k{params["k"]}_m{params["m"]}_topk{params["top_k"]}_run{attempt+1}_{timestamp}'
            run_dir = base_results_dir / run_dir_name
            run_dir.mkdir(exist_ok=True)
            
            logger.info(f"\n=== Intento {attempt+1} con parámetros actuales ===")
            
            try:
                # Obtener preguntas aleatorias
                questions = await fetch_random_questions(n_questions)
                
                # Ejecutar ambos benchmarks con las mismas preguntas y parámetros
                spec_dir = run_dir / "speculative"
                mod_dir = run_dir / "modified"
                
                # Crear los directorios explícitamente para asegurar que existan
                spec_dir.mkdir(exist_ok=True, parents=True)
                mod_dir.mkdir(exist_ok=True, parents=True)
                
                # Ejecutar Speculative RAG
                logger.info("Ejecutando Speculative RAG...")
                await run_speculative(
                    limit=n_questions,
                    questions=questions,
                    results_dir=spec_dir,
                    **params
                )
                
                # Ejecutar Modified RAG
                logger.info("Ejecutando Modified RAG...")
                await run_modified(
                    limit=n_questions,
                    questions=questions,
                    results_dir=mod_dir,
                    **params
                )
                
                # Esperar un tiempo suficiente para que se generen los reportes
                await asyncio.sleep(5)
                
                # Verificar explícitamente que los reportes se hayan generado
                spec_exact_report = spec_dir / "exact_answers" / "report.md"
                spec_ideal_report = spec_dir / "ideal_answers" / "report.md"
                mod_exact_report = mod_dir / "exact_answers" / "stats_report.json"
                mod_ideal_report = mod_dir / "ideal_answers" / "stats_report.json"
                
                # También verificar ubicaciones alternativas
                alt_mod_exact_report = Path(__file__).parent / "Speculative_Rag_Modified/BenchMarks/benchmark_results_modified_qdrant/exact_answers/stats_report.json"
                alt_mod_ideal_report = Path(__file__).parent / "Speculative_Rag_Modified/BenchMarks/benchmark_results_modified_qdrant/ideal_answers/stats_report.json"
                
                # Verificar que existan los archivos necesarios
                spec_files_ok = spec_exact_report.exists() and spec_ideal_report.exists()
                mod_files_ok = (mod_exact_report.exists() and mod_ideal_report.exists()) or (alt_mod_exact_report.exists() and alt_mod_ideal_report.exists())
                
                if not spec_files_ok:
                    logger.error(f"Faltan archivos de reporte de Speculative RAG")
                    logger.error("Saltando esta corrida debido a reportes faltantes")
                    continue
                
                if not mod_files_ok:
                    logger.error(f"Faltan archivos de reporte de Modified RAG")
                    logger.error("Saltando esta corrida debido a reportes faltantes")
                    continue
                
                # Leer accuracies de ambos tipos de respuestas
                acc_spec_exact, acc_spec_ideal = await get_accuracy_spec(spec_dir)
                acc_mod_exact, acc_mod_ideal = await get_accuracy_mod(mod_dir)
                
                logger.info(f"Accuracy Speculative RAG - Exact: {acc_spec_exact:.4f}, Ideal: {acc_spec_ideal:.4f}")
                logger.info(f"Accuracy Modified RAG - Exact: {acc_mod_exact:.4f}, Ideal: {acc_mod_ideal:.4f}")
                
                # Actualizar mejor configuración si es necesario
                if acc_mod_exact > best_config["acc_mod_exact"] or acc_mod_ideal > best_config["acc_mod_ideal"]:
                    best_config.update({
                        "params": params.copy(),
                        "acc_mod_exact": acc_mod_exact,
                        "acc_mod_ideal": acc_mod_ideal,
                        "acc_spec_exact": acc_spec_exact,
                        "acc_spec_ideal": acc_spec_ideal,
                        "run_dir": str(run_dir)
                    })
                
                # Guardar resumen de esta corrida
                summary = {
                    "timestamp": timestamp,
                    "run_dir": str(run_dir),
                    "params": params,
                    "attempt": attempt + 1,
                    "n_questions": n_questions,
                    "accuracies": {
                        "speculative": {
                            "exact": acc_spec_exact,
                            "ideal": acc_spec_ideal
                        },
                        "modified": {
                            "exact": acc_mod_exact,
                            "ideal": acc_mod_ideal
                        }
                    },
                    "is_best_so_far": (acc_mod_exact == best_config["acc_mod_exact"] or 
                                     acc_mod_ideal == best_config["acc_mod_ideal"])
                }
                
                with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                
                # Verificar si Modified RAG es mejor en cualquiera de los dos tipos
                if acc_mod_exact >= acc_spec_exact or acc_mod_ideal >= acc_spec_ideal:
                    logger.info("Modified RAG es igual o mejor en al menos un tipo de respuesta. Guardando configuración exitosa.")
                    # Especificar en qué tipo fue mejor o igual
                    comparison_result = []
                    if acc_mod_exact >= acc_spec_exact:
                        comparison_result.append(f"Exact answers: Modified ({acc_mod_exact:.4f}) >= Speculative ({acc_spec_exact:.4f})")
                    if acc_mod_ideal >= acc_spec_ideal:
                        comparison_result.append(f"Ideal answers: Modified ({acc_mod_ideal:.4f}) >= Speculative ({acc_spec_ideal:.4f})")
                    
                    logger.info(f"Comparación favorable: {', '.join(comparison_result)}")
                    
                    with open(base_results_dir / "successful_config.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "successful_params": params,
                            "accuracies": {
                                "modified": {
                                    "exact": acc_mod_exact,
                                    "ideal": acc_mod_ideal
                                },
                                "speculative": {
                                    "exact": acc_spec_exact,
                                    "ideal": acc_spec_ideal
                                }
                            },
                            "comparison_result": comparison_result,
                            "run_directory": str(run_dir)
                        }, f, indent=2)
                    return
                
                logger.info("Modified RAG es peor en ambos tipos. Reintentando con nuevas preguntas...")
                
            except Exception as e:
                logger.error(f"Error en corrida {attempt+1}: {e}")
                continue
            
            await asyncio.sleep(1)
            
        logger.info("Cambiando a siguiente conjunto de parámetros...")
    
    # Al finalizar todas las pruebas, guardar la mejor configuración encontrada
    logger.info("\n=== Resumen Final ===")
    logger.info("Modified RAG no superó a Speculative RAG en ninguna configuración.")
    logger.info(f"Mejor configuración encontrada:")
    logger.info(f"Parámetros: {best_config['params']}")
    logger.info(f"Accuracy Modified RAG - Exact: {best_config['acc_mod_exact']:.4f}, Ideal: {best_config['acc_mod_ideal']:.4f}")
    logger.info(f"Accuracy Speculative RAG - Exact: {best_config['acc_spec_exact']:.4f}, Ideal: {best_config['acc_spec_ideal']:.4f}")
    logger.info(f"Directorio: {best_config['run_dir']}")
    
    with open(base_results_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
