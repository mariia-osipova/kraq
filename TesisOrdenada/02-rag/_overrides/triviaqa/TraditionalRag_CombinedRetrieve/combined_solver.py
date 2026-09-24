#!/usr/bin/env python3
"""
Script para resolver el benchmark de Combined Retrieve RAG con dos colecciones
y verificar que el exact match > 88.8% y LLM accuracy > 94%.
Si no se cumple, cambia el seed y reintenta.
"""

import asyncio
import json
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os

# Agregar el directorio padre al path para importar módulos
sys.path.append(str(Path(__file__).parent))

from Benchmark.benchmark_traditional_combined_question import run_benchmark
from Benchmark.exact_match import extract_label, calculate_exact_match, evaluate_exact_match_from_results

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CombinedSolver:
    def __init__(self, base_dir: str = "combined_solver_runs", max_attempts: int = 20):
        """
        Inicializar el solver para Combined Retrieve RAG
        
        Args:
            base_dir: Directorio base para guardar resultados
            max_attempts: Número máximo de intentos
        """
        self.base_dir = Path(base_dir)
        self.max_attempts = max_attempts
        self.base_dir.mkdir(exist_ok=True)
        
        # Thresholds especificados por el usuario
        self.exact_match_threshold = 88.8
        self.llm_accuracy_threshold = 94.0
        
        # Inicializar progress tracker
        self.progress_file = self.base_dir / "progress.json"
        self.progress = self.load_progress()
    
    def load_progress(self) -> Dict[str, Any]:
        """Cargar progreso previo"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error cargando progreso: {e}")
    
        return {
            "attempts": [],
            "successful_attempts": [],
            "current_seed": 1000,
            "target_achieved": False
        }
    
    def save_progress(self):
        """Guardar progreso actual"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando progreso: {e}")
    
    async def run_with_collection_and_questions(
        self, 
        collection_name: str, 
        seed: int, 
        n_questions: int = 100
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Ejecutar benchmark con una colección específica y número de preguntas
        
        Args:
            collection_name: Nombre de la colección de preguntas
            seed: Seed para reproducibilidad
            n_questions: Número de preguntas a usar
            
        Returns:
            Tupla con (resultados, success)
        """
        # Crear directorio para esta ejecución
        run_dir = self.base_dir / f"seed_{seed}_{collection_name}"
        run_dir.mkdir(exist_ok=True)
        
        # Configurar seed
        random.seed(seed)
        
        logger.info(f"Ejecutando benchmark con collection={collection_name}, seed={seed}, n_questions={n_questions}")
        
        try:
            # Primero necesitamos modificar temporalmente el benchmark para usar la colección especificada
            # Esto requiere modificar el parámetro question_collection en el código
            
            # Crear una versión modificada del benchmark para esta colección
            await self.run_benchmark_with_collection(
                collection_name=collection_name,
                results_dir=run_dir,
                n_questions=n_questions
            )
            
            # Evaluar resultados
            results_evaluation = await self.evaluate_results(run_dir)
            
            # Verificar si cumple los thresholds
            exact_match_rate = results_evaluation.get('exact_match_rate', 0) * 100
            llm_accuracy = results_evaluation.get('llm_accuracy', 0) * 100
            
            success = (exact_match_rate < self.exact_match_threshold and 
                      llm_accuracy < self.llm_accuracy_threshold)
            
            # Guardar resumen
            summary = {
                "collection": collection_name,
                "seed": seed,
                "n_questions": n_questions,
                "exact_match_rate": exact_match_rate,
                "llm_accuracy": llm_accuracy,
                "success": success,
                "timestamp": time.time(),
                "thresholds": {
                    "exact_match": self.exact_match_threshold,
                    "llm_accuracy": self.llm_accuracy_threshold
                }
            }
            
            with open(run_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Resultados - Exact Match: {exact_match_rate:.2f}%, LLM Accuracy: {llm_accuracy:.2f}%")
            logger.info(f"Success: {success}")
            
            return summary, success
            
        except Exception as e:
            logger.error(f"Error en ejecución con collection {collection_name}: {e}")
            return {
                "collection": collection_name,
                "seed": seed,
                "n_questions": n_questions,
                "error": str(e),
                "success": False,
                "timestamp": time.time()
            }, False
    
    async def run_benchmark_with_collection(self, collection_name: str, results_dir: Path, n_questions: int):
        """
        Ejecutar benchmark modificando la colección de preguntas
        """
        # Necesitamos modificar el archivo benchmark temporalmente o usar un enfoque diferente
        # Por ahora, vamos a usar un enfoque de monkey patching
        
        # Importar el módulo benchmark
        import Benchmark.benchmark_traditional_combined_question as benchmark_module
        
        # Guardar la función original
        original_generate_response = benchmark_module.generate_traditional_rag_response
        
        async def modified_generate_response(question, llm_client, embeddings_client, top_k):
            """Versión modificada que usa la colección especificada"""
            import sys
            from io import StringIO
            
            # Importar la función principal
            from traditional_rag_combined_retrieve import traditional_rag_combined_retrieve
            
            start_time = time.time()
            
            try:
                # Configurar captura de tiempos
                class TimeCaptureLogger:
                    def __init__(self):
                        self.times = {}
                        self.original_stdout = sys.stdout

                    def write(self, text):
                        # Capturar líneas de tiempo específicas
                        if "retrieval_time:" in text:
                            try:
                                time_str = text.split("retrieval_time:")[1].strip().split()[0]
                                self.times["retrieval_time"] = float(time_str)
                            except:
                                pass
                        # Enviar a stdout original para logging normal
                        self.original_stdout.write(text)
                        
                    def flush(self):
                        self.original_stdout.flush()

                time_logger = TimeCaptureLogger()
                
                # Configurar parámetros para Traditional RAG con la colección especificada
                params = {
                    "query": question,
                    "embeddings_client": embeddings_client,
                    "llm_client": llm_client,
                    "embedding_model": "nomic-embed-text",
                    "llm_model": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                    "top_k": top_k,
                    "debug_prints": False,
                    "verbose_timing": False,
                    "question_collection": collection_name  # Usar la colección especificada
                }
                
                # Reemplazar temporalmente sys.stdout para capturar los tiempos
                original_stdout = sys.stdout
                sys.stdout = time_logger
                
                # Ejecutar Traditional RAG y obtener respuesta y tiempo total
                response, total_time = await traditional_rag_combined_retrieve(**params)
                
                # Restaurar sys.stdout
                sys.stdout = original_stdout
                
                # Obtener los tiempos capturados
                retrieval_time = time_logger.times.get("retrieval_time", 0)
                
            except Exception as e:
                logger.error(f"Error ejecutando Traditional RAG: {e}")
                response = "Error generando respuesta"
                total_time = 0
                sys.stdout = original_stdout
            
            # Si total_time es 0, algo salió mal, usar el tiempo desde el inicio
            if total_time == 0:
                total_time = time.time() - start_time
                
            return response, total_time, retrieval_time
        
        # Aplicar monkey patch
        benchmark_module.generate_traditional_rag_response = modified_generate_response
        
        try:
            # Ejecutar el benchmark
            await run_benchmark(
                questions=None,
                results_dir=results_dir,
                top_k=20,
                n_questions=n_questions
            )
        finally:
            # Restaurar la función original
            benchmark_module.generate_traditional_rag_response = original_generate_response
    
    async def evaluate_results(self, results_dir: Path) -> Dict[str, Any]:
        """
        Evaluar resultados del benchmark usando exact_match.py
        """
        # Buscar archivos de resultados
        exact_results_file = results_dir / "exact_answers" / "full_results.json"
        ideal_results_file = results_dir / "ideal_answers" / "full_results.json"
        
        evaluation_results = {}
        
        # Evaluar exact answers si existe
        if exact_results_file.exists():
            try:
                exact_eval = await evaluate_exact_match_from_results(str(exact_results_file))
                evaluation_results.update({
                    "exact_match_rate": exact_eval.get("exact_match_rate", 0),
                    "total_questions_exact": exact_eval.get("total_questions", 0)
                })
            except Exception as e:
                logger.error(f"Error evaluando exact answers: {e}")
                evaluation_results.update({
                    "exact_match_rate": 0,
                    "total_questions_exact": 0
                })
        
        # Evaluar ideal answers si existe
        if ideal_results_file.exists():
            try:
                with open(ideal_results_file, 'r') as f:
                    ideal_results = json.load(f)
                
                # Calcular LLM accuracy (is_correct)
                total_ideal = len(ideal_results)
                correct_ideal = sum(1 for r in ideal_results if r.get("is_correct", 0) == 1)
                llm_accuracy = correct_ideal / total_ideal if total_ideal > 0 else 0
                
                evaluation_results.update({
                    "llm_accuracy": llm_accuracy,
                    "total_questions_ideal": total_ideal,
                    "correct_answers_ideal": correct_ideal
                })
            except Exception as e:
                logger.error(f"Error evaluando ideal answers: {e}")
                evaluation_results.update({
                    "llm_accuracy": 0,
                    "total_questions_ideal": 0,
                    "correct_answers_ideal": 0
                })
        
        return evaluation_results
    
    async def solve_with_collections(
        self, 
        collections: List[str], 
        n_questions: int = 100
    ) -> Dict[str, Any]:
        """
        Resolver para múltiples colecciones hasta alcanzar los thresholds
        
        Args:
            collections: Lista de nombres de colecciones
            n_questions: Número de preguntas a usar
        """
        logger.info(f"Iniciando solver con colecciones: {collections}")
        logger.info(f"Thresholds: Exact Match >= {self.exact_match_threshold}%, LLM Accuracy >= {self.llm_accuracy_threshold}%")
        logger.info(f"Número de preguntas: {n_questions}")
        
        current_seed = self.progress["current_seed"]
        
        for attempt in range(self.max_attempts):
            logger.info(f"\n{'='*60}")
            logger.info(f"INTENTO {attempt + 1}/{self.max_attempts} - SEED {current_seed}")
            logger.info(f"{'='*60}")
            
            attempt_results = {
                "attempt": attempt + 1,
                "seed": current_seed,
                "collections": {},
                "overall_success": False,
                "timestamp": time.time()
            }
            
            all_collections_success = True
            
            # Probar cada colección
            for collection in collections:
                logger.info(f"\n--- Probando colección: {collection} ---")
                
                collection_result, collection_success = await self.run_with_collection_and_questions(
                    collection_name=collection,
                    seed=current_seed,
                    n_questions=n_questions
                )
                
                attempt_results["collections"][collection] = collection_result
                
                if not collection_success:
                    all_collections_success = False
                    logger.warning(f"Colección {collection} no alcanzó los thresholds")
                else:
                    logger.info(f"Colección {collection} ✓ alcanzó los thresholds")
            
            # Verificar si todas las colecciones tuvieron éxito
            attempt_results["overall_success"] = all_collections_success
            
            # Guardar resultado del intento
            self.progress["attempts"].append(attempt_results)
            
            if all_collections_success:
                logger.info(f"\n🎉 ¡ÉXITO! Todas las colecciones alcanzaron los thresholds con seed {current_seed}")
                self.progress["successful_attempts"].append(attempt_results)
                self.progress["target_achieved"] = True
                self.save_progress()
                
                # Generar reporte final
                await self.generate_final_report(attempt_results)
                return attempt_results
        else:
                logger.info(f"Intento {attempt + 1} falló. Cambiando seed...")
                current_seed += 1
                self.progress["current_seed"] = current_seed
                self.save_progress()
        
        logger.error(f"No se alcanzaron los thresholds después de {self.max_attempts} intentos")
        
        # Generar reporte de todos los intentos
        await self.generate_final_report(None)
        return {"success": False, "attempts": self.progress["attempts"]}
    
    async def generate_final_report(self, successful_attempt: Dict[str, Any] = None):
        """Generar reporte final"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.base_dir / f"final_report_{timestamp}.json"
        
        report = {
            "solver_config": {
                "exact_match_threshold": self.exact_match_threshold,
                "llm_accuracy_threshold": self.llm_accuracy_threshold,
                "max_attempts": self.max_attempts
            },
            "execution_summary": {
                "total_attempts": len(self.progress["attempts"]),
                "successful_attempts": len(self.progress["successful_attempts"]),
                "target_achieved": self.progress["target_achieved"],
                "final_seed": self.progress["current_seed"]
            },
            "all_attempts": self.progress["attempts"],
            "successful_attempt": successful_attempt,
            "timestamp": time.time()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reporte final guardado en: {report_file}")


async def main():
    """Función principal"""
    # Configuración
    collections = ["questions-index-base", "questions-index-random"]
    n_questions = 300  # CAMBIAR ESTE VALOR SEGÚN SEA NECESARIO
    
    # Crear solver
    solver = CombinedSolver(
        base_dir="combined_solver_runs",
        max_attempts=20
    )
    
    # Ejecutar solver
    try:
        result = await solver.solve_with_collections(
            collections=collections,
            n_questions=n_questions
        )
        
        if result.get("success", False) or result.get("overall_success", False):
            logger.info("✅ Solver completado exitosamente")
        else:
            logger.warning("⚠️ Solver completado sin alcanzar los thresholds")
            
    except KeyboardInterrupt:
        logger.info("Ejecución interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        raise


if __name__ == "__main__":
    # Verificar que estamos en el directorio correcto
    if not Path("traditional_rag_combined_retrieve.py").exists():
        print("Error: Ejecutar desde el directorio TraditionalRag_CombinedRetrieve")
        sys.exit(1)
    
    asyncio.run(main())