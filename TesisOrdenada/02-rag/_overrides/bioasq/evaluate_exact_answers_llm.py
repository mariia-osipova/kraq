#!/usr/bin/env python3
"""
Evaluador LLM para respuestas exactas de BioASQ Solver Results

Este script evalúa las respuestas exactas de los archivos full_results.json
de ambos directorios usando el mismo evaluate_correctness que se usaba antes.
Hace las llamadas en paralelo de a 30 al LLM evaluador.
"""

import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import statistics
from loguru import logger
from openai import AsyncOpenAI
import aiofiles

# Configuración
VLLM_BASE_URL = "http://localhost:8000/v1"
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MAX_CONCURRENT_CALLS = 30

# Rutas de los archivos de resultados
RESULTS_DIRS = {
    "run_1_index-random_20250527_220303": "bioasq_solver_results/run_1_index-random_20250527_220303/full_results.json",
    "run_1_index-base_20250527_190739": "bioasq_solver_results/run_1_index-base_20250527_190739/full_results.json"
}

class ExactAnswerEvaluator:
    def __init__(self):
        """Inicializar el evaluador"""
        self.llm_client = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)
        self.setup_logging()
        
    def setup_logging(self):
        """Configurar logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"exact_answers_evaluation_{timestamp}.log"
        logger.add(log_file, rotation="100 MB", level="INFO")
        logger.info("Evaluador de respuestas exactas inicializado")
        
    async def initialize_client(self):
        """Inicializar cliente LLM"""
        self.llm_client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed"
        )
        logger.info("Cliente LLM inicializado")

    async def evaluate_correctness(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        answer_type: str = "exact_answer"
    ) -> int:
        """
        Evalúa si la respuesta generada es correcta según criterios más estrictos
        (Misma función que se usaba antes)
        """
        async with self.semaphore:
            try:
                # Para respuestas exactas
                if isinstance(reference_answer, list) and len(reference_answer) > 0:
                    # Convertir cada elemento a string y unirlos
                    reference_str = ", ".join([str(ans) for ans in reference_answer])
                    prompt = f"""
You are an expert evaluator of question answering systems with a focus on factual accuracy. Assess if the generated answer correctly addresses the question based on the reference answers.

Question: {question}
Generated Answer: {generated_answer}
Reference Answers: {reference_str}

The reference answers are provided as a list of true answers to the question.

The generated answer should cover most items in the reference answers

Contradicting ANY of the reference answers is a critical error
Including incorrect information invalidates the answer

Is the generated answer correct?
Answer with a single digit:
1 - Yes, the generated answer meets the criteria for correctness
0 - No, the generated answer fails to meet one or more key criteria

Your response (just the digit 1 or 0):
"""
                else:
                    # Una sola respuesta exacta
                    prompt = f"""
You are an expert evaluator of question answering systems with a focus on factual accuracy. Assess if the generated answer correctly addresses the question based on the reference answer.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

The reference answer represents the ground truth to the question.

The generated answer MUST contain the essential information from the reference answer

It must not contain any statements that contradict the reference answer

While the wording may differ, the core facts and meaning must be preserved. Partial answers that miss key points should be considered incorrect.

Is the generated answer correct?
Answer with a single digit:
1 - Yes, the generated answer meets the criteria for correctness
0 - No, the generated answer fails to meet one or more key criteria

Your response (just the digit 1 or 0):
"""

                # Usar el modelo para la evaluación
                response = await self.llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                
                answer_text = response.choices[0].message.content.strip()
                return 1 if answer_text.startswith("1") else 0
                    
            except Exception as e:
                logger.error(f"Error evaluando corrección con modelo local: {e}")
                return 0  # Por defecto, si hay error, consideramos la respuesta incorrecta

    async def load_results_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Cargar archivo de resultados JSON"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                
            # Filtrar solo items que tienen exact_answer
            exact_answers = [item for item in data if 'exact_answer' in item and item['exact_answer']]
            logger.info(f"Cargadas {len(exact_answers)} respuestas exactas de {file_path}")
            return exact_answers
            
        except Exception as e:
            logger.error(f"Error cargando archivo {file_path}: {e}")
            return []

    async def evaluate_single_answer(self, item: Dict[str, Any], run_name: str) -> Dict[str, Any]:
        """Evaluar una sola respuesta"""
        try:
            question = item['question']
            generated_answer = item['generated_answer']
            exact_answer_data = item['exact_answer']
            reference_answer = exact_answer_data['reference_answer']
            question_id = item['question_id']
            
            # Evaluar con LLM
            start_time = time.time()
            llm_correctness = await self.evaluate_correctness(
                question=question,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                answer_type="exact_answer"
            )
            evaluation_time = time.time() - start_time
            
            # Obtener métricas originales del exact_match si existe
            original_exact_match = exact_answer_data.get('exact_match', {})
            original_is_correct = original_exact_match.get('is_match', False)
            
            result = {
                'run_name': run_name,
                'question_id': question_id,
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference_answer,
                'original_is_correct': 1 if original_is_correct else 0,
                'llm_is_correct': llm_correctness,
                'evaluation_time': evaluation_time,
                'original_metrics': {
                    'exact_match_percentage': original_exact_match.get('match_percentage', 0.0),
                    'matched_count': original_exact_match.get('matched_count', 0),
                    'total_reference_answers': original_exact_match.get('total_reference_answers', 0),
                    'cosine_similarity': exact_answer_data.get('cosine_similarity', 0.0),
                    'bert_score_f1': exact_answer_data.get('bert_score_f1', 0.0),
                    'total_time': item.get('total_time', 0.0)
                }
            }
            
            logger.debug(f"Evaluada pregunta {question_id} de {run_name}: LLM={llm_correctness}, Original={original_is_correct}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluando pregunta {item.get('question_id', 'unknown')}: {e}")
            return None

    async def evaluate_run(self, file_path: str, run_name: str) -> List[Dict[str, Any]]:
        """Evaluar todas las respuestas de un run"""
        logger.info(f"Iniciando evaluación de {run_name}")
        
        # Cargar datos
        data = await self.load_results_file(file_path)
        if not data:
            logger.warning(f"No se encontraron datos para {run_name}")
            return []
        
        # Crear tareas para evaluación en paralelo
        tasks = []
        for item in data:
            task = self.evaluate_single_answer(item, run_name)
            tasks.append(task)
        
        # Ejecutar evaluaciones en paralelo
        logger.info(f"Ejecutando {len(tasks)} evaluaciones en paralelo para {run_name}")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados válidos
        valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        total_time = time.time() - start_time
        logger.info(f"Completada evaluación de {run_name}: {len(valid_results)}/{len(tasks)} exitosas en {total_time:.2f}s")
        
        return valid_results

    def calculate_metrics(self, results: List[Dict[str, Any]], run_name: str) -> Dict[str, Any]:
        """Calcular métricas para un run"""
        if not results:
            return {
                'run_name': run_name,
                'total_questions': 0,
                'llm_accuracy': 0.0,
                'original_accuracy': 0.0,
                'agreement_rate': 0.0,
                'avg_evaluation_time': 0.0
            }
        
        total_questions = len(results)
        llm_correct = sum(r['llm_is_correct'] for r in results)
        original_correct = sum(r['original_is_correct'] for r in results)
        
        # Calcular acuerdo entre evaluaciones
        agreements = sum(1 for r in results if r['llm_is_correct'] == r['original_is_correct'])
        
        # Tiempos de evaluación
        evaluation_times = [r['evaluation_time'] for r in results]
        
        # Métricas adicionales
        avg_match_percentage = statistics.mean([r['original_metrics']['exact_match_percentage'] for r in results])
        avg_cosine_similarity = statistics.mean([r['original_metrics']['cosine_similarity'] for r in results])
        avg_bert_f1 = statistics.mean([r['original_metrics']['bert_score_f1'] for r in results])
        
        metrics = {
            'run_name': run_name,
            'total_questions': total_questions,
            'llm_accuracy': llm_correct / total_questions,
            'llm_correct_count': llm_correct,
            'original_accuracy': original_correct / total_questions,
            'original_correct_count': original_correct,
            'agreement_rate': agreements / total_questions,
            'agreement_count': agreements,
            'avg_evaluation_time': statistics.mean(evaluation_times),
            'total_evaluation_time': sum(evaluation_times),
            'min_evaluation_time': min(evaluation_times),
            'max_evaluation_time': max(evaluation_times),
            'avg_exact_match_percentage': avg_match_percentage,
            'avg_cosine_similarity': avg_cosine_similarity,
            'avg_bert_f1': avg_bert_f1
        }
        
        return metrics

    async def save_results(self, all_results: List[Dict[str, Any]], all_metrics: List[Dict[str, Any]]):
        """Guardar resultados en archivos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resultados detallados
        results_file = f"exact_answers_llm_evaluation_detailed_{timestamp}.json"
        async with aiofiles.open(results_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(all_results, indent=2, ensure_ascii=False))
        
        # Guardar métricas
        metrics_file = f"exact_answers_llm_evaluation_metrics_{timestamp}.json"
        async with aiofiles.open(metrics_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(all_metrics, indent=2, ensure_ascii=False))
        
        # Crear reporte de resumen
        report_file = f"exact_answers_llm_evaluation_report_{timestamp}.txt"
        async with aiofiles.open(report_file, 'w', encoding='utf-8') as f:
            await f.write("=== EVALUACIÓN LLM DE RESPUESTAS EXACTAS ===\n\n")
            await f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            await f.write(f"Modelo evaluador: {LLM_MODEL}\n")
            await f.write(f"Concurrencia máxima: {MAX_CONCURRENT_CALLS}\n\n")
            
            for metrics in all_metrics:
                await f.write(f"RUN: {metrics['run_name']}\n")
                await f.write(f"  Total preguntas: {metrics['total_questions']}\n")
                await f.write(f"  Precisión LLM: {metrics['llm_accuracy']:.3f} ({metrics['llm_correct_count']}/{metrics['total_questions']})\n")
                await f.write(f"  Precisión original (exact match): {metrics['original_accuracy']:.3f} ({metrics['original_correct_count']}/{metrics['total_questions']})\n")
                await f.write(f"  Tasa de acuerdo: {metrics['agreement_rate']:.3f} ({metrics['agreement_count']}/{metrics['total_questions']})\n")
                await f.write(f"  Tiempo promedio evaluación: {metrics['avg_evaluation_time']:.3f}s\n")
                await f.write(f"  Tiempo total evaluación: {metrics['total_evaluation_time']:.2f}s\n")
                await f.write(f"  Promedio exact match %: {metrics['avg_exact_match_percentage']:.3f}\n")
                await f.write(f"  Promedio cosine similarity: {metrics['avg_cosine_similarity']:.3f}\n")
                await f.write(f"  Promedio BERT F1: {metrics['avg_bert_f1']:.3f}\n\n")
        
        logger.info(f"Resultados guardados:")
        logger.info(f"  - Detallados: {results_file}")
        logger.info(f"  - Métricas: {metrics_file}")
        logger.info(f"  - Reporte: {report_file}")

    async def run_evaluation(self):
        """Ejecutar evaluación completa"""
        logger.info("=== INICIANDO EVALUACIÓN LLM DE RESPUESTAS EXACTAS ===")
        
        # Inicializar cliente
        await self.initialize_client()
        
        all_results = []
        all_metrics = []
        
        # Evaluar cada run
        for run_name, file_path in RESULTS_DIRS.items():
            if not Path(file_path).exists():
                logger.warning(f"Archivo no encontrado: {file_path}")
                continue
                
            # Evaluar run
            results = await self.evaluate_run(file_path, run_name)
            
            # Calcular métricas
            metrics = self.calculate_metrics(results, run_name)
            
            # Agregar a resultados globales
            all_results.extend(results)
            all_metrics.append(metrics)
            
            # Log métricas
            logger.info(f"Métricas para {run_name}:")
            logger.info(f"  LLM Accuracy: {metrics['llm_accuracy']:.3f}")
            logger.info(f"  Original Accuracy (exact match): {metrics['original_accuracy']:.3f}")
            logger.info(f"  Agreement Rate: {metrics['agreement_rate']:.3f}")
        
        # Guardar resultados
        await self.save_results(all_results, all_metrics)
        
        # Resumen final
        if all_metrics:
            total_questions = sum(m['total_questions'] for m in all_metrics)
            total_llm_correct = sum(m['llm_correct_count'] for m in all_metrics)
            total_original_correct = sum(m['original_correct_count'] for m in all_metrics)
            total_agreements = sum(m['agreement_count'] for m in all_metrics)
            
            logger.info("=== RESUMEN FINAL ===")
            logger.info(f"Total preguntas evaluadas: {total_questions}")
            logger.info(f"Precisión LLM global: {total_llm_correct/total_questions:.3f}")
            logger.info(f"Precisión original (exact match) global: {total_original_correct/total_questions:.3f}")
            logger.info(f"Tasa de acuerdo global: {total_agreements/total_questions:.3f}")

async def main():
    """Función principal"""
    evaluator = ExactAnswerEvaluator()
    await evaluator.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 