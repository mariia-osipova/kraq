#!/usr/bin/env python3
"""
BioASQ Solver for Traditional RAG with Combined Retrieve

This script runs Traditional RAG with Combined Retrieve on 300 BioASQ questions,
testing both index-base and index-random question collections until accuracy < 79%.
It includes exact match calculation as specified.
"""

import json
import sys
import os
import asyncio
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union
import statistics
from datetime import datetime
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from bert_score import score
import torch
from openai import AsyncOpenAI
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the Traditional RAG Combined Retrieve function
from TraditionalRag_CombinedRetrieve.traditional_rag_combined_retrieve import traditional_rag_combined_retrieve

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# Load sentence transformer for similarity calculations
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1')

class BioASQSolver:
    def __init__(self, 
                 bioasq_file: str = "DataBase/qa_benchmark.json",
                 target_accuracy: float = 0.79,
                 num_questions: int = 300,
                 exact_match_threshold: float = 0.5):
        """
        Initialize the BioASQ Solver.
        
        Args:
            bioasq_file: Path to BioASQ questions file
            target_accuracy: Target accuracy threshold (stop when both configs < this)
            num_questions: Number of questions to process
            exact_match_threshold: Threshold for exact match calculation (default: 50%)
        """
        self.bioasq_file = Path(bioasq_file)
        self.target_accuracy = target_accuracy
        self.num_questions = num_questions
        self.exact_match_threshold = exact_match_threshold
        
        # Question collections to test
        self.question_collections = {
            "index-base": "questions-index-base",
            "index-random": "questions-index-random"
        }
        
        # Initialize clients
        self.embeddings_client = None
        self.llm_client = None
        
        # Results storage
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"bioasq_solver_{timestamp}.log"
        
        logger.add(log_file, rotation="100 MB", level="INFO")
        logger.info("BioASQ Solver initialized")
        
    async def initialize_clients(self):
        """Initialize OpenAI clients for embeddings and LLM."""
        self.embeddings_client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"
        )
        
        self.llm_client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed"
        )
        
        logger.info("Clients initialized successfully")
        
    def load_bioasq_questions(self) -> List[Dict[str, Any]]:
        """
        Load BioASQ questions from the JSON file.
        
        Returns:
            List of question dictionaries
        """
        if not self.bioasq_file.exists():
            raise FileNotFoundError(f"BioASQ file not found: {self.bioasq_file}")
            
        with open(self.bioasq_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        if len(questions) < self.num_questions:
            logger.warning(f"Only {len(questions)} questions available, requested {self.num_questions}")
            return questions
            
        # Randomly sample the requested number of questions
        selected_questions = random.sample(questions, self.num_questions)
        
        logger.info(f"Loaded {len(selected_questions)} questions from BioASQ")
        return selected_questions
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not isinstance(text, str):
            return ""
        
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common punctuation
        text = re.sub(r'[.,;:!?()"\'-]', '', text)
        
        return text
        
    def flatten_reference_answers(self, reference_answer: Union[str, List]) -> List[str]:
        """Flatten reference answers into a list of strings."""
        if isinstance(reference_answer, str):
            return [reference_answer]
        
        if not isinstance(reference_answer, list):
            return []
        
        flattened = []
        for item in reference_answer:
            if isinstance(item, str):
                flattened.append(item)
            elif isinstance(item, list):
                flattened.extend(self.flatten_reference_answers(item))
        
        return flattened
        
    def check_single_answer_match(self, reference_text: str, generated_text: str) -> bool:
        """Check if a single reference answer is contained in the generated text."""
        if not reference_text or not generated_text:
            return False
        
        ref_normalized = self.normalize_text(reference_text)
        gen_normalized = self.normalize_text(generated_text)
        
        if not ref_normalized:
            return False
        
        # Check for exact substring match
        if ref_normalized in gen_normalized:
            return True
        
        # For yes/no questions, check for semantic equivalence
        if ref_normalized in ['yes', 'no']:
            if gen_normalized.startswith(ref_normalized):
                return True
            if ref_normalized == 'yes' and any(word in gen_normalized for word in ['yes', 'true', 'correct', 'indeed', 'affirmative']):
                return True
            if ref_normalized == 'no' and any(word in gen_normalized for word in ['no', 'false', 'incorrect', 'negative', 'not']):
                return True
        
        # For very short reference answers, be more flexible
        if len(ref_normalized) <= 3 and ref_normalized.isalpha():
            pattern = r'\b' + re.escape(ref_normalized) + r'\b'
            if re.search(pattern, gen_normalized):
                return True
        
        return False
        
    def calculate_exact_match(self, reference_answer: Union[str, List], generated_answer: str) -> Dict[str, Any]:
        """
        Calculate exact match based on the threshold.
        
        Args:
            reference_answer: The reference answer (can be string or list)
            generated_answer: The generated answer
            
        Returns:
            Dictionary with match information
        """
        reference_answers = self.flatten_reference_answers(reference_answer)
        
        if not generated_answer or not reference_answers:
            return {
                'is_match': False,
                'matched_answers': [],
                'total_reference_answers': len(reference_answers),
                'matched_count': 0,
                'match_percentage': 0.0
            }
        
        matched_answers = []
        
        for ref_answer in reference_answers:
            if not ref_answer:
                continue
                
            if self.check_single_answer_match(ref_answer, generated_answer):
                matched_answers.append(ref_answer)
        
        match_percentage = len(matched_answers) / len(reference_answers) if reference_answers else 0.0
        is_match = match_percentage >= self.exact_match_threshold
        
        return {
            'is_match': is_match,
            'matched_answers': matched_answers,
            'total_reference_answers': len(reference_answers),
            'matched_count': len(matched_answers),
            'match_percentage': match_percentage
        }
        
    def calculate_cosine_similarity(self, text1: str, text2: Any) -> float:
        """Calculate cosine similarity between two texts."""
        if isinstance(text1, list):
            text1 = str(text1)
        if isinstance(text2, list):
            text2 = str(text2)
        
        if not text1 or not text2:
            return 0.0
        
        try:
            embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
            embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
            
            if len(embedding1.shape) > 1 and embedding1.shape[0] > 1:
                embedding1 = torch.mean(embedding1, dim=0)
            if len(embedding2.shape) > 1 and embedding2.shape[0] > 1:
                embedding2 = torch.mean(embedding2, dim=0)
                
            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            
            if isinstance(similarity, torch.Tensor) and similarity.numel() > 1:
                similarity = similarity.mean()
                
            return similarity.item()
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
            
    def calculate_bert_score(self, candidate: str, reference: Any) -> Tuple[float, float, float]:
        """Calculate BERTScore between candidate and reference."""
        if isinstance(candidate, list):
            candidate = str(candidate)
        if isinstance(reference, list):
            try:
                reference = str(max(reference, key=lambda x: len(str(x))))
            except:
                reference = str(reference)
        
        try:
            P, R, F1 = score([candidate], [reference], lang="en", verbose=False, device='cuda:1')
            return P.item(), R.item(), F1.item()
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return 0.0, 0.0, 0.0
            
    async def evaluate_correctness_llm(self, question: str, generated_answer: str, reference_answer: str) -> int:
        """Evaluate if the generated answer is correct using LLM with ideal answer."""
        try:
            prompt = f"""
You are an expert evaluator of question answering systems with a focus on completeness and accuracy.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {reference_answer}

The generated answer MUST contain the essential information from the reference answer.
It must not contain any statements that contradict the reference answer.
While the wording may differ, the core facts and meaning must be preserved.

Is the generated answer correct?
Answer with a single digit:
1 - Yes, the generated answer meets the criteria for correctness
0 - No, the generated answer fails to meet one or more key criteria

Your response (just the digit 1 or 0):
"""

            response = await self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract just the digit
            match = re.search(r'[01]', answer)
            if match:
                return int(match.group())
            else:
                logger.warning(f"Could not parse LLM evaluation response: {answer}")
                return 0
                
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0
            
    async def process_question(self, question_data: Dict[str, Any], question_collection: str) -> Dict[str, Any]:
        """
        Process a single question with the specified question collection.
        
        Args:
            question_data: Question data dictionary
            question_collection: Name of the question collection to use
            
        Returns:
            Dictionary with results
        """
        question_id = question_data.get('id', 'unknown')
        question = question_data.get('question', '')
        ideal_answer = question_data.get('ideal_answer', '')
        exact_answer = question_data.get('exact_answer', '')
        
        logger.info(f"Processing question {question_id} with collection {question_collection}")
        
        try:
            # Generate response using Traditional RAG Combined Retrieve
            generated_answer, total_time = await traditional_rag_combined_retrieve(
                query=question,
                embeddings_client=self.embeddings_client,
                llm_client=self.llm_client,
                embedding_model=EMBEDDING_MODEL,
                llm_model=LLM_MODEL,
                top_k=5,
                debug_prints=False,
                verbose_timing=False,
                collection_name="chunks",
                question_collection=question_collection
            )
            
            # Initialize results
            results = {
                'question_id': question_id,
                'question': question,
                'generated_answer': generated_answer,
                'total_time': total_time,
                'question_collection': question_collection
            }
            
            # Calculate exact match ONLY with exact answers
            exact_match_result = None
            if exact_answer:
                exact_match_result = self.calculate_exact_match(exact_answer, generated_answer)
                
                # Calculate additional metrics for exact answers
                exact_cosine_similarity = self.calculate_cosine_similarity(generated_answer, exact_answer)
                exact_bert_precision, exact_bert_recall, exact_bert_f1 = self.calculate_bert_score(generated_answer, exact_answer)
                
                results['exact_answer'] = {
                    'reference_answer': exact_answer,
                    'exact_match': exact_match_result,
                    'cosine_similarity': exact_cosine_similarity,
                    'bert_score_precision': exact_bert_precision,
                    'bert_score_recall': exact_bert_recall,
                    'bert_score_f1': exact_bert_f1
                }
            
            # Calculate LLM accuracy ONLY with ideal answers
            llm_correct = None
            if ideal_answer:
                llm_correct = await self.evaluate_correctness_llm(question, generated_answer, ideal_answer)
                
                # Calculate additional metrics for ideal answers
                ideal_cosine_similarity = self.calculate_cosine_similarity(generated_answer, ideal_answer)
                ideal_bert_precision, ideal_bert_recall, ideal_bert_f1 = self.calculate_bert_score(generated_answer, ideal_answer)
                
                results['ideal_answer'] = {
                    'reference_answer': ideal_answer,
                    'llm_correct': llm_correct,
                    'cosine_similarity': ideal_cosine_similarity,
                    'bert_score_precision': ideal_bert_precision,
                    'bert_score_recall': ideal_bert_recall,
                    'bert_score_f1': ideal_bert_f1
                }
            
            # Store the main metrics for easy access
            results['main_metrics'] = {
                'exact_match_available': exact_match_result is not None,
                'exact_match_correct': exact_match_result['is_match'] if exact_match_result else None,
                'llm_accuracy_available': llm_correct is not None,
                'llm_correct': llm_correct
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            return {
                'question_id': question_id,
                'question': question,
                'error': str(e),
                'question_collection': question_collection
            }
            
    def calculate_run_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics for a run."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        metrics = {
            'total_questions': len(results),
            'valid_questions': len(valid_results),
            'avg_total_time': statistics.mean([r['total_time'] for r in valid_results])
        }
        
        # Calculate exact match metrics (only from exact answers)
        exact_match_results = [r for r in valid_results if r.get('main_metrics', {}).get('exact_match_available', False)]
        if exact_match_results:
            exact_match_scores = [r['main_metrics']['exact_match_correct'] for r in exact_match_results]
            exact_match_count = sum(exact_match_scores)
            exact_match_accuracy = statistics.mean(exact_match_scores)
            
            # Additional metrics for exact answers
            exact_cosine_similarities = [r['exact_answer']['cosine_similarity'] for r in exact_match_results]
            exact_bert_f1_scores = [r['exact_answer']['bert_score_f1'] for r in exact_match_results]
            
            metrics['exact_match_metrics'] = {
                'total_questions': len(exact_match_results),
                'exact_match_accuracy': exact_match_accuracy,
                'exact_match_count': exact_match_count,
                'avg_cosine_similarity': statistics.mean(exact_cosine_similarities),
                'avg_bert_f1': statistics.mean(exact_bert_f1_scores)
            }
        else:
            metrics['exact_match_metrics'] = {
                'total_questions': 0,
                'exact_match_accuracy': 0.0,
                'exact_match_count': 0,
                'avg_cosine_similarity': 0.0,
                'avg_bert_f1': 0.0
            }
        
        # Calculate LLM accuracy metrics (only from ideal answers)
        llm_results = [r for r in valid_results if r.get('main_metrics', {}).get('llm_accuracy_available', False)]
        if llm_results:
            llm_correct_scores = [r['main_metrics']['llm_correct'] for r in llm_results]
            llm_correct_count = sum(llm_correct_scores)
            llm_accuracy = statistics.mean(llm_correct_scores)
            
            # Additional metrics for ideal answers
            ideal_cosine_similarities = [r['ideal_answer']['cosine_similarity'] for r in llm_results]
            ideal_bert_f1_scores = [r['ideal_answer']['bert_score_f1'] for r in llm_results]
            
            metrics['llm_accuracy_metrics'] = {
                'total_questions': len(llm_results),
                'llm_accuracy': llm_accuracy,
                'llm_correct_count': llm_correct_count,
                'avg_cosine_similarity': statistics.mean(ideal_cosine_similarities),
                'avg_bert_f1': statistics.mean(ideal_bert_f1_scores)
            }
        else:
            metrics['llm_accuracy_metrics'] = {
                'total_questions': 0,
                'llm_accuracy': 0.0,
                'llm_correct_count': 0,
                'avg_cosine_similarity': 0.0,
                'avg_bert_f1': 0.0
            }
        
        return metrics
        
    async def run_single_configuration(self, questions: List[Dict[str, Any]], collection_name: str, collection_key: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run a single configuration (collection) on all questions.
        
        Args:
            questions: List of questions to process
            collection_name: Name of the question collection
            collection_key: Key for the collection (index-base or index-random)
            
        Returns:
            Tuple of (results_list, metrics_dict)
        """
        logger.info(f"Starting run with collection: {collection_name} ({collection_key})")
        
        results = []
        
        for i, question_data in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)} with {collection_key}")
            
            result = await self.process_question(question_data, collection_name)
            results.append(result)
            
            # Small delay to avoid overwhelming the system
            if i % 10 == 0:
                await asyncio.sleep(1)
                logger.info(f"Completed {i}/{len(questions)} questions with {collection_key}")
        
        # Calculate metrics
        metrics = self.calculate_run_metrics(results)
        
        logger.info(f"Completed run with {collection_key}")
        logger.info(f"Exact Match Accuracy: {metrics.get('exact_match_metrics', {}).get('exact_match_accuracy', 0)*100:.2f}% ({metrics.get('exact_match_metrics', {}).get('total_questions', 0)} questions)")
        logger.info(f"LLM Accuracy: {metrics.get('llm_accuracy_metrics', {}).get('llm_accuracy', 0)*100:.2f}% ({metrics.get('llm_accuracy_metrics', {}).get('total_questions', 0)} questions)")
        
        return results, metrics
        
    def save_run_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any], collection_key: str, run_number: int):
        """Save results for a single run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = Path(f"bioasq_solver_results/run_{run_number}_{collection_key}_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "full_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save metrics
        with open(output_dir / "metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        self.generate_summary_report(metrics, output_dir / "summary_report.txt")
        
        logger.info(f"Results saved to {output_dir}")
        
    def generate_summary_report(self, metrics: Dict[str, Any], output_file: Path):
        """Generate a human-readable summary report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BIOASQ SOLVER SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Questions: {metrics.get('total_questions', 0)}\n")
            f.write(f"Valid Questions: {metrics.get('valid_questions', 0)}\n")
            f.write(f"Average Total Time: {metrics.get('avg_total_time', 0):.2f}s\n\n")
            
            # Exact Match Metrics (from exact answers)
            exact_metrics = metrics.get('exact_match_metrics', {})
            f.write("EXACT MATCH METRICS (from exact answers):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Questions with exact answers: {exact_metrics.get('total_questions', 0)}\n")
            f.write(f"Exact Match Accuracy: {exact_metrics.get('exact_match_accuracy', 0)*100:.2f}%\n")
            f.write(f"Exact Match Count: {exact_metrics.get('exact_match_count', 0)}\n")
            f.write(f"Avg Cosine Similarity: {exact_metrics.get('avg_cosine_similarity', 0):.4f}\n")
            f.write(f"Avg BERT F1: {exact_metrics.get('avg_bert_f1', 0):.4f}\n\n")
            
            # LLM Accuracy Metrics (from ideal answers)
            llm_metrics = metrics.get('llm_accuracy_metrics', {})
            f.write("LLM ACCURACY METRICS (from ideal answers):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Questions with ideal answers: {llm_metrics.get('total_questions', 0)}\n")
            f.write(f"LLM Accuracy: {llm_metrics.get('llm_accuracy', 0)*100:.2f}%\n")
            f.write(f"LLM Correct Count: {llm_metrics.get('llm_correct_count', 0)}\n")
            f.write(f"Avg Cosine Similarity: {llm_metrics.get('avg_cosine_similarity', 0):.4f}\n")
            f.write(f"Avg BERT F1: {llm_metrics.get('avg_bert_f1', 0):.4f}\n\n")
                
    async def run_solver(self):
        """
        Main solver loop. Runs until both configurations have accuracy < target_accuracy.
        """
        logger.info("Starting BioASQ Solver")
        logger.info(f"Target accuracy: {self.target_accuracy*100:.1f}%")
        logger.info(f"Number of questions: {self.num_questions}")
        logger.info(f"Exact match threshold: {self.exact_match_threshold*100:.1f}%")
        logger.info("Exact match calculated with exact answers, LLM accuracy with ideal answers")
        logger.info("Each run uses new random questions, but same questions within each run for both configurations")
        
        # Initialize clients
        await self.initialize_clients()
        
        run_number = 1
        
        while True:
            logger.info(f"\n{'='*80}")
            logger.info(f"STARTING RUN {run_number}")
            logger.info(f"{'='*80}")
            
            # Load NEW questions for this run
            questions = self.load_bioasq_questions()
            logger.info(f"Loaded {len(questions)} new random questions for run {run_number}")
            
            # Track if both configurations are below target
            both_below_target = True
            
            # Test both question collections with THE SAME questions within this run
            for collection_key, collection_name in self.question_collections.items():
                logger.info(f"\nTesting {collection_key} ({collection_name}) with the same {len(questions)} questions")
                
                # Run the configuration
                results, metrics = await self.run_single_configuration(questions, collection_name, collection_key)
                
                # Save results
                self.save_run_results(results, metrics, collection_key, run_number)
                
                # Store results for this run
                if run_number not in self.results:
                    self.results[run_number] = {}
                self.results[run_number][collection_key] = {
                    'results': results,
                    'metrics': metrics
                }
                
                # Check if accuracy is above target (use both exact match and LLM accuracy)
                exact_match_accuracy = metrics.get('exact_match_metrics', {}).get('exact_match_accuracy', 0)
                llm_accuracy = metrics.get('llm_accuracy_metrics', {}).get('llm_accuracy', 0)
                
                logger.info(f"{collection_key} - Exact Match Accuracy: {exact_match_accuracy*100:.2f}%")
                logger.info(f"{collection_key} - LLM Accuracy: {llm_accuracy*100:.2f}%")
                
                # If either accuracy is >= target, we continue
                if exact_match_accuracy >= self.target_accuracy or llm_accuracy >= self.target_accuracy:
                    both_below_target = False
                    logger.info(f"{collection_key} still above target accuracy")
                else:
                    logger.info(f"{collection_key} below target accuracy")
            
            # Check stopping condition
            if both_below_target:
                logger.info(f"\nSTOPPING: Both configurations have accuracy < {self.target_accuracy*100:.1f}%")
                break
            else:
                logger.info(f"\nCONTINUING: At least one configuration still has accuracy >= {self.target_accuracy*100:.1f}%")
                logger.info("Next run will use NEW random questions")
            
            run_number += 1
            
            # Safety check to prevent infinite loops
            if run_number > 20:  # Maximum 20 runs
                logger.warning("Maximum number of runs reached. Stopping.")
                break
        
        # Generate final summary
        self.generate_final_summary()
        
        logger.info("BioASQ Solver completed")
        
    def generate_final_summary(self):
        """Generate a final summary of all runs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path(f"bioasq_solver_final_summary_{timestamp}.json")
        
        final_summary = {
            'metadata': {
                'target_accuracy': self.target_accuracy,
                'num_questions': self.num_questions,
                'exact_match_threshold': self.exact_match_threshold,
                'total_runs': len(self.results),
                'question_collections': self.question_collections,
                'evaluation_method': 'exact_match_from_exact_answers_llm_accuracy_from_ideal_answers'
            },
            'runs': self.results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final summary saved to {summary_file}")
        
        # Also create a human-readable summary
        readable_summary_file = Path(f"bioasq_solver_final_summary_{timestamp}.txt")
        with open(readable_summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BIOASQ SOLVER FINAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Target Accuracy: {self.target_accuracy*100:.1f}%\n")
            f.write(f"Number of Questions: {self.num_questions}\n")
            f.write(f"Exact Match Threshold: {self.exact_match_threshold*100:.1f}%\n")
            f.write(f"Total Runs: {len(self.results)}\n")
            f.write("Evaluation Method: Exact match from exact answers, LLM accuracy from ideal answers\n\n")
            
            for run_num, run_data in self.results.items():
                f.write(f"RUN {run_num}:\n")
                f.write("-" * 40 + "\n")
                
                for collection_key, collection_data in run_data.items():
                    metrics = collection_data['metrics']
                    
                    exact_match_accuracy = metrics.get('exact_match_metrics', {}).get('exact_match_accuracy', 0)
                    llm_accuracy = metrics.get('llm_accuracy_metrics', {}).get('llm_accuracy', 0)
                    exact_questions = metrics.get('exact_match_metrics', {}).get('total_questions', 0)
                    ideal_questions = metrics.get('llm_accuracy_metrics', {}).get('total_questions', 0)
                    
                    f.write(f"  {collection_key}:\n")
                    f.write(f"    Exact Match Accuracy: {exact_match_accuracy*100:.2f}% ({exact_questions} questions)\n")
                    f.write(f"    LLM Accuracy: {llm_accuracy*100:.2f}% ({ideal_questions} questions)\n")
                
                f.write("\n")
        
        logger.info(f"Human-readable summary saved to {readable_summary_file}")


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BioASQ Solver with Traditional RAG Combined Retrieve')
    parser.add_argument('--bioasq-file', default='DataBase/qa_benchmark.json', help='Path to BioASQ questions file')
    parser.add_argument('--target-accuracy', type=float, default=0.79, help='Target accuracy threshold (default: 0.79)')
    parser.add_argument('--num-questions', type=int, default=1000, help='Number of questions to process (default: 300)')
    parser.add_argument('--exact-match-threshold', type=float, default=0.5, help='Exact match threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create solver
    solver = BioASQSolver(
        bioasq_file=args.bioasq_file,
        target_accuracy=args.target_accuracy,
        num_questions=args.num_questions,
        exact_match_threshold=args.exact_match_threshold
    )
    
    # Run solver
    await solver.run_solver()


if __name__ == "__main__":
    asyncio.run(main()) 