#!/usr/bin/env python3
"""
Exact Match Metric Calculator for Traditional RAG Results

This script calculates exact match metrics comparing the performance between
'TraditionalRag' and 'TraditionalRag_CombinedRetrieve' approaches for both 'exact_answers' and 'ideal_answers'.
The exact match metric checks if at least 50% of reference answers are contained in the generated answer.
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple, Union
from pathlib import Path
import argparse

class ExactMatchCalculator:
    def __init__(self, base_dir: str, match_threshold: float = 0.5):
        """
        Initialize the calculator with the base directory containing the results.
        
        Args:
            base_dir: Path to the base directory containing TraditionalRag folders
            match_threshold: Minimum percentage of reference answers that must match (default: 0.5 = 50%)
        """
        self.base_dir = Path(base_dir)
        self.match_threshold = match_threshold
        self.results = {}
        
    def load_full_results(self, approach: str, answer_type: str) -> List[Dict[str, Any]]:
        """
        Load full_results.json from the specified approach and answer type.
        
        Args:
            approach: Either 'TraditionalRag' or 'TraditionalRag_CombinedRetrieve'
            answer_type: Either 'exact_answers' or 'ideal_answers'
            
        Returns:
            List of result dictionaries
        """
        if approach == "TraditionalRag":
            file_path = self.base_dir / approach / "Benchmark" / "benchmark_results_traditional_qdrant_600q" / answer_type / "full_results.json"
        else:
            file_path = self.base_dir / approach / "Benchmark" / "benchmark_results_traditional_qdrant_500q_noprompt" / answer_type / "full_results.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing extra whitespace, 
        converting to lowercase, and removing punctuation.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common punctuation that might interfere with matching
        text = re.sub(r'[.,;:!?()"\'-]', '', text)
        
        return text
    
    def flatten_reference_answers(self, reference_answer: Union[str, List]) -> List[str]:
        """
        Flatten reference answers into a list of strings, handling nested structures.
        
        Args:
            reference_answer: The reference answer (can be string, list, or nested list)
            
        Returns:
            List of string answers
        """
        if isinstance(reference_answer, str):
            return [reference_answer]
        
        if not isinstance(reference_answer, list):
            return []
        
        flattened = []
        for item in reference_answer:
            if isinstance(item, str):
                flattened.append(item)
            elif isinstance(item, list):
                # Recursively flatten nested lists
                flattened.extend(self.flatten_reference_answers(item))
        
        return flattened
    
    def flatten_generated_answer(self, generated_answer: Union[str, List]) -> str:
        """
        Convert generated answer to a single string for comparison.
        
        Args:
            generated_answer: The generated answer (can be string or list)
            
        Returns:
            String representation of the answer
        """
        if isinstance(generated_answer, str):
            return generated_answer
        
        if isinstance(generated_answer, list):
            # Join list elements with spaces
            return " ".join(str(item) for item in generated_answer)
        
        return str(generated_answer)
    
    def check_single_answer_match(self, reference_text: str, generated_text: str) -> bool:
        """
        Check if a single reference answer is contained in the generated text.
        
        Args:
            reference_text: Single reference answer
            generated_text: Generated answer text
            
        Returns:
            True if the reference is found in the generated text
        """
        if not reference_text or not generated_text:
            return False
        
        ref_normalized = self.normalize_text(reference_text)
        gen_normalized = self.normalize_text(generated_text)
        
        if not ref_normalized:
            return False
        
        # Check for exact substring match
        if ref_normalized in gen_normalized:
            return True
        
        # For yes/no questions, also check for semantic equivalence
        if ref_normalized in ['yes', 'no']:
            # Check if the generated answer starts with yes/no
            if gen_normalized.startswith(ref_normalized):
                return True
            # Check for common affirmative/negative patterns
            if ref_normalized == 'yes' and any(word in gen_normalized for word in ['yes', 'true', 'correct', 'indeed', 'affirmative']):
                return True
            if ref_normalized == 'no' and any(word in gen_normalized for word in ['no', 'false', 'incorrect', 'negative', 'not']):
                return True
        
        # For very short reference answers (like abbreviations), be more flexible
        if len(ref_normalized) <= 3 and ref_normalized.isalpha():
            # Check if it appears as a whole word
            pattern = r'\b' + re.escape(ref_normalized) + r'\b'
            if re.search(pattern, gen_normalized):
                return True
        
        return False
    
    def check_exact_match(self, reference_answer: Union[str, List], generated_answer: Union[str, List]) -> Dict[str, Any]:
        """
        Check if enough reference answers are contained in the generated answer.
        Uses the match_threshold to determine if it's considered an exact match.
        
        Args:
            reference_answer: The reference answer (can be string or list of strings/lists)
            generated_answer: The generated answer to check (can be string or list)
            
        Returns:
            Dictionary with match information
        """
        # Flatten and normalize inputs
        reference_answers = self.flatten_reference_answers(reference_answer)
        generated_text = self.flatten_generated_answer(generated_answer)
        
        if not generated_text or not reference_answers:
            return {
                'is_match': False,
                'matched_answers': [],
                'reference_answers': reference_answers,
                'generated_text': generated_text,
                'total_reference_answers': len(reference_answers),
                'matched_count': 0,
                'match_percentage': 0.0
            }
        
        matched_answers = []
        
        # Check each reference answer individually
        for ref_answer in reference_answers:
            if not ref_answer:
                continue
            
            if self.check_single_answer_match(ref_answer, generated_text):
                matched_answers.append(ref_answer)
        
        # Calculate match percentage
        match_percentage = len(matched_answers) / len(reference_answers) if reference_answers else 0.0
        
        # Determine if it's considered an exact match based on threshold
        is_match = match_percentage >= self.match_threshold
        
        return {
            'is_match': is_match,
            'matched_answers': matched_answers,
            'reference_answers': reference_answers,
            'generated_text': generated_text,
            'total_reference_answers': len(reference_answers),
            'matched_count': len(matched_answers),
            'match_percentage': match_percentage
        }
    
    def calculate_exact_match(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate exact match metrics from results using actual text comparison.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary containing exact match metrics
        """
        total_questions = len(results)
        exact_matches = 0
        partial_matches = 0  # Questions where some but not enough reference answers matched
        
        match_details = []
        total_match_percentage = 0.0
        
        for result in results:
            reference_answer = result.get('reference_answer')
            generated_answer = result.get('generated_answer', '')
            
            match_info = self.check_exact_match(reference_answer, generated_answer)
            
            if match_info['is_match']:
                exact_matches += 1
            elif match_info['matched_count'] > 0:
                partial_matches += 1
            
            total_match_percentage += match_info['match_percentage']
            
            match_details.append({
                'question_id': result.get('question_id'),
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'is_exact_match': match_info['is_match'],
                'matched_answers': match_info['matched_answers'],
                'total_reference_answers': match_info['total_reference_answers'],
                'matched_count': match_info['matched_count'],
                'match_percentage': match_info['match_percentage'],
                'llm_is_correct': result.get('is_correct', 0) == 1  # Keep LLM evaluation for comparison
            })
        
        exact_match_score = exact_matches / total_questions if total_questions > 0 else 0
        avg_match_percentage = total_match_percentage / total_questions if total_questions > 0 else 0
        
        return {
            'total_questions': total_questions,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'exact_match_score': exact_match_score,
            'accuracy_percentage': exact_match_score * 100,
            'avg_match_percentage': avg_match_percentage,
            'match_threshold': self.match_threshold,
            'match_details': match_details
        }
    
    def calculate_detailed_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate detailed metrics including timing and scoring information.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary containing detailed metrics
        """
        if not results:
            return {}
            
        # Basic exact match metrics
        basic_metrics = self.calculate_exact_match(results)
        
        # Timing metrics (Traditional RAG has different timing structure)
        total_times = [r.get('total_time', 0) for r in results]
        retrieval_times = [r.get('retrieval_time', 0) for r in results]
        
        # Scoring metrics
        bert_f1_scores = [r.get('bert_score_f1', 0) for r in results]
        cosine_similarities = [r.get('cosine_similarity', 0) for r in results]
        
        # Compare exact match vs LLM evaluation
        llm_correct = sum(1 for r in results if r.get('is_correct', 0) == 1)
        llm_accuracy = llm_correct / len(results) if results else 0
        
        detailed_metrics = {
            **basic_metrics,
            'timing': {
                'avg_total_time': sum(total_times) / len(total_times) if total_times else 0,
                'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
                'total_processing_time': sum(total_times)
            },
            'scoring': {
                'avg_bert_f1': sum(bert_f1_scores) / len(bert_f1_scores) if bert_f1_scores else 0,
                'avg_cosine_similarity': sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0
            },
            'comparison_with_llm': {
                'llm_accuracy': llm_accuracy,
                'llm_correct_count': llm_correct,
                'exact_match_vs_llm_difference': basic_metrics['exact_match_score'] - llm_accuracy,
                'agreement_rate': self.calculate_agreement_rate(basic_metrics['match_details'])
            }
        }
        
        return detailed_metrics
    
    def calculate_agreement_rate(self, match_details: List[Dict[str, Any]]) -> float:
        """
        Calculate the agreement rate between exact match and LLM evaluation.
        
        Args:
            match_details: List of match detail dictionaries
            
        Returns:
            Agreement rate as a float
        """
        if not match_details:
            return 0.0
            
        agreements = sum(1 for detail in match_details 
                        if detail['is_exact_match'] == detail['llm_is_correct'])
        
        return agreements / len(match_details)
    
    def compare_approaches(self, answer_type: str) -> Dict[str, Any]:
        """
        Compare TraditionalRag vs TraditionalRag_CombinedRetrieve approaches for a given answer type.
        
        Args:
            answer_type: Either 'exact_answers' or 'ideal_answers'
            
        Returns:
            Dictionary containing comparison metrics
        """
        # Load results for both approaches
        traditional_results = self.load_full_results('TraditionalRag', answer_type)
        combined_results = self.load_full_results('TraditionalRag_CombinedRetrieve', answer_type)
        
        # Calculate metrics for each approach
        traditional_metrics = self.calculate_detailed_metrics(traditional_results)
        combined_metrics = self.calculate_detailed_metrics(combined_results)
        
        # Calculate differences
        accuracy_diff = combined_metrics['exact_match_score'] - traditional_metrics['exact_match_score']
        time_diff = combined_metrics['timing']['avg_total_time'] - traditional_metrics['timing']['avg_total_time']
        
        comparison = {
            'answer_type': answer_type,
            'traditional': traditional_metrics,
            'combined_retrieve': combined_metrics,
            'comparison': {
                'exact_match_difference': accuracy_diff,
                'exact_match_improvement_percentage': (accuracy_diff / traditional_metrics['exact_match_score'] * 100) if traditional_metrics['exact_match_score'] > 0 else 0,
                'time_difference': time_diff,
                'time_improvement_percentage': (time_diff / traditional_metrics['timing']['avg_total_time'] * 100) if traditional_metrics['timing']['avg_total_time'] > 0 else 0,
                'combined_better_exact_match': accuracy_diff > 0,
                'combined_is_faster': time_diff < 0
            }
        }
        
        return comparison
    
    def analyze_question_level_differences(self, answer_type: str) -> List[Dict[str, Any]]:
        """
        Analyze differences at the question level between approaches.
        
        Args:
            answer_type: Either 'exact_answers' or 'ideal_answers'
            
        Returns:
            List of question-level comparisons
        """
        traditional_results = self.load_full_results('TraditionalRag', answer_type)
        combined_results = self.load_full_results('TraditionalRag_CombinedRetrieve', answer_type)
        
        # Calculate exact match for both approaches
        traditional_metrics = self.calculate_exact_match(traditional_results)
        combined_metrics = self.calculate_exact_match(combined_results)
        
        # Create dictionaries for quick lookup
        traditional_dict = {detail['question_id']: detail for detail in traditional_metrics['match_details']}
        combined_dict = {detail['question_id']: detail for detail in combined_metrics['match_details']}
        
        question_comparisons = []
        
        # Find common questions
        common_questions = set(traditional_dict.keys()) & set(combined_dict.keys())
        
        for question_id in common_questions:
            traditional_detail = traditional_dict[question_id]
            combined_detail = combined_dict[question_id]
            
            # Get timing info from original results
            traditional_result = next((r for r in traditional_results if r['question_id'] == question_id), {})
            combined_result = next((r for r in combined_results if r['question_id'] == question_id), {})
            
            comparison = {
                'question_id': question_id,
                'question': traditional_result.get('question', ''),
                'reference_answer': traditional_detail['reference_answer'],
                'traditional_generated_answer': traditional_detail['generated_answer'],
                'combined_generated_answer': combined_detail['generated_answer'],
                'traditional_exact_match': traditional_detail['is_exact_match'],
                'combined_exact_match': combined_detail['is_exact_match'],
                'traditional_match_percentage': traditional_detail['match_percentage'],
                'combined_match_percentage': combined_detail['match_percentage'],
                'traditional_matched_answers': traditional_detail['matched_answers'],
                'combined_matched_answers': combined_detail['matched_answers'],
                'traditional_llm_correct': traditional_detail['llm_is_correct'],
                'combined_llm_correct': combined_detail['llm_is_correct'],
                'traditional_time': traditional_result.get('total_time', 0),
                'combined_time': combined_result.get('total_time', 0),
                'exact_match_changed': traditional_detail['is_exact_match'] != combined_detail['is_exact_match'],
                'time_difference': combined_result.get('total_time', 0) - traditional_result.get('total_time', 0)
            }
            
            question_comparisons.append(comparison)
        
        return question_comparisons
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report comparing all approaches and answer types.
        
        Returns:
            Dictionary containing the complete analysis report
        """
        report = {
            'metadata': {
                'base_directory': str(self.base_dir),
                'analysis_type': 'exact_match_text_comparison_traditional_rag',
                'description': f'Exact match calculated by checking if at least {self.match_threshold*100}% of reference answers are contained in generated answer',
                'match_threshold': self.match_threshold,
                'approaches_compared': ['TraditionalRag', 'TraditionalRag_CombinedRetrieve']
            },
            'comparisons': {},
            'question_level_analysis': {}
        }
        
        # Analyze both answer types
        for answer_type in ['exact_answers', 'ideal_answers']:
            try:
                # High-level comparison
                comparison = self.compare_approaches(answer_type)
                report['comparisons'][answer_type] = comparison
                
                # Question-level analysis
                question_analysis = self.analyze_question_level_differences(answer_type)
                report['question_level_analysis'][answer_type] = question_analysis
                
            except FileNotFoundError as e:
                print(f"Warning: Could not analyze {answer_type}: {e}")
                continue
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = None):
        """
        Save the analysis report to a JSON file.
        
        Args:
            report: The report dictionary to save
            output_file: Output file path (optional)
        """
        if output_file is None:
            output_file = self.base_dir / "exact_match_traditional_analysis_report.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to: {output_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """
        Print a human-readable summary of the analysis.
        
        Args:
            report: The analysis report
        """
        print("=" * 80)
        print("EXACT MATCH ANALYSIS SUMMARY - TRADITIONAL RAG COMPARISON")
        print(f"(Threshold: {self.match_threshold*100}% of reference answers must match)")
        print("=" * 80)
        
        for answer_type, comparison in report['comparisons'].items():
            print(f"\n{answer_type.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            traditional = comparison['traditional']
            combined = comparison['combined_retrieve']
            comp = comparison['comparison']
            
            print(f"Traditional RAG:")
            print(f"  - Exact Match: {traditional['accuracy_percentage']:.2f}% ({traditional['exact_matches']}/{traditional['total_questions']})")
            print(f"  - Partial Matches: {traditional['partial_matches']}")
            print(f"  - Avg Match %: {traditional['avg_match_percentage']*100:.2f}%")
            print(f"  - LLM Accuracy: {traditional['comparison_with_llm']['llm_accuracy']*100:.2f}%")
            print(f"  - Agreement Rate: {traditional['comparison_with_llm']['agreement_rate']*100:.2f}%")
            print(f"  - Avg Time: {traditional['timing']['avg_total_time']:.2f}s")
            
            print(f"\nTraditional RAG + Combined Retrieve:")
            print(f"  - Exact Match: {combined['accuracy_percentage']:.2f}% ({combined['exact_matches']}/{combined['total_questions']})")
            print(f"  - Partial Matches: {combined['partial_matches']}")
            print(f"  - Avg Match %: {combined['avg_match_percentage']*100:.2f}%")
            print(f"  - LLM Accuracy: {combined['comparison_with_llm']['llm_accuracy']*100:.2f}%")
            print(f"  - Agreement Rate: {combined['comparison_with_llm']['agreement_rate']*100:.2f}%")
            print(f"  - Avg Time: {combined['timing']['avg_total_time']:.2f}s")
            
            print(f"\nComparison:")
            print(f"  - Exact Match Difference: {comp['exact_match_difference']:.4f} ({comp['exact_match_improvement_percentage']:+.2f}%)")
            print(f"  - Time Difference: {comp['time_difference']:+.2f}s ({comp['time_improvement_percentage']:+.2f}%)")
            print(f"  - Combined Better Exact Match: {'Yes' if comp['combined_better_exact_match'] else 'No'}")
            print(f"  - Combined Faster: {'Yes' if comp['combined_is_faster'] else 'No'}")


def main():
    """Main function to run the exact match analysis."""
    parser = argparse.ArgumentParser(description='Calculate exact match metrics for Traditional RAG results')
    parser.add_argument('base_dir', help='Base directory containing TraditionalRag and TraditionalRag_CombinedRetrieve folders')
    parser.add_argument('--output', '-o', help='Output file for the detailed report (JSON)')
    parser.add_argument('--summary-only', action='store_true', help='Only print summary, do not save detailed report')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Match threshold (default: 0.5 = 50%%)')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = ExactMatchCalculator(args.base_dir, args.threshold)
    
    try:
        # Generate report
        print("Analyzing exact match metrics for Traditional RAG approaches...")
        report = calculator.generate_report()
        
        # Print summary
        calculator.print_summary(report)
        
        # Save detailed report unless summary-only is specified
        if not args.summary_only:
            calculator.save_report(report, args.output)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
