import json
import os
import sys
import numpy as np
from tqdm import tqdm
import time
import requests
from typing import Dict, List, Tuple, Set
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Add src to path for correct module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Qdrant modules
from src.DataBase.qdrant_config import get_configured_qdrant_client, get_embedding_from_ollama, COLLECTIONS

# Collection name for finetuned questions
COLLECTION_NAME = COLLECTIONS["random"]
VLLM_API_URL = "http://localhost:8000/v1"
client_llm = OpenAI(base_url=VLLM_API_URL, api_key="EMPTY")

def extract_entities(text: str) -> List[str]:
    """
    Extract entities from text using vLLM API through OpenAI client.
    """
    system_prompt = f"""
### System
You are an expert information-extraction assistant. Your task is STRICTLY named-entity recognition (NER) over a single natural-language QUESTION.

Return **ONLY** valid JSON with this schema: {{ "entities": [string, …] }}  
Rules:
• Include ONLY People, Geographic Locations, and Events.
• People: individual names, groups of people, historical figures, etc.
• Geographic Locations: countries, cities, mountains, rivers, regions, etc.
• Events: historical events, dates, time periods, celebrations, etc.
• Preserve original spelling/capitalisation.
• No extra text or explanations.
• If nothing qualifies, return {{ "entities": [] }}.

### Few-shot
User: Question: "How tall is Mount Everest compared to Mount Kilimanjaro?"
Assistant: {{ "entities": ["Mount Everest", "Mount Kilimanjaro"] }}

User: Question: "When did Christopher Columbus arrive in America?"
Assistant: {{ "entities": ["Christopher Columbus", "America"] }}

User: Question: "Who was the president during World War II?"
Assistant: {{ "entities": ["World War II"] }}

User: Question: "What happened during the Battle of Waterloo?"
Assistant: {{ "entities": ["Battle of Waterloo"] }}

User: Question: "Where did Albert Einstein work in 1905?"
Assistant: {{ "entities": ["Albert Einstein", "1905"] }}"""
    
    user_prompt = f"""User: Question: "{text}"
"""
    
    try:
        response = client_llm.chat.completions.create(
            model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",  # este es el default para vLLM
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        # Obtener la respuesta del modelo
        entities_text = response.choices[0].message.content.strip()
        
        try:
            # Intentar parsear la respuesta como JSON
            entities = json.loads(entities_text)
            
            # Validar que sea una lista de strings
            if not isinstance(entities, list) or not all(isinstance(x, str) for x in entities):
                print(f"Invalid response format. Expected list of strings, got: {entities_text}")
                return []
            
            # Limpiar y normalizar entidades
            cleaned_entities = [
                entity.strip()
                for entity in entities
                if entity.strip() and len(entity.strip()) > 1  # ignorar entidades vacías o de un solo carácter
            ]
            
            return cleaned_entities
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {entities_text}")
            print(f"Error details: {str(e)}")
            return []
            
    except Exception as e:
        print(f"Error calling vLLM API: {str(e)}")
        return []


def find_most_similar_question(query_question: str, client) -> Dict:
    """Find the most similar question in the Qdrant collection."""
    try:
        query_vector = get_embedding_from_ollama(query_question)
        
        if not query_vector:
            print(f"Error: Could not generate embedding for query: {query_question}")
            return None
        
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,
            with_payload=True
        )
        
        if search_result:
            match = search_result[0]
            return {
                "benchmark_question": query_question,
                "representative_question": match.payload["question"],
                "cosine_similarity": match.score,
                "qdrant_id": match.id,
                "question_type": match.payload.get("type", "unknown"),
                "level": match.payload.get("level", "unknown")
            }
        return None
            
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        return None

def load_benchmark_questions():
    """Load questions from local benchmark file."""
    benchmark_path = os.path.join(project_root, "src", "Benchmarks", "qa_benchmark.json")
    
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    return [item['question'] for item in qa_data]

async def extract_entities_async(text: str) -> List[str]:
    """
    Versión asíncrona de extract_entities que permite paralelización.
    """
    system_prompt = f"""
### System
You are an expert information-extraction assistant. Your task is STRICTLY named-entity recognition (NER) over a single natural-language QUESTION.

Return **ONLY** valid JSON with this schema: {{ "entities": [string, …] }}  
Rules:
• Include ONLY People, Geographic Locations, and Events.
• People: individual names, groups of people, historical figures, etc.
• Geographic Locations: countries, cities, mountains, rivers, regions, etc.
• Events: historical events, dates, time periods, celebrations, etc.
• Preserve original spelling/capitalisation.
• No extra text or explanations.
• If nothing qualifies, return {{ "entities": [] }}.

### Few-shot
User: Question: "How tall is Mount Everest compared to Mount Kilimanjaro?"
Assistant: {{ "entities": ["Mount Everest", "Mount Kilimanjaro"] }}

User: Question: "When did Christopher Columbus arrive in America?"
Assistant: {{ "entities": ["Christopher Columbus", "America"] }}

User: Question: "Who was the president during World War II?"
Assistant: {{ "entities": ["World War II"] }}

User: Question: "What happened during the Battle of Waterloo?"
Assistant: {{ "entities": ["Battle of Waterloo"] }}

User: Question: "Where did Albert Einstein work in 1905?"
Assistant: {{ "entities": ["Albert Einstein", "1905"] }}"""
    
    user_prompt = f"""User: Question: "{text}"
"""
    
    try:
        response = await asyncio.to_thread(
            client_llm.chat.completions.create,
            model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        
        # Obtener la respuesta del modelo
        entities_text = response.choices[0].message.content.strip()
        
        try:
            # Intentar parsear la respuesta como JSON
            entities_json = json.loads(entities_text)
            
            # Extraer la lista de entidades
            if "entities" in entities_json and isinstance(entities_json["entities"], list):
                entities = entities_json["entities"]
                
                # Validar que sea una lista de strings
                if not all(isinstance(x, str) for x in entities):
                    print(f"Invalid entity types in response: {entities_text}")
                    return []
                
                return entities
            else:
                print(f"Missing 'entities' field in response: {entities_text}")
                return []
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {entities_text}")
            print(f"Error details: {str(e)}")
            return []
            
    except Exception as e:
        print(f"Error calling vLLM API: {str(e)}")
        return []

async def analyze_entity_overlap_with_llm(benchmark_entities: List[str], representative_entities: List[str]) -> Dict:
    """
    Utiliza el LLM para analizar el solapamiento semántico entre dos listas de entidades.
    """
    if not benchmark_entities or not representative_entities:
        return {
            "benchmark_entities": benchmark_entities,
            "representative_entities": representative_entities,
            "common_entities": [],
            "overlap_count": 0,
            "benchmark_entity_count": len(benchmark_entities),
            "representative_entity_count": len(representative_entities),
            "overlap_ratio": 0
        }
    
    system_prompt = f"""
You are an assistant that compares two sets of entities and decides how many refer to the SAME underlying real-world concept.

Given JSON containing two arrays, respond with JSON ONLY, using the schema:
{{"overlap_count": integer}}.

• A match exists when two strings are synonyms, abbreviations, alternate spellings, or clearly identical (case-insensitive, ignoring punctuation).
• Treat geopolitical synonyms (e.g., "U.S.", "USA", "United States") as equal.
• People: compare full name vs surname or common nickname.
• Organisations/products: compare official vs abbreviated names (e.g., "International Business Machines" vs "IBM").

Return ONLY the count of such matches as "overlap_count". Do NOT return the matching pairs.
If no overlap exists, return {{"overlap_count": 0}}.
No additional text.

Examples:
Input:
{{"entities_q1": ["USA", "President Joe Biden", "Mount Everest"],
  "entities_q2": ["United States", "Biden", "K2"]}}

Output:
{{"overlap_count": 2}}

Input:
{{"entities_q1": ["Google", "Android"],
  "entities_q2": ["Apple", "iOS"]}}

Output:
{{"overlap_count": 0}}
"""

    # Crear el payload JSON para el prompt
    input_json = {
        "entities_q1": benchmark_entities,
        "entities_q2": representative_entities
    }
    
    user_prompt = json.dumps(input_json)
    
    try:
        response = await asyncio.to_thread(
            client_llm.chat.completions.create,
            model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        # Obtener la respuesta del modelo
        result_text = response.choices[0].message.content.strip()
        
        try:
            # Intentar parsear la respuesta como JSON
            result_json = json.loads(result_text)
            
            # Extraer el contador de overlap
            if "overlap_count" in result_json and isinstance(result_json["overlap_count"], (int, float)):
                overlap_count = int(result_json["overlap_count"])
                
                # Validar el valor (no puede ser mayor que el número de entidades en benchmark)
                if overlap_count > len(benchmark_entities):
                    print(f"Invalid overlap count: {overlap_count} > {len(benchmark_entities)}")
                    overlap_count = len(benchmark_entities)
                
                # Calcular ratio
                overlap_ratio = overlap_count / len(benchmark_entities) if benchmark_entities else 0
                
                return {
                    "benchmark_entities": benchmark_entities,
                    "representative_entities": representative_entities,
                    "common_entities": [],  # No tenemos detalle de cuáles son las entidades comunes
                    "overlap_count": overlap_count,
                    "benchmark_entity_count": len(benchmark_entities),
                    "representative_entity_count": len(representative_entities),
                    "overlap_ratio": overlap_ratio
                }
            else:
                print(f"Missing or invalid 'overlap_count' in response: {result_text}")
                return calculate_entity_overlap(benchmark_entities, representative_entities)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {result_text}")
            print(f"Error details: {str(e)}")
            return calculate_entity_overlap(benchmark_entities, representative_entities)
            
    except Exception as e:
        print(f"Error calling LLM for entity overlap analysis: {str(e)}")
        return calculate_entity_overlap(benchmark_entities, representative_entities)

async def process_question_batch_async(questions: List[str], client_qdrant) -> List[Dict]:
    """
    Procesa un batch de preguntas en paralelo, enviando múltiples consultas a vLLM a la vez.
    """
    results = []
    
    # Primero encontrar las preguntas similares
    similar_questions = []
    for question in questions:
        similar = find_most_similar_question(question, client_qdrant)
        if similar:
            similar_questions.append(similar)
    
    if not similar_questions:
        return results
    
    # Preparar todas las preguntas para extracción de entidades (tanto benchmark como representativas)
    benchmark_questions = [q['benchmark_question'] for q in similar_questions]
    representative_questions = [q['representative_question'] for q in similar_questions]
    
    # Extraer entidades de todas las preguntas en paralelo
    benchmark_entities_tasks = [extract_entities_async(q) for q in benchmark_questions]
    representative_entities_tasks = [extract_entities_async(q) for q in representative_questions]
    
    # Ejecutar todas las tareas en paralelo
    benchmark_entities_results = await asyncio.gather(*benchmark_entities_tasks)
    representative_entities_results = await asyncio.gather(*representative_entities_tasks)
    
    # Crear tareas para analizar overlap con LLM
    overlap_tasks = [
        analyze_entity_overlap_with_llm(benchmark_entities_results[i], representative_entities_results[i]) 
        for i in range(len(similar_questions))
    ]
    overlap_results = await asyncio.gather(*overlap_tasks)
    
    # Combinar resultados
    for i, similar in enumerate(similar_questions):
        result = {
            **similar,
            **overlap_results[i],
            "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results.append(result)
        print(f"Processed pair with semantic overlap ratio: {result['overlap_ratio']:.4f}")
    
    return results

def calculate_and_save_entity_overlap():
    """Main function to calculate entity overlap between benchmark and representative questions."""
    print("=== Starting Entity Overlap Analysis (Finetuned Model) ===")
    
    print("\nConnecting to Qdrant...")
    client = get_configured_qdrant_client()
    
    # Check collection existence
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Error: Collection '{COLLECTION_NAME}' not found in Qdrant.")
            print(f"Available collections: {collection_names}")
            return None
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None
    
    print("\nLoading benchmark questions...")
    benchmark_questions = load_benchmark_questions()
    print(f"Loaded {len(benchmark_questions)} benchmark questions")
    
    batch_size = 50  # Procesar 30 preguntas en paralelo
    results = []
    
    try:
        # Crear event loop para procesamiento asíncrono
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(benchmark_questions), batch_size):
            batch = benchmark_questions[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{len(benchmark_questions)//batch_size + 1}")
            
            # Procesar batch de forma asíncrona
            batch_results = loop.run_until_complete(
                process_question_batch_async(batch, client)
            )
            
            results.extend(batch_results)
            print(f"Processed {len(batch_results)} questions in this batch")
            
            time.sleep(0.5)  # Pequeña pausa entre batches
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")
    finally:
        if results:
            # Calcular estadísticas
            overlap_ratios = [r['overlap_ratio'] for r in results]
            overlap_counts = [r['overlap_count'] for r in results]
            
            statistics = {
                "total_questions": len(results),
                "average_overlap_ratio": float(np.mean(overlap_ratios)),
                "median_overlap_ratio": float(np.median(overlap_ratios)),
                "average_overlap_count": float(np.mean(overlap_counts)),
                "median_overlap_count": float(np.median(overlap_counts)),
                "max_overlap_ratio": float(np.max(overlap_ratios)),
                "min_overlap_ratio": float(np.min(overlap_ratios)),
                "percentile_25_overlap": float(np.percentile(overlap_ratios, 25)),
                "percentile_75_overlap": float(np.percentile(overlap_ratios, 75))
            }
            
            # Preparar resultado final con información detallada
            final_result = {
                "collection_info": {
                    "collection_name": COLLECTION_NAME,
                    "model_type": "random",
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_benchmark_questions": len(benchmark_questions),
                    "successfully_processed": len(results)
                },
                "statistics": statistics,
                "detailed_results": results
            }
            
            # Guardar resultados
            os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
            output_path = os.path.join(project_root, "output", "entity_overlap_random.json")
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            print(f"\nResults saved to: {output_path}")
            
            # Imprimir estadísticas y resultados detallados
            print("\n=== Entity Overlap Statistics (Finetuned Model) ===")
            print(f"Average Overlap Ratio: {statistics['average_overlap_ratio']:.4f}")
            print(f"Median Overlap Ratio: {statistics['median_overlap_ratio']:.4f}")
            print(f"Average Overlap Count: {statistics['average_overlap_count']:.4f}")
            print(f"Total Questions Processed: {statistics['total_questions']}")
            
            print("\n=== Detailed Results ===")
            for result in results:
                print("\nQuestion Pair:")
                print(f"Benchmark: {result['benchmark_question']}")
                print(f"Representative: {result['representative_question']}")
                print(f"Cosine Similarity: {result['cosine_similarity']:.4f}")
                print(f"Overlap Ratio: {result['overlap_ratio']:.4f}")
                print("Benchmark Entities:", result['benchmark_entities'])
                print("Representative Entities:", result['representative_entities'])
                print("Common Entities:", result['common_entities'])
                print("-" * 80)
            
            return final_result
        return results

if __name__ == "__main__":
    print("Starting entity overlap benchmark for finetuned model")
    results = calculate_and_save_entity_overlap()