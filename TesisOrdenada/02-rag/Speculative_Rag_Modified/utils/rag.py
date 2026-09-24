"""
RAG utility functions for Speculative RAG
"""
from typing import Any, Tuple, List, Dict
import asyncio
from statistics import mean
import numpy as np
import json
import traceback
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from loguru import logger
import time

# Configuración para vLLM local
VLLM_BASE_URL = "http://localhost:8000/v1"

class RagDraftingResponse(BaseModel):
    rationale: str = Field(description="Response rationale.")
    response: str = Field(description="Response to the instruction.")


async def rag_drafting_generator(
    query: str,
    context: List[str],
    client: AsyncOpenAI = None,
    model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
) -> Dict[str, Any]:
    """
    Generate RAG draft with rationale and probabilities
    """
    # Si no se proporciona cliente, crear uno que apunte a vLLM local
    if client is None:
        client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed"
        )
    
    prompt = """Response to the instruction. Also provide a concise rationale that justifies the response.

###Instruction: 
{instruction}
###Evidence:
{evidence}

Your response must be a valid JSON object with the following format:
{{"response": "your response here", "rationale": "your rationale here"}}
"""

    # Format evidence with numbered citations
    formatted_evidence = "\n".join(
        f"[{i+1}] {doc}" for i, doc in enumerate(context)
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You can only use the evidence provided to answer the question."},
        {
            "role": "user",
            "content": prompt.format(
                instruction=query,
                evidence=formatted_evidence
            )
        }
    ]

    logger.debug(f"Generando draft con modelo {model}")
    start_time = time.time()
    
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            seed=42,
            max_tokens=1200,
            logprobs=True,
            top_logprobs=1
        )
        
        response_content = completion.choices[0].message.content.strip()
        
        # Extraer el JSON de la respuesta con manejo mejorado
        rationale = "Error parsing rationale"
        response = "Error parsing response"
        
        try:
            # Intento 1: Cargar directamente como JSON
            draft_dict = json.loads(response_content)
            rationale = draft_dict.get("rationale", "")
            response = draft_dict.get("response", "")
        except json.JSONDecodeError:
            logger.warning(f"JSON directo inválido, intentando extraer: {response_content[:100]}...")
            
            try:
                # Intento 2: Buscar el objeto JSON dentro del texto
                # Buscar donde comienza el objeto JSON (primer '{')
                start_idx = response_content.find('{')
                # Buscar donde termina el objeto JSON (último '}')
                end_idx = response_content.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = response_content[start_idx:end_idx+1]
                    draft_dict = json.loads(json_str)
                    rationale = draft_dict.get("rationale", "")
                    response = draft_dict.get("response", "")
                else:
                    # Intento 3: Extraer manualmente las claves "response" y "rationale"
                    response_start = response_content.find('"response"')
                    if response_start != -1:
                        # Buscar el valor después de "response":
                        value_start = response_content.find(':', response_start) + 1
                        # Eliminar espacios en blanco
                        while value_start < len(response_content) and response_content[value_start].isspace():
                            value_start += 1
                        # Comprobar si el valor comienza con comillas
                        if value_start < len(response_content) and response_content[value_start] == '"':
                            # Buscar la comilla de cierre
                            value_end = response_content.find('"', value_start + 1)
                            if value_end != -1:
                                response = response_content[value_start+1:value_end]
                    
                    rationale_start = response_content.find('"rationale"')
                    if rationale_start != -1:
                        # Buscar el valor después de "rationale":
                        value_start = response_content.find(':', rationale_start) + 1
                        # Eliminar espacios en blanco
                        while value_start < len(response_content) and response_content[value_start].isspace():
                            value_start += 1
                        # Comprobar si el valor comienza con comillas
                        if value_start < len(response_content) and response_content[value_start] == '"':
                            # Buscar la comilla de cierre
                            value_end = response_content.find('"', value_start + 1)
                            if value_end != -1:
                                rationale = response_content[value_start+1:value_end]
                    
                    # Si todavía no tenemos una respuesta válida, usar toda la respuesta como respuesta
                    if response == "Error parsing response":
                        # Caso de emergencia: usar toda la respuesta y marcar como se pudo
                        response = response_content
                        rationale = "Extracted from raw response"
                        
            except Exception as e:
                logger.error(f"Error en extracción manual de JSON: {str(e)}")
                # Caso de emergencia: usar toda la respuesta
                response = response_content
                rationale = "Failed to extract JSON structure"
        
        # Para mantener compatibilidad con el resto del código
        draft_response = {
            "rationale": rationale,
            "response": response
        }
        
        # Obtener logprobs de manera más segura
        try:
            logprobs = completion.choices[0].logprobs
            if hasattr(logprobs, 'content'):
                # Obtener todos los tokens y sus logprobs
                tokens = [token for token in logprobs.content if hasattr(token, 'logprob')]
                token_texts = [t.token for t in tokens if hasattr(t, 'token')]
                
                # Buscar patrones específicos basados en el análisis del tokenizador
                rationale_indices = []
                response_indices = []
                
                # Buscar patrones para 'rationale'
                rationale_patterns = ['ration', 'rational', ' rationale']
                for i, token_text in enumerate(token_texts):
                    for pattern in rationale_patterns:
                        if pattern in token_text.lower():
                            rationale_indices.append(i)
                            break  # Solo agregar una vez por token
                
                # Buscar patrones para 'response'
                response_patterns = ['response', ' response', 'resp']
                for i, token_text in enumerate(token_texts):
                    for pattern in response_patterns:
                        if pattern in token_text.lower():
                            response_indices.append(i)
                            break  # Solo agregar una vez por token
                
                # Filtrar y ordenar los índices encontrados
                if rationale_indices and response_indices:
                    # Tomar el primer índice de cada campo
                    first_rationale = min(rationale_indices)
                    first_response = min(response_indices)
                    
                    # Determinar qué campo viene primero
                    if first_rationale < first_response:
                        # 'rationale' viene primero, luego 'response'
                        rationale_logprobs = [t.logprob for t in tokens[first_rationale:first_response]]
                        response_logprobs = [t.logprob for t in tokens[first_response:]]
                    else:
                        # 'response' viene primero, luego 'rationale'
                        response_logprobs = [t.logprob for t in tokens[first_response:first_rationale]]
                        rationale_logprobs = [t.logprob for t in tokens[first_rationale:]]
                else:
                    # Si no se encuentran ambos campos, dividir por la mitad
                    mid_point = len(tokens) // 2
                    rationale_logprobs = [t.logprob for t in tokens[:mid_point]]
                    response_logprobs = [t.logprob for t in tokens[mid_point:]]
                
                # Calcular las probabilidades
                p_rationale = np.exp(mean(rationale_logprobs)) if rationale_logprobs else 0.5
                p_response = np.exp(mean(response_logprobs)) if response_logprobs else 0.5
                p_consistency = np.exp(mean([t.logprob for t in tokens])) if tokens else 0.5
                
            else:
                logger.warning("No content attribute in logprobs")
                p_rationale = p_response = 0.5
                p_consistency = 0.5
                
        except Exception as e:
            logger.error(f"Error processing logprobs: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            p_rationale = p_response = 0.5
            p_consistency = 0.5
    
    except Exception as e:
        logger.error(f"Error generando draft: {str(e)}")
        # Crear una respuesta por defecto en caso de error
        draft_response = {
            "rationale": "Error generating rationale.",
            "response": "Error generating response."
        }
        p_rationale = p_response = p_consistency = 0.1
    
    elapsed = time.time() - start_time
    logger.debug(f"Draft generado en {elapsed:.2f} segundos")
    
    p_draft = p_rationale + p_response
    
    return {
        "draft": draft_response,
        "p_draft": p_draft,
        "p_consistency": p_consistency,
        "rationale": draft_response["rationale"],
        "response": draft_response["response"]
    }

async def rag_verifier_generator(
    query: str,
    drafts: List[Dict[str, str]],
    context: List[str],
    client: AsyncOpenAI = None,
    model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
) -> Dict[str, Any]:
    """
    Verify RAG drafts sequentially and return probabilities with timing information
    """
    # Si no se proporciona cliente, crear uno que apunte a vLLM local
    if client is None:
        client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed"
        )
    
    verifier_prompt = """Instruction: {instruction}

Response: {response}

Rationale: {rationale}

Is the rationale good enough to support the answer? 

You must respond with only a single word: "Yes" or "No".
Do not include any explanation or additional text."""

    logger.debug(f"Verificando {len(drafts)} drafts secuencialmente con modelo {model}")
    start_time = time.time()
    
    # Ejecutar verificaciones de manera secuencial y registrar tiempos individuales
    verifications = []
    verification_times = []
    
    for i, draft in enumerate(drafts):
        messages = [
            {
                "role": "user",
                "content": verifier_prompt.format(
                    instruction=query,
                    response=draft["response"],
                    rationale=draft["rationale"]
                )
            }
        ]
        
        # Medir tiempo de esta verificación individual
        single_start_time = time.time()
        
        # Ejecutar verificación
        verification = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            logprobs=True,
            max_tokens=10
        )
        
        # Registrar tiempo
        single_elapsed = time.time() - single_start_time
        verification_times.append(single_elapsed)
        verifications.append(verification)
        
        logger.debug(f"Verificación {i+1}/{len(drafts)} completada en {single_elapsed:.2f} segundos")
    
    # Calcular estadísticas de tiempo
    total_elapsed = time.time() - start_time
    avg_verification_time = sum(verification_times) / len(verification_times) if verification_times else 0
    parallel_estimate = max(verification_times) if verification_times else 0
    
    logger.debug(f"Verificaciones secuenciales completadas en {total_elapsed:.2f} segundos")
    logger.debug(f"Tiempo promedio por verificación: {avg_verification_time:.2f} segundos")
    logger.debug(f"Tiempo estimado en paralelo: {parallel_estimate:.2f} segundos")
    
    # Get probabilities for "Yes" responses
    p_yes = []
    for verification in verifications:
        response = verification.choices[0].message.content.strip().lower()
        logger.debug(f"Resultado de verificación: {response}")
        
        try:
            logprobs = verification.choices[0].logprobs
            if hasattr(logprobs, 'content'):
                # Obtener todos los logprobs de la respuesta
                all_logprobs = [token.logprob for token in logprobs.content if hasattr(token, 'logprob')]
                # Calcular la probabilidad promedio
                avg_logprob = mean(all_logprobs)
                if response == "yes":
                    p_yes.append(np.exp(avg_logprob))
                else:
                    p_yes.append(1 - np.exp(avg_logprob))
            else:
                p_yes.append(0.5)
        except Exception as e:
            logger.error(f"Error processing verification logprobs: {str(e)}")
            p_yes.append(0.5)
    
    # Return best draft, its probability, and timing information
    best_idx = np.argmax(p_yes)
    return {
        "response": drafts[best_idx]["response"],
        "rationale": drafts[best_idx]["rationale"],
        "p_yes": p_yes,
        "sequential_time": total_elapsed,
        "average_time": avg_verification_time,
        "parallel_estimate": parallel_estimate,
        "individual_times": verification_times
    }


async def speculative_rag_with_probabilities(
    query: str,
    drafts_contexts: List[List[str]],
    client: AsyncOpenAI = None,
    drafter_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    verifier_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    all_context: List[str] = None,
    debug_prints: bool = False,
    drafter_client: AsyncOpenAI = None,
    verifier_client: AsyncOpenAI = None
) -> Dict[str, Any]:
    """
    Performs Speculative RAG considering both draft and verification probabilities
    
    Args:
        query: User query
        drafts_contexts: List of context sets for each draft
        client: OpenAI client (opcional, si no se proporciona se usará vLLM local)
        drafter_model: Model to use for drafting
        verifier_model: Model to use for verification
        all_context: All context documents (for verification)
        debug_prints: Whether to print debug information
        drafter_client: Client for drafter model (if None, uses client)
        verifier_client: Client for verifier model (if None, uses client)
        
    Returns:
        Dictionary with best draft, its p_yes, combined score, and timing information
    """
    # Si no se proporciona cliente, crear uno que apunte a vLLM local
    if client is None:
        client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed"
        )
    
    # Configurar clientes específicos si no se proporcionan
    if drafter_client is None:
        drafter_client = client
    
    if verifier_client is None:
        verifier_client = client
    
    # Si all_context es None, usar el primer conjunto de contextos
    if all_context is None:
        all_context = []
        for context_set in drafts_contexts:
            all_context.extend(context_set)
    
    # Generate drafts sequentially, not in parallel
    start_time = time.time()
    draft_results = []
    draft_times = []
    
    for i, context_set in enumerate(drafts_contexts):
        single_start_time = time.time()
        
        # Generate a single draft
        draft_result = await rag_drafting_generator(
            query=query,
            context=context_set,
            client=drafter_client,
            model=drafter_model
        )
        
        # Record time for this draft
        single_elapsed = time.time() - single_start_time
        draft_times.append(single_elapsed)
        draft_results.append(draft_result)
        
        logger.debug(f"Draft {i+1}/{len(drafts_contexts)} generado en {single_elapsed:.2f} segundos")
    
    total_drafting_time = time.time() - start_time
    avg_draft_time = sum(draft_times) / len(draft_times) if draft_times else 0
    parallel_draft_estimate = max(draft_times) if draft_times else 0
    
    logger.debug(f"Drafts secuenciales completados en {total_drafting_time:.2f} segundos")
    logger.debug(f"Tiempo promedio por draft: {avg_draft_time:.2f} segundos")
    logger.debug(f"Tiempo estimado en paralelo: {parallel_draft_estimate:.2f} segundos")
    
    # Separate drafts and their probabilities
    drafts = draft_results
    
    if debug_prints:
        print("\n📝 GENERATED DRAFTS WITH PROBABILITIES:")
        for i, draft in enumerate(drafts):
            print(f"\nDraft {i+1}:")
            print(f"Rationale: {draft['rationale']}")
            print(f"Response: {draft['response'][:200]}...")
        print("-" * 80)
    
    # Verify all drafts sequentially
    verify_response = await rag_verifier_generator(
        query=query,
        drafts=drafts,
        context=all_context,
        client=verifier_client,
        model=verifier_model
    )
    
    # Get p_yes
    p_yes = verify_response["p_yes"]
    
    # Calculate combined scores
    combined_scores = []
    for i, draft in enumerate(drafts):
        # Combined score based on all factors
        combined_score = draft["p_draft"] * p_yes[i] * draft["p_consistency"]
        combined_scores.append(combined_score)
    
    # Find the draft with the highest combined score
    best_draft_idx = np.argmax(combined_scores)
    best_draft = drafts[best_draft_idx]
    best_score = combined_scores[best_draft_idx]
    
    if debug_prints:
        print("\n🏆 DRAFT SELECTION:")
        for i, (draft, score) in enumerate(zip(drafts, combined_scores)):
            print(f"\nDraft {i+1}:")
            print(f"Draft probability: {draft['p_draft']:.4f}")
            print(f"Consistency: {draft['p_consistency']:.4f}")
            print(f"Verifier p_yes: {p_yes[i]:.4f}")
            print(f"Combined score: {score:.4f}")
            print("-" * 40)
            print(f"Rationale: {draft['rationale'][:100]}...")
            print(f"Response: {draft['response'][:100]}...")
            print("-" * 40)
        
        print(f"\nBest draft: {best_draft_idx+1}")
        print(f"Score: {best_score:.4f}")
        
        # Print timing info
        print("\n⏱️ TIMING INFORMATION:")
        print(f"Drafting secuencial total: {total_drafting_time:.2f} segundos")
        print(f"Tiempo promedio por draft: {avg_draft_time:.2f} segundos")
        print(f"Estimación en paralelo para drafts: {parallel_draft_estimate:.2f} segundos")
        print(f"Verificación secuencial total: {verify_response.get('sequential_time', 0):.2f} segundos")
        print(f"Tiempo promedio por verificación: {verify_response.get('average_time', 0):.2f} segundos")
        print(f"Estimación en paralelo para verificación: {verify_response.get('parallel_estimate', 0):.2f} segundos")
    
    return {
        "response": best_draft["response"],
        "rationale": best_draft["rationale"],
        "p_yes": p_yes[best_draft_idx],
        "p_draft": best_draft["p_draft"],
        "p_consistency": best_draft["p_consistency"],
        "combined_score": best_score,
        "draft_sequential_time": total_drafting_time,
        "draft_average_time": avg_draft_time,
        "draft_parallel_estimate": parallel_draft_estimate,
        "verify_sequential_time": verify_response.get("sequential_time", 0),
        "verify_average_time": verify_response.get("average_time", 0),
        "verify_parallel_estimate": verify_response.get("parallel_estimate", 0),
        "draft_individual_times": draft_times,
        "verify_individual_times": verify_response.get("individual_times", [])
    } 