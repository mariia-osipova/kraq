"""
RAG utility functions for Speculative RAG
"""
from typing import Any, Tuple, List, Dict
import asyncio
from statistics import mean
import numpy as np
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class RagDraftingResponse(BaseModel):
    rationale: str = Field(description="Response rationale.")
    response: str = Field(description="Response to the instruction.")


async def rag_drafting_generator(
    query: str,
    context: List[str],
    client: AsyncOpenAI,
    model: str,
) -> Dict[str, Any]:
    """
    Generate RAG draft with rationale and probabilities
    """
    prompt = """Based on the evidence provided, generate a response to the instruction. 
Include both a rationale explaining your reasoning and a final response.

Instruction: {instruction}
Evidence:
{evidence}"""

    # Format evidence with numbered citations
    formatted_evidence = "\n".join(
        f"[{i+1}] {doc}" for i, doc in enumerate(context)
    )
    
    messages = [
        {
            "role": "user",
            "content": prompt.format(
                instruction=query,
                evidence=formatted_evidence
            )
        }
    ]

    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format=RagDraftingResponse,
        seed=42,
        logprobs=True,
        top_logprobs=1
    )
    
    draft_response = completion.choices[0].message.parsed
    
    # Obtener logprobs de manera más segura
    try:
        logprobs = completion.choices[0].logprobs
        if hasattr(logprobs, 'content'):
            # Obtener todos los tokens y sus logprobs
            tokens = [token for token in logprobs.content if hasattr(token, 'logprob')]
            
            # Encontrar los índices donde comienzan "rationale" y "response" en el JSON
            rationale_start = next(i for i, t in enumerate(tokens) if 'ationale' in t.token) #El token de rationale es "ationale"
            response_start = next(i for i, t in enumerate(tokens) if 'response' in t.token) #El token de response es "response"
            
            # Obtener los logprobs específicos para cada sección
            rationale_logprobs = [t.logprob for t in tokens[rationale_start:response_start]]
            response_logprobs = [t.logprob for t in tokens[response_start:]]
            
            # Calcular las probabilidades
            p_rationale = np.exp(mean(rationale_logprobs))
            p_response = np.exp(mean(response_logprobs))
            p_consistency = np.exp(mean([t.logprob for t in tokens]))
        else:
            print("\nNo content attribute in logprobs")
            p_rationale = p_response = 0.0
            
    except Exception as e:
        print(f"\nError processing logprobs: {str(e)}")
        p_rationale = p_response = 0.0
        tokens = [token for token in logprobs.content if hasattr(token, 'logprob')]
        p_consistency = np.exp(mean([t.logprob for t in tokens]))
    
    p_draft = p_rationale + p_response
    
    return {
        "draft": draft_response,  # Ahora es un objeto RagDraftingResponse
        "p_draft": p_draft,
        "p_consistency": p_consistency,
        "rationale": draft_response.rationale,
        "response": draft_response.response
    }

async def rag_verifier_generator(
    query: str,
    drafts: List[Dict[str, str]],
    context: List[str],
    client: AsyncOpenAI,
    model: str,
) -> Dict[str, Any]:
    """
    Verify RAG drafts and return probabilities
    """
    verifier_prompt = """Instruction: {instruction}

Response: {response}

Rationale: {rationale}

Is the rationale good enough to support the answer? (Yes or No)"""

    verify_tasks = []
    for draft in drafts:
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
        
        verify_tasks.append(
            client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                logprobs=True
            )
        )
    
    verifications = await asyncio.gather(*verify_tasks)
    
    # Get probabilities for "Yes" responses
    p_yes = []
    for verification in verifications:
        response = verification.choices[0].message.content.strip().lower()
        print(f"Verificacion: {response}")
        
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
            print(f"Error processing verification logprobs: {str(e)}")
            p_yes.append(0.5)
    
    # Return best draft and its probability
    best_idx = np.argmax(p_yes)
    return {
        "response": drafts[best_idx]["response"],
        "rationale": drafts[best_idx]["rationale"],
        "p_yes": p_yes
    }


async def speculative_rag_with_probabilities(
    query: str,
    drafts_contexts: List[List[str]],
    client: AsyncOpenAI,
    drafter_model: str,
    verifier_model: str,
    all_context: List[str],
    debug_prints: bool = False
) -> Dict[str, Any]:
    """
    Performs Speculative RAG considering both draft and verification probabilities
    
    Args:
        query: User query
        drafts_contexts: List of context sets for each draft
        client: OpenAI client
        drafter_model: Model to use for drafting
        verifier_model: Model to use for verification
        all_context: All context documents (for verification)
        debug_prints: Whether to print debug information
        
    Returns:
        Dictionary with best draft, its p_yes, and combined score
    """
    # Generate drafts for each context set
    draft_tasks = []
    for context_set in drafts_contexts:
        draft_task = rag_drafting_generator(
            query=query,
            context=context_set,
            client=client,
            model=drafter_model
        )
        draft_tasks.append(draft_task)
    
    # Await all drafts
    draft_results = await asyncio.gather(*draft_tasks)
    
    # Separate drafts and their probabilities
    drafts = [result for result in draft_results]
    
    if debug_prints:
        print("\n📝 GENERATED DRAFTS WITH PROBABILITIES:")
        for i, draft in enumerate(drafts):
            print(f"\nDraft {i+1}:")
            print(f"Rationale: {draft['rationale']}")
            print(f"Response: {draft['response'][:200]}...")
        print("-" * 80)
    
    # Verify all drafts
    verify_response = await rag_verifier_generator(
        query=query,
        drafts=drafts,
        context=all_context,
        client=client,
        model=verifier_model
    )
    
    # Get p_yes
    p_yes = verify_response["p_yes"]
    
    # Calculate combined scores (weighted average of draft probability and p_yes)
    # Weight factors can be adjusted as needed
    combined_scores = []
    for i, draft in enumerate(drafts):
        # Combined score: 30% draft generation probability, 70% verification p_yes
        combined_score =  draft["p_draft"] * p_yes[i] * draft["p_consistency"]
        combined_scores.append(combined_score)
    
    # Find the draft with the highest combined score
    best_draft_idx = np.argmax(combined_scores)
    best_draft = drafts[best_draft_idx]
    best_score = combined_scores[best_draft_idx]
    
    if debug_prints:
        print("\n🏆 DRAFT SELECTION:")
        print(f"p_yes: {p_yes:.4f}")
        for i, (draft, combined_score) in enumerate(zip(drafts, combined_scores)):
            is_best = "✓" if i == best_draft_idx else " "
            print(f"Draft {i+1}: Rationale={draft['rationale']}, Response={draft['response'][:200]}... Combined Score={combined_score:.4f} {is_best}")
        print("-" * 80)
    
    result = {
        "best_draft": best_draft,
        "p_yes": p_yes,
        "combined_score": best_score,
        "drafts": drafts,
        "combined_scores": combined_scores,
        "verifier_response": verify_response["response"],
        "best_draft_idx": best_draft_idx
    }
    
    return result 