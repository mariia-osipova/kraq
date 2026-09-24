"""
Main implementation of Speculative RAG with Qdrant
"""
import asyncio
import numpy as np
import os
from time import perf_counter
from typing import Any, Dict, List, Optional
from pathlib import Path
from statistics import mean

from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

from SpeculativeRag.utils.clustering import multi_perspective_sampling
from SpeculativeRag.utils.rag import rag_drafting_generator, rag_verifier_generator


async def get_qdrant_client():
    path = Path(__file__).parent.parent / "qdrant_client"
    return AsyncQdrantClient(path=path)

async def speculative_rag(
    query: str,
    embedding_model: str,
    k: int,
    seed: int,
    client: AsyncOpenAI,  # Cliente para embeddings (Ollama)
    drafter_model: str,
    verifier_model: str,
    m: int = 3,
    top_k: int = 10,
    verbose: bool = True,
    debug_prints: bool = False,
    print_documents: bool = False,
    print_embeddings: bool = False,
    print_clusters: bool = False,
    print_drafts: bool = False,
    print_scores: bool = False,
    collection_name: str = "chunks",  # Cambiado a "chunks" por defecto
    ollama_client: AsyncOpenAI = None,  # Cliente para embeddings
    llm_client: AsyncOpenAI = None,  # Cliente para LLM
    drafter_client: AsyncOpenAI = None,  # Cliente específico para el drafter
    verifier_client: AsyncOpenAI = None  # Cliente específico para el verifier
) -> tuple[str, float]:
    """
    Perform Speculative RAG on a query
    
    Args:
        query: User query
        embedding_model: Name of embedding model
        k: Number of clusters
        seed: Random seed for reproducibility
        client: OpenAI client for embeddings (Ollama)
        drafter_model: Model to use for drafting
        verifier_model: Model to use for verification
        m: Number of document subsets to generate for drafting
        top_k: Number of results to retrieve
        verbose: Whether to print verbose output
        debug_prints: Whether to enable debug prints
        print_documents: Whether to print documents
        print_embeddings: Whether to print embeddings
        print_clusters: Whether to print clusters
        print_drafts: Whether to print drafts
        print_scores: Whether to print scores
        collection_name: Nombre de la colección en Qdrant
        ollama_client: Cliente para embeddings (Ollama)
        llm_client: Cliente para LLM (vLLM)
        drafter_client: Cliente específico para el drafter (si es None, usa llm_client)
        verifier_client: Cliente específico para el verifier (si es None, usa llm_client)
        
    Returns:
        Generated RAG response and total execution time
    """
    qdrant_client = None
    
    # Si no se proporciona cliente para LLM, usar el mismo cliente para todo
    if llm_client is None:
        llm_client = client
    
    # Si no se proporciona cliente para embeddings, usar el cliente principal
    if ollama_client is None:
        ollama_client = client
    
    # Si no se proporcionan clientes específicos para drafter y verifier, usar el llm_client
    if drafter_client is None:
        drafter_client = llm_client
    
    if verifier_client is None:
        verifier_client = llm_client
    
    try:
        qdrant_client = await get_qdrant_client()
        
        total_start_time = perf_counter()
        
        if debug_prints:
            print("\n" + "="*80)
            print("🔍 RUNNING SPECULATIVE RAG")
            print(f"Query: {query}")
            print(f"Parameters: k={k}, top_k={top_k}, seed={seed}")
            print(f"Models: drafter={drafter_model}, verifier={verifier_model}")
            print(f"Collection: {collection_name}")
            print("="*80 + "\n")

        # Medir tiempo de embeddings - Usando Ollama para embeddings
        embedding_start_time = perf_counter()
        
        # Crear embedding de la consulta con Ollama
        embedding_response = await ollama_client.embeddings.create(
            input=query, 
            model=embedding_model
        )
        query_embedding = embedding_response.data[0].embedding
        embedding_time = perf_counter() - embedding_start_time
        
        if debug_prints:
            print(f"\n⏱️ Query embedding time: {embedding_time:.2f} seconds")

        # Medir tiempo de búsqueda en Qdrant
        retrieval_start_time = perf_counter()
        search_result = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            search_params=models.SearchParams(
                exact=False,  # Use HNSW index
                hnsw_ef=128  # Control recall/speed trade-off
            )
        )
        retrieval_time = perf_counter() - retrieval_start_time
        
        if debug_prints:
            print(f"⏱️ Document retrieval time: {retrieval_time:.2f} seconds")

        # Print retrieved documents if debugging is enabled
        if debug_prints and print_documents:
            print("\n📑 RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(search_result):
                print(f"\nDocument {i+1} (Score: {doc.score:.4f}):")
                
                if collection_name == "chunks":
                    # Para la colección chunks
                    print(f"Text: {doc.payload.get('text', '')[:200]}...")
                    if 'filename' in doc.payload:
                        print(f"Source file: {doc.payload.get('filename', '')}")
                else:
                    # Para otras colecciones como questions-benchmark
                    if 'ideal_answer' in doc.payload:
                        question_text = doc.payload.get('question', '')
                        answer_text = doc.payload.get('ideal_answer', '')
                        print(f"Question: {question_text}")
                        print(f"Answer: {answer_text[:200]}...")
                    elif 'question' in doc.payload:
                        print(f"Question: {doc.payload.get('question', '')[:200]}...")
                    elif 'text' in doc.payload:
                        print(f"Text: {doc.payload.get('text', '')[:200]}...")
                    else:
                        print(f"Payload: {doc.payload}")
            print("-" * 80)

        # Format documents and embeddings
        documents = []
        for doc in search_result:
            if collection_name == "chunks":
                # Para la colección chunks, simplemente usamos el campo 'text'
                if 'text' in doc.payload:
                    documents.append(doc.payload.get('text', ''))
                else:
                    logger.warning(f"Documento en colección 'chunks' sin campo 'text': {doc.payload}")
            else:
                # Para la colección questions-benchmark y otras
                if 'ideal_answer' in doc.payload:
                    question = doc.payload.get('question', '')
                    answer = doc.payload.get('ideal_answer', '')
                    documents.append(f"Question: {question}\nAnswer: {answer}")
                elif 'exact_answer' in doc.payload:
                    question = doc.payload.get('question', '')
                    exact_answer = doc.payload.get('exact_answer', '')
                    # Manejar distintos formatos de exact_answer
                    if isinstance(exact_answer, list):
                        if exact_answer and isinstance(exact_answer[0], list):
                            answer_text = ', '.join([item[0] for item in exact_answer if item])
                        else:
                            answer_text = ', '.join([str(item) for item in exact_answer if item])
                    else:
                        answer_text = str(exact_answer)
                    documents.append(f"Question: {question}\nAnswer: {answer_text}")
                elif 'text' in doc.payload:
                    documents.append(doc.payload.get('text', ''))

        document_embeddings = []
        
        # Obtener embeddings de los documentos recuperados
        for doc in search_result:
            try:
                # Intentar obtener el vector del documento
                points = await qdrant_client.retrieve(
                    collection_name=collection_name,
                    ids=[doc.id],
                    with_vectors=True
                )
                if points and points[0].vector is not None:
                    document_embeddings.append(points[0].vector)
                else:
                    # Vector vacío con la dimensión correcta para nomic-embed-text
                    document_embeddings.append(np.zeros(768).tolist())
            except Exception as e:
                logger.error(f"Error getting vector for document {doc.id}: {e}")
                # Vector vacío con la dimensión correcta para nomic-embed-text
                document_embeddings.append(np.zeros(768).tolist())

        # Generate InBeddings using CustomInBedder
        try:
            from .utils.inbedder import InBedder
            import torch
            
            # Create instruction from query
            instruction = f"{query}"
            
            # Initialize the InBedder
            inbedder = InBedder()
            
            if debug_prints:
                print("\nUsing InBedder to generate embeddings...")
            
            # Medir tiempo de InBedder
            inbedder_start_time = perf_counter()
            
            # Generate embeddings
            doc_embeddings = await inbedder.encode(documents, instruction, n_mask=3)
            
            if debug_prints and print_embeddings:
                print("\n🔍 EMBEDDINGS GENERATED BY INBEDDER:")
                print(f"Embedding shape: {doc_embeddings.shape}")
                print("\nDebugging embeddings:")
                print(f"Type: {type(doc_embeddings)}")
                print(f"Device: {doc_embeddings.device if torch.is_tensor(doc_embeddings) else 'N/A'}")
                print(f"Requires grad: {doc_embeddings.requires_grad if torch.is_tensor(doc_embeddings) else 'N/A'}")
                
                for i, emb in enumerate(doc_embeddings[:3]):
                    print(f"\nDocument {i+1}:")
                    print(f"Type: {type(emb)}")
                    print(f"First 5 values: {emb[:5].cpu().numpy()}")
                    if torch.is_tensor(emb):
                        norm = torch.norm(emb).item()
                        print(f"Norm: {norm:.6f}")
                print("-" * 80)
            
            # Convert to numpy for clustering
            with torch.no_grad():
                doc_embeddings = doc_embeddings.numpy() if torch.is_tensor(doc_embeddings) else doc_embeddings
            
            if debug_prints and print_embeddings:
                print("\nFinal embeddings for clustering:")
                print(f"Shape: {doc_embeddings.shape}")
                print(f"Type: {type(doc_embeddings)}")
                print(f"First document (first 5 values): {doc_embeddings[0][:5]}")
                print("-" * 80)
            
            inbedder_time = perf_counter() - inbedder_start_time
            
            if debug_prints:
                print(f"⏱️ InBedder processing time: {inbedder_time:.2f} seconds")
            
        except Exception as e:
            if debug_prints:
                print(f"\n⚠️ ERROR USING INBEDDER:")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                print(f"Using original embeddings...\n")
                import traceback
                print("Traceback:")
                print(traceback.format_exc())
            doc_embeddings = np.array(document_embeddings)

        # Perform clustering
        cluster_start_time = perf_counter()
        result = multi_perspective_sampling(doc_embeddings, seed=seed, k=k)
        clusters = result["clusters"]
        
        if debug_prints and print_clusters:
            print("\n🔢 GENERATED CLUSTERS:")
            print(f"Total clusters: {len(clusters)}")
            print(f"Total documents: {len(documents)}")
            for i, cluster in enumerate(clusters):
                print(f"\nCluster {i+1} ({len(cluster)} documents):")
                print("-" * 40)
                for j, doc_idx in enumerate(cluster):
                    print(f"\nDocument {j+1} (Index {doc_idx}):")
                    print(f"Text: {documents[doc_idx]}")
                    print("-" * 40)
        
        # Generate m subsets of documents, each containing one document from each cluster
        np.random.seed(seed)
        document_subsets = []
        for _ in range(m):
            subset = []
            for cluster in clusters:
                # Randomly select one document from each cluster (with replacement)
                doc_idx = np.random.choice(cluster)
                subset.append(documents[doc_idx])
            document_subsets.append(subset)
        
        if debug_prints:
            print("\n📑 GENERATED DOCUMENT SUBSETS:")
            for i, subset in enumerate(document_subsets):
                print(f"\nSubset {i+1}:")
                for j, doc in enumerate(subset):
                    print(f"\nDocument from cluster {j+1}:")
                    print(doc)
                print("-" * 80)

        cluster_time = perf_counter() - cluster_start_time

        if debug_prints:
            print(f"\nTime for clustering and subset generation: {cluster_time:.2f} seconds")
        
        # Medir tiempo de drafting - ejecutando secuencialmente
        drafting_start_time = perf_counter()
        draft_results = []
        draft_times = []
        
        for i, subset in enumerate(document_subsets):
            subset_start_time = perf_counter()
            
            # Generar un único draft
            draft_result = await rag_drafting_generator(
                query=query,
                context=subset,
                client=drafter_client,
                model=drafter_model,
            )
            
            subset_time = perf_counter() - subset_start_time
            draft_times.append(subset_time)
            draft_results.append(draft_result)
            
            if debug_prints:
                print(f"Draft {i+1}/{len(document_subsets)} generado en {subset_time:.2f} segundos")
        
        drafting_time = perf_counter() - drafting_start_time
        avg_draft_time = sum(draft_times) / len(draft_times) if draft_times else 0
        parallel_draft_estimate = max(draft_times) if draft_times else 0

        # Medir tiempo de verificación - ejecutando secuencialmente
        verification_start_time = perf_counter()
        
        # Lista para almacenar tiempos individuales
        verification_times = []
        p_yes_list = []
        
        # Verificar cada draft secuencialmente
        for i, draft in enumerate(draft_results):
            single_verif_start = perf_counter()
            
            # Crear prompt para verificación
            verifier_prompt = """Instruction: {instruction}

Response: {response}

Rationale: {rationale}

Is the rationale good enough to support the answer? 

You must respond with only a single word: "Yes" or "No".
Do not include any explanation or additional text."""
            
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
            
            # Ejecutar verificación
            verification = await verifier_client.chat.completions.create(
                model=verifier_model,
                messages=messages,
                temperature=0.0,
                logprobs=True,
                max_tokens=10
            )
            
            # Obtener resultado y calcular probabilidad
            response = verification.choices[0].message.content.strip().lower()
            p_yes = 0.5  # Valor por defecto
            
            try:
                logprobs = verification.choices[0].logprobs
                if hasattr(logprobs, 'content'):
                    # Obtener todos los logprobs de la respuesta
                    all_logprobs = [token.logprob for token in logprobs.content if hasattr(token, 'logprob')]
                    # Calcular la probabilidad promedio
                    avg_logprob = mean(all_logprobs)
                    if response == "yes":
                        p_yes = np.exp(avg_logprob)
                    else:
                        p_yes = 1 - np.exp(avg_logprob)
            except Exception as e:
                logger.error(f"Error al procesar logprobs: {e}")
            
            p_yes_list.append(p_yes)
            
            # Medir tiempo de esta verificación
            single_verif_time = perf_counter() - single_verif_start
            verification_times.append(single_verif_time)
            
            if debug_prints:
                print(f"Verificación {i+1}/{len(draft_results)} completada en {single_verif_time:.2f} segundos")
        
        verification_time = perf_counter() - verification_start_time
        avg_verification_time = sum(verification_times) / len(verification_times) if verification_times else 0
        parallel_verify_estimate = max(verification_times) if verification_times else 0
        
        # Construir un resultado de verificación similar al original
        verify_response = {
            "p_yes": p_yes_list,
            "sequential_time": verification_time,
            "average_time": avg_verification_time,
            "parallel_estimate": parallel_verify_estimate
        }
        
        # Calculate final scores
        final_scores = []
        for draft_result, p_yes in zip(draft_results, verify_response["p_yes"]):
            score = (
                draft_result["p_draft"] *
                draft_result["p_consistency"] *
                p_yes
            )
            final_scores.append(score)
        
        # Get best draft based on final score
        best_idx = np.argmax(final_scores)
        best_draft = draft_results[best_idx]

        if debug_prints and print_scores:
            print("\n📊 DRAFT SCORES:")
            for i, (score, draft) in enumerate(zip(final_scores, draft_results)):
                print(f"\nDraft {i+1}:")
                print(f"Draft probability (rationale + response): {draft['p_draft']:.4f}")
                print(f"Self-consistency (rationale * response): {draft['p_consistency']:.4f}")
                print(f"Self-reflection (verifier p_yes): {verify_response['p_yes'][i]:.4f}")
                print(f"Final score: {score:.4f}")
            print("-" * 80)
        
        total_execution_time = perf_counter() - total_start_time
        
        if debug_prints:
            print("\n⏱️ TIMING SUMMARY:")
            print(f"Query embedding time: {embedding_time:.2f} seconds")
            print(f"Document retrieval time: {retrieval_time:.2f} seconds")
            print(f"InBedder processing time: {inbedder_time:.2f} seconds")
            print(f"Drafting time (secuencial): {drafting_time:.2f} seconds")
            print(f"Drafting tiempo promedio: {avg_draft_time:.2f} seconds")
            print(f"Drafting estimado en paralelo: {parallel_draft_estimate:.2f} seconds")
            print(f"Verification time (secuencial): {verification_time:.2f} seconds")
            print(f"Verification tiempo promedio: {avg_verification_time:.2f} seconds")
            print(f"Verification estimado en paralelo: {parallel_verify_estimate:.2f} seconds")
            print(f"Total execution time: {total_execution_time:.2f} seconds")
            print(f"Cluster and Subset generation time: {cluster_time:.2f} seconds")
            print("-" * 80)

            print("\n🏆 WINNING DRAFT DETAILS:")
            print("="*80)
            print("📑 EVIDENCE USED:")
            for i, doc in enumerate(document_subsets[best_idx], 1):
                print(f"\nDocument {i}:")
                print(f"{doc[:200]}...")
            print("\n📝 RATIONALE:")
            print(best_draft["rationale"])
            print("\n✨ FINAL RESPONSE:")
            print(best_draft["response"])
            print("\n📊 SCORES:")
            print(f"Draft probability: {best_draft['p_draft']:.4f}")
            print(f"Consistency: {best_draft['p_consistency']:.4f}")
            print(f"Verifier probability: {verify_response['p_yes'][best_idx]:.4f}")
            print(f"Final score: {final_scores[best_idx]:.4f}")
            print("="*80)

        return best_draft["response"], total_execution_time
    finally:
        if qdrant_client:
            await qdrant_client.close()
