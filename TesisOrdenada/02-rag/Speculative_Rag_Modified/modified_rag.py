from time import perf_counter
import numpy as np
import asyncio
import os
import sys
from loguru import logger
from openai import AsyncOpenAI
from typing import List, Dict, Any
from pathlib import Path
from qdrant_client import AsyncQdrantClient, models

# Ajustar sys.path para incluir el directorio del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importaciones absolutas
from Speculative_Rag_Modified.utils.clustering import multi_perspective_sampling
from Speculative_Rag_Modified.utils.rag import rag_drafting_generator, rag_verifier_generator


async def get_qdrant_client():
    path = Path(__file__).parent.parent / "qdrant_client"
    return AsyncQdrantClient(path=path)

async def modified_rag(
    query: str,
    embedding_model: str,
    k: int,
    seed: int,
    client: AsyncOpenAI,  # Este ahora es el cliente genérico
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
    collection_name: str = "chunks",  # Colección de documentos 
    questions_collection: str = "questions-index-finetuned",  # Colección de preguntas
    ollama_client: AsyncOpenAI = None,  # Cliente específico para embeddings
    llm_client: AsyncOpenAI = None,  # Cliente para LLM
    drafter_client: AsyncOpenAI = None,  # Cliente específico para el drafter
    verifier_client: AsyncOpenAI = None  # Cliente específico para el verifier
) -> tuple[str, float]:
    """
    Modified RAG that uses similar questions for document retrieval
    
    Args:
        query: User query
        embedding_model: Name of embedding model
        k: Number of clusters
        seed: Random seed for reproducibility
        client: OpenAI client (generic, used if specific clients not provided)
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
        collection_name: Collection for document retrieval
        questions_collection: Collection for similar questions
        ollama_client: Client specifically for embeddings (Ollama)
        llm_client: Client for LLM operations (vLLM)
        drafter_client: Client specifically for drafter (if None, uses llm_client)
        verifier_client: Client specifically for verifier (if None, uses llm_client)
    
    Returns:
        Generated response and total execution time
    """
    qdrant_client = None
    
    # Configurar los clientes según los parámetros o crear nuevos
    if ollama_client is None:
        # Crear cliente de Ollama para embeddings
        ollama_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
    
    if llm_client is None:
        # Crear cliente de vLLM para LLMs
        llm_client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="not-needed"  # vLLM no requiere API key
        )
    
    # Configurar clientes específicos para drafter y verifier
    if drafter_client is None:
        drafter_client = llm_client  # Por defecto, usar el cliente LLM
    
    if verifier_client is None:
        verifier_client = llm_client  # Por defecto, usar el cliente LLM
    
    try:
        qdrant_client = await get_qdrant_client()
        
        total_start_time = perf_counter()
        
        if debug_prints:
            print("\n" + "="*80)
            print("🔍 RUNNING MODIFIED RAG")
            print(f"Query: {query}")
            print(f"Parameters: k={k}, top_k={top_k}, seed={seed}, m={m}")
            print(f"Models: drafter={drafter_model}, verifier={verifier_model}")
            print(f"Collections: documents={collection_name}, questions={questions_collection}")
            print("="*80 + "\n")

        # 1. Buscar pregunta similar en la colección de preguntas
        
        # Generar embedding de la pregunta actual - Usando Ollama
        embedding_start_time = perf_counter()
        query_embedding_response = await ollama_client.embeddings.create(
            input=query,
            model=embedding_model
        )
        query_embedding = query_embedding_response.data[0].embedding
        embedding_time = perf_counter() - embedding_start_time
        
        if debug_prints:
            print(f"⏱️ Query embedding time: {embedding_time:.2f} seconds")
            if print_embeddings:
                print(f"Query embedding dimension: {len(query_embedding)}")
                print(f"First 5 values: {query_embedding[:5]}")
        
        # Buscar pregunta similar - Ahora medimos solo el tiempo de búsqueda en Qdrant
        if debug_prints:
            print(f"\n🔍 SEARCHING FOR SIMILAR QUESTION IN '{questions_collection}'...")
        
        similar_question_start = perf_counter()
        similar_questions = await qdrant_client.search(
            collection_name=questions_collection,
            query_vector=query_embedding,
            limit=1,  # Solo necesitamos la más similar
            search_params=models.SearchParams(
                exact=False,
                hnsw_ef=128
            )
        )
        similar_question_time = perf_counter() - similar_question_start

        similar_question = similar_questions[0].payload.get("question", "")
        #logger.info(f"Similar question: {similar_question}")
        similar_question_score = similar_questions[0].score

        if debug_prints:
            print("\n🔍 SIMILAR QUESTION FOUND:")
            print(f"Original query: {query}")
            print(f"Similar question: {similar_question}")
            print(f"Similarity score: {similar_question_score:.4f}")
            print(f"Source file: {similar_questions[0].payload.get('source_file', 'Unknown')}")
            print(f"Question type: {similar_questions[0].payload.get('type', 'Unknown')}")
            print(f"Community ID: {similar_questions[0].payload.get('community_id', 'Unknown')}")
            if 'explanation' in similar_questions[0].payload:
                print(f"\nQuestion explanation: {similar_questions[0].payload['explanation'][:200]}...")
            print(f"Similar question search time: {similar_question_time:.2f} seconds")
            print("-" * 80)

        # 2. Usar la pregunta original para recuperar documentos de la colección "chunks"
        retrieval_start_time = perf_counter()
        if debug_prints:
            print(f"\n📚 RETRIEVING DOCUMENTS FROM '{collection_name}'...")
        
        retrieved_docs = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,  # Usar el embedding de la pregunta original
            limit=top_k,
            search_params=models.SearchParams(
                exact=False,
                hnsw_ef=128
            )
        )
        retrieval_time = perf_counter() - retrieval_start_time

        if debug_prints:
            print(f"⏱️ Document retrieval time: {retrieval_time:.2f} seconds")
            print(f"Retrieved {len(retrieved_docs)} documents")

        # Format documents y extraer texto
        documents = []
        for doc in retrieved_docs:
            if 'text' in doc.payload:
                documents.append(doc.payload.get('text', ''))
            else:
                logger.warning(f"Documento sin campo 'text': {doc.payload}")
        
        document_embeddings = []
        
        # Obtener embeddings de los documentos recuperados
        for doc in retrieved_docs:
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
                    # Vector vacío con dimensión correcta
                    document_embeddings.append(np.zeros(768).tolist())
            except Exception as e:
                logger.error(f"Error getting vector for document {doc.id}: {e}")
                document_embeddings.append(np.zeros(768).tolist())
        
        # Print retrieved documents if debugging is enabled
        if debug_prints and print_documents:
            print("\n📑 RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(retrieved_docs):
                print(f"\nDocument {i+1} (Score: {doc.score:.4f}):")
                print(f"Text: {doc.payload.get('text', '')[:200]}...")
            print("-" * 80)

        # 3. Generate InBeddings usando la pregunta similar
        if debug_prints:
            print(f"\n🧠 GENERATING INBEDDINGS USING SIMILAR QUESTION...")
        
        try:
            from Speculative_Rag_Modified.utils.inbedder import InBedder
            import torch
            import multiprocessing
            
            instruction = f"{similar_question}"
            # Configurar batch_size aquí, al crear el InBedder
            n_cores = multiprocessing.cpu_count()
            batch_size = min(32, max(1, len(documents) // n_cores))
            inbedder = InBedder(batch_size=batch_size)  # Lo pasamos al constructor
            
            inbedder_start_time = perf_counter()
            doc_embeddings, batch_time = await inbedder.encode(documents, instruction, n_mask=3)
            
            with torch.no_grad():
                doc_embeddings = doc_embeddings.numpy() if torch.is_tensor(doc_embeddings) else doc_embeddings
            
            # Usar el tiempo de procesamiento del batch en lugar del tiempo total
            inbedder_time = batch_time
            
            if debug_prints:
                print(f"✓ InBeddings generated successfully")
                print(f"⏱️ InBedder processing time: {inbedder_time:.2f} seconds")
                print(f"🔢 Batch size used: {batch_size}")
                if print_embeddings:
                    print(f"InBedding shape: {doc_embeddings.shape}")
                    
        except Exception as e:
            if debug_prints:
                print(f"\n⚠️ Error using InBedder: {str(e)}")
                print("Using original embeddings...")
            doc_embeddings = np.array(document_embeddings)
            inbedder_time = 0.0

        # 4. Clustering y sampling
        if debug_prints:
            print(f"\n🔍 PERFORMING MULTI-PERSPECTIVE SAMPLING (k={k})...")
        
        clustering_start_time = perf_counter()
        result = multi_perspective_sampling(doc_embeddings, seed=seed, k=k)
        clusters = result["clusters"]
        clustering_time = perf_counter() - clustering_start_time
        
        if debug_prints:
            print(f"⏱️ Clustering time: {clustering_time:.2f} seconds")
            print(f"Found {len(clusters)} clusters")
            
            if print_clusters:
                print("\n📊 CLUSTERS:")
                for i, cluster in enumerate(clusters):
                    print(f"Cluster {i+1}: {cluster}")
                print("-" * 80)
        
        # Generate document subsets
        if debug_prints:
            print(f"\n🧩 GENERATING {m} DOCUMENT SUBSETS...")
        
        document_subsets = []
        np.random.seed(seed)
        for i in range(m):
            subset = []
            for j, cluster in enumerate(clusters):
                doc_idx = np.random.choice(cluster)
                subset.append(documents[doc_idx])
            document_subsets.append(subset)
            
            if debug_prints and print_documents:
                print(f"\nSubset {i+1}:")
                for j, doc in enumerate(subset):
                    print(f"\nDocument from cluster {j+1}:")
                    print(f"{doc[:200]}...")
                print("-" * 80)

        # Generate drafts secuencialmente usando la pregunta original y el cliente específico para drafter
        if debug_prints:
            print(f"\n✏️ GENERATING {m} DRAFTS SEQUENTIALLY...")
        
        drafting_start_time = perf_counter()
        draft_results = []
        draft_times = []
        
        # Ejecutar generación de drafts secuencialmente
        for i, subset in enumerate(document_subsets):
            subset_start_time = perf_counter()
            
            # Generar un único draft
            draft_result = await rag_drafting_generator(
                query=query,  # Usar la pregunta original
                context=subset,
                client=drafter_client,  # Usar cliente específico para drafter
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
        
        if debug_prints and print_drafts:
            print("\n📝 GENERATED DRAFTS:")
            for i, draft in enumerate(draft_results):
                print(f"\nDraft {i+1}:")
                print(f"Rationale: {draft['rationale'][:200]}...")
                print(f"Response: {draft['response'][:200]}...")
                print(f"Draft probability: {draft['p_draft']:.4f}")
                print(f"Consistency: {draft['p_consistency']:.4f}")
                print("-" * 80)

        # Verification secuencial con cliente específico para verifier
        if debug_prints:
            print(f"\n✅ VERIFYING DRAFTS SEQUENTIALLY...")
        
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
                    from statistics import mean
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
        
        if debug_prints and print_scores:
            print("\n📊 VERIFICATION SCORES:")
            for i, p_yes in enumerate(verify_response["p_yes"]):
                print(f"Draft {i+1}: {p_yes:.4f}")
            print("-" * 80)

        # Calculate final scores
        final_scores = []
        for draft_result, p_yes in zip(draft_results, verify_response["p_yes"]):
            score = (
                draft_result["p_draft"] *
                draft_result["p_consistency"] *
                p_yes
            )
            final_scores.append(score)
        
        best_idx = np.argmax(final_scores)
        best_draft = draft_results[best_idx]
        
        if debug_prints and print_scores:
            print("\n🏆 FINAL SCORES:")
            for i, score in enumerate(final_scores):
                star = "⭐" if i == best_idx else " "
                print(f"{star} Draft {i+1}: {score:.4f}")
            print("-" * 80)

        total_execution_time = perf_counter() - total_start_time
        
        if debug_prints:
            print("\n⏱️ TIMING SUMMARY:")
            print(f"Query embedding time: {embedding_time:.2f} seconds")
            print(f"Similar question search time: {similar_question_time:.2f} seconds")
            print(f"Document retrieval time: {retrieval_time:.2f} seconds")
            print(f"InBedder processing time: {inbedder_time:.2f} seconds")
            print(f"Clustering time: {clustering_time:.2f} seconds")
            print(f"Drafting time (secuencial): {drafting_time:.2f} seconds")
            print(f"Drafting tiempo promedio: {avg_draft_time:.2f} seconds")
            print(f"Drafting estimado en paralelo: {parallel_draft_estimate:.2f} seconds")
            print(f"Verification time (secuencial): {verification_time:.2f} seconds")
            print(f"Verification tiempo promedio: {avg_verification_time:.2f} seconds")
            print(f"Verification estimado en paralelo: {parallel_verify_estimate:.2f} seconds")
            
            # Calcular tiempos secuenciales y paralelos totales
            sequential_llm_time = drafting_time + verification_time
            parallel_llm_time = parallel_draft_estimate + parallel_verify_estimate
            time_saved = sequential_llm_time - parallel_llm_time
            
            print(f"\nTiempo total LLM (secuencial): {sequential_llm_time:.2f} seconds")
            print(f"Tiempo total LLM (estimado paralelo): {parallel_llm_time:.2f} seconds")
            print(f"Tiempo estimado ahorrado: {time_saved:.2f} seconds")
            print(f"Speedup estimado: {sequential_llm_time/parallel_llm_time:.2f}x")
            
            print(f"Total execution time: {total_execution_time:.2f} seconds")
            print("-" * 80)

            print("\n🏆 WINNING DRAFT DETAILS:")
            print("="*80)
            print("📑 EVIDENCE USED:")
            for i, doc in enumerate(document_subsets[best_idx], 1):
                print(f"\nDocument {i}:")
                print(f"{doc[:200]}...")
            print("\n❓ QUESTIONS:")
            print(f"Original: {query}")
            print(f"Similar: {similar_question} (score: {similar_question_score:.4f})")
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