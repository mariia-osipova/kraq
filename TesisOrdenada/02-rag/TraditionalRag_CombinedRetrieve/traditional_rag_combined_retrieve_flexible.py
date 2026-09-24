from time import perf_counter
import asyncio
from loguru import logger
from openai import AsyncOpenAI
from typing import List, Dict, Any, Set
from pathlib import Path
from qdrant_client import AsyncQdrantClient, models
import math

# Configuración para modelos locales
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"

async def get_qdrant_client():
    # Usar ruta absoluta al directorio raíz del proyecto
    path = Path(__file__).parent.parent / "qdrant_client"
    return AsyncQdrantClient(path=path)

async def traditional_rag_combined_retrieve_flexible(
    query: str,
    embeddings_client: AsyncOpenAI = None,
    llm_client: AsyncOpenAI = None,
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    top_k: int = 15,
    alpha: float = 0.5,  # Porcentaje para la pregunta original (0.0 a 1.0)
    num_similar_questions: int = 2,  # Número de preguntas similares a recuperar
    debug_prints: bool = True,
    verbose_timing: bool = False,
    collection_name: str = "chunks",  # Colección para recuperar documentos
    question_collection: str = "questions-index-finetuned"  # Colección para preguntas similares
) -> tuple[str, float]:
    """
    Implementación de RAG basado en recuperación combinada con parámetros flexibles
    
    Args:
        query: Pregunta del usuario
        embeddings_client: Cliente para embeddings (Ollama)
        llm_client: Cliente para LLM (vLLM)
        embedding_model: Modelo para generar embeddings
        llm_model: Modelo LLM para generar respuestas
        top_k: Número total de documentos a recuperar
        alpha: Porcentaje de documentos para la pregunta original (0.0 a 1.0)
        num_similar_questions: Número de preguntas similares a recuperar (1 o más)
        debug_prints: Activar impresión de información de debug
        verbose_timing: Activar impresión detallada de tiempos de cada etapa
        collection_name: Nombre de la colección para recuperar documentos
        question_collection: Nombre de la colección para preguntas similares
        
    Returns:
        Tupla con respuesta generada por el modelo y tiempo total de ejecución
    """
    # Validar parámetros
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha debe estar entre 0.0 y 1.0, recibido: {alpha}")
    if num_similar_questions < 1:
        raise ValueError(f"num_similar_questions debe ser al menos 1, recibido: {num_similar_questions}")
    if top_k < 1:
        raise ValueError(f"top_k debe ser al menos 1, recibido: {top_k}")
    
    # Si no se proporciona cliente de embeddings, crear uno
    if embeddings_client is None:
        embeddings_client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"
        )
    
    # Si no se proporciona cliente LLM, crear uno
    if llm_client is None:
        llm_client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="not-needed"
        )
    
    qdrant_client = None
    try:
        qdrant_client = await get_qdrant_client()
        
        total_start_time = perf_counter()
        
        if debug_prints:
            print("\n" + "="*80)
            print("🔍 RUNNING TRADITIONAL RAG WITH FLEXIBLE COMBINED RETRIEVE")
            print(f"Original Query: {query}")
            print(f"Parameters: top_k={top_k}, alpha={alpha:.2f}, num_similar_questions={num_similar_questions}")
            print(f"Models: embedding={embedding_model}, llm={llm_model}")
            print(f"Collections: questions={question_collection}, documents={collection_name}")
            print("="*80 + "\n")

        # Etapa 1: Generar embedding de la consulta
        embedding_start_time = perf_counter()
        embedding_response = await embeddings_client.embeddings.create(
            input=query,
            model=embedding_model
        )
        query_embedding = embedding_response.data[0].embedding
        embedding_time = perf_counter() - embedding_start_time
        
        if debug_prints or verbose_timing:
            print(f"⏱️ Query embedding time: {embedding_time:.4f} seconds")
            
        # Etapa 2: Buscar preguntas similares en la colección de preguntas
        similar_question_start_time = perf_counter()
        similar_question_result = await qdrant_client.search(
            collection_name=question_collection,
            query_vector=query_embedding,
            limit=num_similar_questions,  # Recuperar el número especificado de preguntas similares
            search_params=models.SearchParams(
                exact=False,
                hnsw_ef=128
            )
        )
        
        # Lista para almacenar preguntas similares y sus embeddings
        similar_questions = []
        similar_question_embeddings = []
        
        if not similar_question_result or len(similar_question_result) == 0:
            logger.warning(f"No se encontraron preguntas similares en {question_collection}")
            # Si no hay preguntas similares, usar solo la consulta original
            similar_questions.append(query)
            similar_question_embeddings.append(query_embedding)
            actual_num_similar = 1
        else:
            # Procesar las preguntas similares encontradas
            for i, similar_point in enumerate(similar_question_result):
                question = similar_point.payload.get('question', query)
                score = similar_point.score
                similar_questions.append(question)
                
                # Usar el embedding almacenado si está disponible
                if hasattr(similar_point, 'vector') and similar_point.vector:
                    similar_question_embeddings.append(similar_point.vector)
                else:
                    # Si no hay embedding almacenado, generamos uno nuevo
                    embedding_response = await embeddings_client.embeddings.create(
                        input=question,
                        model=embedding_model
                    )
                    similar_question_embeddings.append(embedding_response.data[0].embedding)
                
                if debug_prints:
                    print(f"📝 Similar question {i+1}: {question}")
                    print(f"📊 Similarity score {i+1}: {score:.4f}")
            
            actual_num_similar = len(similar_question_result)
            
            # Si se encontraron menos preguntas similares de las solicitadas, rellenar con duplicados
            while len(similar_questions) < num_similar_questions:
                # Duplicar la última pregunta encontrada
                similar_questions.append(similar_questions[-1])
                similar_question_embeddings.append(similar_question_embeddings[-1])
                
        similar_question_time = perf_counter() - similar_question_start_time
        
        if debug_prints:
            print(f"⏱️ Similar question search time: {similar_question_time:.4f} seconds")
            print(f"📊 Found {actual_num_similar} unique similar questions, using {len(similar_questions)} total")

        # Etapa 3: Calcular la distribución de documentos a recuperar
        retrieval_start_time = perf_counter()
        
        # Calcular documentos para la consulta original
        original_query_docs = int(math.ceil(top_k * alpha))
        
        # Calcular documentos restantes para distribuir entre preguntas similares
        remaining_docs = top_k - original_query_docs
        
        # Distribuir equitativamente entre las preguntas similares
        if len(similar_questions) > 0:
            # Distribución base: cada pregunta similar recibe al menos esta cantidad
            docs_per_similar = remaining_docs // len(similar_questions)
            # Documentos extra que hay que distribuir uno por uno
            extra_docs = remaining_docs % len(similar_questions)
            
            # Crear lista de documentos por pregunta similar
            similar_docs_distribution = []
            for i in range(len(similar_questions)):
                docs_for_this_question = docs_per_similar
                # Distribuir documentos extra: uno a cada pregunta hasta que se acaben
                if i < extra_docs:
                    docs_for_this_question += 1
                similar_docs_distribution.append(docs_for_this_question)
        else:
            # Si no hay preguntas similares, todos los documentos van a la original
            similar_docs_distribution = []
            original_query_docs = top_k
        
        if debug_prints:
            print(f"📊 Document distribution:")
            print(f"   - Original query: {original_query_docs} docs ({original_query_docs/top_k*100:.1f}%)")
            for i, docs in enumerate(similar_docs_distribution):
                percentage = (docs / top_k * 100) if top_k > 0 else 0
                print(f"   - Similar question {i+1}: {docs} docs ({percentage:.1f}%)")
            total_distributed = original_query_docs + sum(similar_docs_distribution)
            print(f"   - Total: {total_distributed}/{top_k} docs")
            
            # Verificar que la distribución es correcta
            if total_distributed != top_k:
                logger.warning(f"Distribución incorrecta: {total_distributed} != {top_k}")
            
            # Verificar que la diferencia máxima entre preguntas similares es 1
            if similar_docs_distribution:
                min_docs = min(similar_docs_distribution)
                max_docs = max(similar_docs_distribution)
                print(f"   - Diferencia máxima entre similares: {max_docs - min_docs} (debe ser ≤ 1)")
        
        # Recuperación combinada usando todos los embeddings
        results = []
        retrieved_doc_ids = set()  # Conjunto para rastrear IDs de documentos ya recuperados
        
        # 1. Recuperar documentos con la consulta original
        if original_query_docs > 0:
            search_limit = int(original_query_docs * 2)  # Margen para duplicados
            
            original_results = await qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=search_limit,
                search_params=models.SearchParams(
                    exact=False,
                    hnsw_ef=128
                )
            )
            
            # Filtrar para obtener exactamente original_query_docs únicos
            docs_added = 0
            for doc in original_results:
                if docs_added >= original_query_docs:
                    break
                
                doc_id = doc.id
                if doc_id not in retrieved_doc_ids:
                    results.append(doc)
                    retrieved_doc_ids.add(doc_id)
                    docs_added += 1
            
            if debug_prints:
                print(f"📄 Retrieved {docs_added} unique documents using original query")
        
        # 2. Recuperar documentos con cada pregunta similar
        for i, (similar_embedding, docs_needed) in enumerate(zip(similar_question_embeddings, similar_docs_distribution)):
            if docs_needed <= 0:
                continue
                
            # Intentar recuperar suficientes documentos únicos
            offset = 0
            limit_factor = 2  # Factor inicial para el límite
            docs_added = 0
            
            # Limitar el número de intentos para evitar bucles infinitos
            max_attempts = 3
            attempts = 0
            
            while docs_added < docs_needed and attempts < max_attempts:
                attempts += 1
                # Solicitar más documentos para tener margen en caso de duplicados
                limit_value = int(docs_needed * limit_factor)
                if limit_value <= 0:
                    limit_value = 1  # Asegurar un mínimo de 1 documento
                
                try:
                    similar_results = await qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=similar_embedding,
                        limit=limit_value,
                        offset=int(offset),
                        search_params=models.SearchParams(
                            exact=False,
                            hnsw_ef=128
                        )
                    )
                    
                    # Si no hay resultados, salir del bucle
                    if not similar_results:
                        break
                    
                    # Filtrar duplicados
                    new_docs_added = 0
                    for doc in similar_results:
                        if docs_added >= docs_needed:
                            break
                        
                        doc_id = doc.id
                        if doc_id not in retrieved_doc_ids:
                            results.append(doc)
                            retrieved_doc_ids.add(doc_id)
                            docs_added += 1
                            new_docs_added += 1
                    
                    # Si no se añadieron nuevos documentos, incrementar offset y limit_factor
                    if new_docs_added == 0:
                        offset += limit_value
                        limit_factor += 1
                    else:
                        break
                        
                except Exception as e:
                    logger.error(f"Error en búsqueda de similar_question_{i+1}: {e}")
                    break
            
            if debug_prints:
                print(f"📄 Retrieved {docs_added}/{docs_needed} unique documents using similar question {i+1}")
        
        retrieval_time = perf_counter() - retrieval_start_time
        
        # Siempre imprimimos el tiempo de recuperación para benchmarking
        print(f"retrieval_time:{retrieval_time}")
        
        if debug_prints or verbose_timing:
            print(f"⏱️ Document retrieval time: {retrieval_time:.4f} seconds")
            print(f"📄 Retrieved {len(results)} unique documents in total")

        # Etapa 4: Preparar contexto
        context_start_time = perf_counter()
        
        documents = []
        for point in results:
            if point.payload:
                # Seleccionar el texto según la colección
                if collection_name == "chunks":
                    # Para la colección chunks, solo usar el campo text
                    if 'text' in point.payload:
                        documents.append(point.payload['text'])
                    else:
                        logger.warning(f"Documento en colección 'chunks' sin campo 'text'")
                else:
                    # Para otras colecciones (como questions-benchmark)
                    if 'text' in point.payload:
                        documents.append(point.payload['text'])
                    elif 'question' in point.payload:
                        question_text = point.payload.get('question', '')
                        if 'ideal_answer' in point.payload:
                            answer_text = point.payload.get('ideal_answer', '')
                            documents.append(f"Question: {question_text}\nAnswer: {answer_text}")
                        elif 'exact_answer' in point.payload:
                            # Manejar diferentes formatos de exact_answer
                            exact_answer = point.payload.get('exact_answer', '')
                            if isinstance(exact_answer, list):
                                if len(exact_answer) > 0:
                                    if isinstance(exact_answer[0], list):
                                        # Si es una lista de listas
                                        answer_text = ", ".join([item[0] for item in exact_answer if item])
                                    else:
                                        # Si es una lista simple
                                        answer_text = ", ".join([str(item) for item in exact_answer if item])
                                else:
                                    answer_text = "No answer available"
                            else:
                                answer_text = str(exact_answer)
                            documents.append(f"Question: {question_text}\nAnswer: {answer_text}")
        
        context = "\n\n".join(documents)
        context_time = perf_counter() - context_start_time
        
        if verbose_timing:
            print(f"⏱️ Context preparation time: {context_time:.4f} seconds")
            print(f"📏 Context length: {len(context)} characters")

        # Etapa 5: Generar respuesta con LLM
        prompt_start_time = perf_counter()
        
        prompt = f"""Below is an instruction that describes a task. Write a response for it and state
 your explanation supporting your response.
 ### Evidence:
 {context}
 ### Instruction: 
 {query}
 ### Response:
"""
        prompt_time = perf_counter() - prompt_start_time
        
        if verbose_timing:
            print(f"⏱️ Prompt construction time: {prompt_time:.4f} seconds")
            print(f"📏 Prompt length: {len(prompt)} characters")

        # Etapa 6: Llamada al LLM
        llm_start_time = perf_counter()
        
        response = await llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        answer = response.choices[0].message.content
        llm_time = perf_counter() - llm_start_time
        
        if debug_prints or verbose_timing:
            print(f"⏱️ LLM response generation time: {llm_time:.4f} seconds")
            print(f"📏 Response length: {len(answer)} characters")

        # Calcular tiempo total
        total_time = perf_counter() - total_start_time
        
        if debug_prints or verbose_timing:
            print(f"⏱️ TOTAL PROCESSING TIME: {total_time:.4f} seconds")
            
            if verbose_timing:
                # Resumen de tiempos
                print("\n" + "="*50)
                print("⏰ TIMING BREAKDOWN:")
                print(f"- Embedding:       {embedding_time:.4f}s ({embedding_time/total_time*100:.1f}%)")
                print(f"- Similar question:{similar_question_time:.4f}s ({similar_question_time/total_time*100:.1f}%)")
                print(f"- Retrieval:       {retrieval_time:.4f}s ({retrieval_time/total_time*100:.1f}%)")
                print(f"- Context prep:    {context_time:.4f}s ({context_time/total_time*100:.1f}%)")
                print(f"- Prompt prep:     {prompt_time:.4f}s ({prompt_time/total_time*100:.1f}%)")
                print(f"- LLM response:    {llm_time:.4f}s ({llm_time/total_time*100:.1f}%)")
                print(f"- Total:           {total_time:.4f}s (100%)")
                print("="*50 + "\n")
        
        # Mostrar respuesta si debug_prints está activado
        if debug_prints:
            print("\n🤖 FINAL ANSWER:")
            print("-"*50)
            print(answer)
            print("-"*50 + "\n")
        
        return answer, total_time
    except Exception as e:
        logger.error(f"Error en traditional_rag_combined_retrieve_flexible: {e}")
        raise  # Re-lanzar la excepción para que sea capturada por el benchmark
    finally:
        if qdrant_client:
            await qdrant_client.close()


# Función de compatibilidad que mantiene la interfaz original
async def traditional_rag_combined_retrieve(
    query: str,
    embeddings_client: AsyncOpenAI = None,
    llm_client: AsyncOpenAI = None,
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    top_k: int = 15,
    debug_prints: bool = True,
    verbose_timing: bool = False,
    collection_name: str = "chunks",
    question_collection: str = "questions-index-finetuned"
) -> tuple[str, float]:
    """
    Función de compatibilidad que mantiene la interfaz original
    Usa alpha=0.5 y num_similar_questions=2 por defecto
    """
    return await traditional_rag_combined_retrieve_flexible(
        query=query,
        embeddings_client=embeddings_client,
        llm_client=llm_client,
        embedding_model=embedding_model,
        llm_model=llm_model,
        top_k=top_k,
        alpha=0.5,  # 50% para la pregunta original (valor original)
        num_similar_questions=2,  # 2 preguntas similares (valor original)
        debug_prints=debug_prints,
        verbose_timing=verbose_timing,
        collection_name=collection_name,
        question_collection=question_collection
    ) 