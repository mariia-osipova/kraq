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

async def traditional_rag_combined_retrieve(
    query: str,
    embeddings_client: AsyncOpenAI = None,
    llm_client: AsyncOpenAI = None,
    embedding_model: str = "nomic-embed-text",
    llm_model: str = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    top_k: int = 5,
    debug_prints: bool = True,
    verbose_timing: bool = False,
    collection_name: str = "chunks",  # Colección para recuperar documentos
    question_collection: str = "questions-index-finetuned"  # Colección para preguntas similares
) -> tuple[str, float]:
    """
    Implementación de RAG basado en recuperación combinada
    
    Args:
        query: Pregunta del usuario
        embeddings_client: Cliente para embeddings (Ollama)
        llm_client: Cliente para LLM (vLLM)
        embedding_model: Modelo para generar embeddings
        llm_model: Modelo LLM para generar respuestas
        top_k: Número de documentos a recuperar
        debug_prints: Activar impresión de información de debug
        verbose_timing: Activar impresión detallada de tiempos de cada etapa
        collection_name: Nombre de la colección para recuperar documentos
        question_collection: Nombre de la colección para preguntas similares
        
    Returns:
        Tupla con respuesta generada por el modelo y tiempo total de ejecución
    """
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
            print("🔍 RUNNING TRADITIONAL RAG WITH COMBINED RETRIEVE")
            print(f"Original Query: {query}")
            print(f"Parameters: top_k={top_k}")
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
            
        # Etapa 2: Buscar las dos preguntas más similares en la colección de preguntas
        similar_question_start_time = perf_counter()
        similar_question_result = await qdrant_client.search(
            collection_name=question_collection,
            query_vector=query_embedding,
            limit=2,  # Recuperar las 2 preguntas más similares
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
            
            # Si solo se encontró una pregunta similar, duplicarla
            if len(similar_question_result) == 1:
                similar_questions.append(similar_questions[0])
                similar_question_embeddings.append(similar_question_embeddings[0])
                
        similar_question_time = perf_counter() - similar_question_start_time
        
        if debug_prints:
            print(f"⏱️ Similar question search time: {similar_question_time:.4f} seconds")

        # Etapa 3: Calcular la distribución de documentos a recuperar
        retrieval_start_time = perf_counter()
        
        # Calculamos cuántos documentos recuperar con cada embedding - Asegurando enteros
        original_query_docs = int(math.ceil(top_k * 0.5))  # 50% con la consulta original
        similar_question_1_docs = int(math.ceil(top_k * 0.25))  # 25% con la primera pregunta similar
        similar_question_2_docs = int(top_k - original_query_docs - similar_question_1_docs)  # 25% restante con la segunda pregunta similar
        
        if debug_prints:
            print(f"📊 Document distribution: Original={original_query_docs}, Similar1={similar_question_1_docs}, Similar2={similar_question_2_docs}")
        
        # Recuperación combinada usando los tres embeddings
        results = []
        retrieved_doc_ids = set()  # Conjunto para rastrear IDs de documentos ya recuperados
        
        # 1. Recuperar documentos con la consulta original
        if original_query_docs > 0:
            # Solicitar más documentos para tener margen en caso de duplicados - Asegurando enteros
            search_limit = int(original_query_docs * 2)  # Convertir explícitamente a entero
            
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
        
        # 2. Recuperar documentos con la primera pregunta similar
        if similar_question_1_docs > 0 and len(similar_question_embeddings) >= 1:
            # Intentar recuperar suficientes documentos únicos
            docs_to_find = similar_question_1_docs
            offset = 0
            limit_factor = 2  # Factor inicial para el límite (entero)
            docs_added = 0
            
            # Limitar el número de intentos para evitar bucles infinitos
            max_attempts = 3
            attempts = 0
            
            while docs_added < docs_to_find and attempts < max_attempts:
                attempts += 1
                # Solicitar más documentos para tener margen en caso de duplicados
                # Asegurarse que limit sea un entero
                limit_value = int(docs_to_find * limit_factor)
                if limit_value <= 0:
                    limit_value = 1  # Asegurar un mínimo de 1 documento
                
                try:
                    similar_1_results = await qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=similar_question_embeddings[0],
                        limit=limit_value,
                        offset=int(offset),  # Conversión explícita a entero
                        search_params=models.SearchParams(
                            exact=False,
                            hnsw_ef=128
                        )
                    )
                    
                    # Si no hay resultados, salir del bucle
                    if not similar_1_results:
                        break
                    
                    # Filtrar duplicados
                    new_docs_added = 0
                    for doc in similar_1_results:
                        if docs_added >= docs_to_find:
                            break
                        
                        doc_id = doc.id
                        if doc_id not in retrieved_doc_ids:
                            results.append(doc)
                            retrieved_doc_ids.add(doc_id)
                            docs_added += 1
                            new_docs_added += 1
                    
                    # Si no se añadieron nuevos documentos, incrementar offset y limit_factor para buscar más
                    if new_docs_added == 0:
                        offset += limit_value
                        limit_factor += 1
                    else:
                        # Si encontramos algunos documentos, actualizar la consola
                        if debug_prints:
                            print(f"📄 Found {new_docs_added} more unique documents using similar question 1")
                        break
                        
                except Exception as e:
                    logger.error(f"Error en búsqueda de similar_question_1: {e}")
                    break
            
            if debug_prints:
                print(f"📄 Retrieved {docs_added}/{docs_to_find} unique documents using similar question 1")
        
        # 3. Recuperar documentos con la segunda pregunta similar
        if similar_question_2_docs > 0 and len(similar_question_embeddings) >= 2:
            # Intentar recuperar suficientes documentos únicos
            docs_to_find = similar_question_2_docs
            offset = 0
            limit_factor = 2  # Factor inicial para el límite (entero)
            docs_added = 0
            
            # Limitar el número de intentos para evitar bucles infinitos
            max_attempts = 3
            attempts = 0
            
            while docs_added < docs_to_find and attempts < max_attempts:
                attempts += 1
                # Solicitar más documentos para tener margen en caso de duplicados
                # Asegurarse que limit sea un entero
                limit_value = int(docs_to_find * limit_factor)
                if limit_value <= 0:
                    limit_value = 1  # Asegurar un mínimo de 1 documento
                
                try:
                    similar_2_results = await qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=similar_question_embeddings[1],
                        limit=limit_value,
                        offset=int(offset),  # Conversión explícita a entero
                        search_params=models.SearchParams(
                            exact=False,
                            hnsw_ef=128
                        )
                    )
                    
                    # Si no hay resultados, salir del bucle
                    if not similar_2_results:
                        break
                    
                    # Filtrar duplicados
                    new_docs_added = 0
                    for doc in similar_2_results:
                        if docs_added >= docs_to_find:
                            break
                        
                        doc_id = doc.id
                        if doc_id not in retrieved_doc_ids:
                            results.append(doc)
                            retrieved_doc_ids.add(doc_id)
                            docs_added += 1
                            new_docs_added += 1
                    
                    # Si no se añadieron nuevos documentos, incrementar offset y limit_factor para buscar más
                    if new_docs_added == 0:
                        offset += limit_value
                        limit_factor += 1
                    else:
                        # Si encontramos algunos documentos, actualizar la consola
                        if debug_prints:
                            print(f"📄 Found {new_docs_added} more unique documents using similar question 2")
                        break
                        
                except Exception as e:
                    logger.error(f"Error en búsqueda de similar_question_2: {e}")
                    break
            
            if debug_prints:
                print(f"📄 Retrieved {docs_added}/{docs_to_find} unique documents using similar question 2")
        
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
        
        prompt = f"""Below is an instruction that describes a task. Write a response using the evidence provided for it and state
 your explanation supporting your response.
 ### Evidence:
 {context}
 ### Instruction: 
 {query}
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
        logger.error(f"Error en traditional_rag_combined_retrieve: {e}")
        raise  # Re-lanzar la excepción para que sea capturada por el benchmark
    finally:
        if qdrant_client:
            await qdrant_client.close() 