from time import perf_counter
import asyncio
from loguru import logger
from openai import AsyncOpenAI
from typing import List, Dict, Any
from pathlib import Path
from qdrant_client import AsyncQdrantClient, models

# Configuración para modelos locales
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"

async def get_qdrant_client():
    # Usar ruta absoluta al directorio raíz del proyecto
    path = Path(__file__).parent.parent / "qdrant_client"
    return AsyncQdrantClient(path=path)

async def traditional_rag_similar_question(
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
    Implementación de RAG basado en recuperación de pregunta similar
    
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
            print("🔍 RUNNING TRADITIONAL RAG WITH SIMILAR QUESTION")
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
            
        # Etapa 2: Buscar la pregunta más similar en la colección de preguntas
        similar_question_start_time = perf_counter()
        similar_question_result = await qdrant_client.search(
            collection_name=question_collection,
            query_vector=query_embedding,
            limit=1,  # Solo necesitamos la pregunta más similar
            search_params=models.SearchParams(
                exact=False,
                hnsw_ef=128
            )
        )
        
        if not similar_question_result:
            logger.warning(f"No se encontraron preguntas similares en {question_collection}")
            similar_question = query  # Usar la consulta original si no hay resultados
            similar_question_embedding = query_embedding
        else:
            # Extraer la pregunta similar
            similar_point = similar_question_result[0]
            similar_question = similar_point.payload.get('question', query)
            similar_question_score = similar_point.score
            
            # Usar el embedding almacenado si está disponible
            if hasattr(similar_point, 'vector') and similar_point.vector:
                similar_question_embedding = similar_point.vector
            else:
                # Si no hay embedding almacenado, generamos uno nuevo
                embedding_response = await embeddings_client.embeddings.create(
                    input=similar_question,
                    model=embedding_model
                )
                similar_question_embedding = embedding_response.data[0].embedding
        
        similar_question_time = perf_counter() - similar_question_start_time
        
        if debug_prints:
            print(f"📝 Similar question found: {similar_question}")
            if 'similar_question_score' in locals():
                print(f"📊 Similarity score: {similar_question_score:.4f}")
            print(f"⏱️ Similar question search time: {similar_question_time:.4f} seconds")

        # Etapa 3: Recuperar documentos relevantes usando la pregunta similar
        retrieval_start_time = perf_counter()
        search_result = await qdrant_client.search(
            collection_name=collection_name,
            query_vector=similar_question_embedding,
            limit=top_k,
            search_params=models.SearchParams(
                exact=False,
                hnsw_ef=128
            )
        )
        retrieval_time = perf_counter() - retrieval_start_time
        
        # Siempre imprimimos el tiempo de recuperación para benchmarking
        print(f"retrieval_time:{retrieval_time}")
        
        if debug_prints or verbose_timing:
            print(f"⏱️ Document retrieval time: {retrieval_time:.4f} seconds")
            print(f"📄 Retrieved {len(search_result)} documents")

        # Etapa 4: Preparar contexto
        context_start_time = perf_counter()
        
        documents = []
        for point in search_result:
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
    finally:
        if qdrant_client:
            await qdrant_client.close() 