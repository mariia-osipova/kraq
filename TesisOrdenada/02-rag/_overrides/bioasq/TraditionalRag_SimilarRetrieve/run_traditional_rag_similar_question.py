import asyncio
import os
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from traditional_rag_similar_question import traditional_rag_similar_question

# Cargar variables de entorno
load_dotenv()

# Configuración para modelos locales
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

# Configurar modelos
EMBEDDING_MODEL = "nomic-embed-text"  # Usando nomic-embed-text de Ollama
LLM_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # Modelo cuantizado en vLLM

# Lista de preguntas de prueba
TEST_QUESTIONS = [
    "Can sorafenib activate AMPK?"
]

async def run_traditional_rag_similar_question_test():
    """
    Ejecuta pruebas de Traditional RAG con recuperación basada en preguntas similares
    """
    logger.info("Iniciando prueba de Traditional RAG usando recuperación por pregunta similar")
    
    # Inicializar cliente para embeddings (Ollama)
    embeddings_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )
    
    # Inicializar cliente para LLM (vLLM)
    llm_client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"  # vLLM no requiere API key
    )
    
    # Procesar cada pregunta
    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Procesando pregunta {i}/{len(TEST_QUESTIONS)}")
        logger.info(f"Pregunta: {question}")
        logger.info('='*80)
        
        try:
            # Ejecutar Traditional RAG con timing detallado activado
            response = await traditional_rag_similar_question(
                query=question,
                embeddings_client=embeddings_client,  # Cliente para embeddings
                llm_client=llm_client,                # Cliente para LLM
                embedding_model=EMBEDDING_MODEL,
                llm_model=LLM_MODEL,
                top_k=15,
                debug_prints=True,
                verbose_timing=True,
                collection_name="chunks",             # Colección para documentos
                question_collection="question-index-finetuned"  # Colección para preguntas similares
            )
            
            logger.info("\nRESPUESTA FINAL:")
            logger.info("-"*80)
            logger.info(response)
            logger.info("-"*80)
            
        except Exception as e:
            logger.error(f"Error procesando la pregunta: {e}")
        
        # Pequeña pausa entre preguntas
        if i < len(TEST_QUESTIONS):
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run_traditional_rag_similar_question_test()) 