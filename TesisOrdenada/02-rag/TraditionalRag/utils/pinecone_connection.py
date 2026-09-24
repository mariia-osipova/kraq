"""
Módulo para manejar la conexión con Pinecone
"""
from pinecone import Pinecone, ServerlessSpec
import os
from loguru import logger

def get_pinecone_connection(dimension=1536, index_name="harry-potter-index"):
    """
    Crea una conexión a Pinecone y asegura que el índice exista
    
    Args:
        dimension: Dimensión de los embeddings (por defecto 1536 para text-embedding-3-small)
        index_name: Nombre del índice a utilizar
        
    Returns:
        Objeto Index de Pinecone
    """
    logger.info("Conectando a Pinecone...")
    
    # Obtener API key de las variables de entorno o usar la proporcionada
    api_key = os.environ["PINECONE_API_KEY"]  # ver .env.example (clave removida del codigo)
    environment = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
    
    pc = Pinecone(api_key=api_key)
    
    # Verificar si el índice ya existe
    existing_indexes = pc.list_indexes().names()
    
    if index_name not in existing_indexes:
        logger.info(f"Creando nuevo índice '{index_name}' con dimensión {dimension}...")
        
        # Configuración del índice serverless para el plan gratuito
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Región soportada por el plan gratuito
        )
        
        pc.create_index(
            name=index_name,
            dimension=dimension,  # Dimensión de los embeddings
            metric="cosine",
            spec=spec
        )
        logger.info(f"Índice '{index_name}' creado exitosamente")
    else:
        logger.info(f"Usando índice existente '{index_name}'")
    
    # Devolver el objeto de índice
    return pc.Index(index_name) 