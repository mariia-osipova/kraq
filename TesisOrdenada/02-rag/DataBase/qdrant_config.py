import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import os

# Configuración para Qdrant on-disk y Ollama
# Por defecto, la base de datos se guardará en una carpeta 'qdrant_db'
# en el directorio desde donde se ejecute el script.
QDRANT_DB_PATH = os.getenv("QDRANT_DB_PATH", "./qdrant_client")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") # URL base para la API compatible con OpenAI
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "questions-index"  # Default collection
VECTOR_DIMENSION = 768 # Dimensión para nomic-embed-text
DISTANCE_METRIC = Distance.COSINE

# Define all collections
COLLECTIONS = {
    "default": COLLECTION_NAME,
    "base": "questions-index-base",
    "finetuned": "questions-index-finetuned",
    "random": "questions-index-random",
    "benchmark": "questions-benchmark",
    "chunks": "chunks"
}

# --- Cliente OpenAI para Ollama ---
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key='ollama', # Clave API requerida, 'ollama' es común para Ollama local
)

def get_qdrant_client():
    """
    Obtiene una instancia del cliente Qdrant configurado para almacenamiento on-disk.
    Los datos se guardarán en la ruta especificada por QDRANT_DB_PATH.
    """
    print(f"Initializing Qdrant client with on-disk storage at: {os.path.abspath(QDRANT_DB_PATH)}")
    # Usar 'path' para almacenamiento local en disco
    client = qdrant_client.QdrantClient(
        path=QDRANT_DB_PATH
    )
    return client

def ensure_qdrant_collection(client: qdrant_client.QdrantClient, collection_name=None):
    """Asegura que la colección exista en Qdrant con la configuración correcta."""
    # Use the provided collection_name or default to COLLECTION_NAME
    collection_name = collection_name or COLLECTION_NAME
    
    try:
        # Verificar si la colección ya existe
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]

        if collection_name in collection_names:
             print(f"Collection '{collection_name}' already exists.")
        else:
            print(f"Collection '{collection_name}' does not exist. Creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC)
            )
            print(f"Collection '{collection_name}' created successfully.")

    except Exception as e:
        # Si get_collections falla (ej. directorio no existe aún), intentar crearla directamente
        print(f"Could not get collections (error: {e}). Attempting to create '{collection_name}'...")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC)
            )
            print(f"Collection '{collection_name}' created successfully.")
        except Exception as create_e:
            print(f"Failed to create collection '{collection_name}': {create_e}")
            # Relanzar la excepción si la creación también falla
            raise create_e

def get_embedding_from_ollama(text: str) -> list[float]:
    """Genera embeddings para el texto dado usando Ollama a través de la API compatible con OpenAI."""
    try:
        # Limpiar/reemplazar saltos de línea que pueden causar problemas
        cleaned_text = text.replace("\n", " ")

        # Usar el cliente OpenAI para obtener embeddings
        response = ollama_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[cleaned_text] # La API espera una lista de textos
        )

        # Extraer el embedding de la respuesta
        if response.data and len(response.data) > 0 and response.data[0].embedding:
            return response.data[0].embedding
        else:
            print(f"Error: No embedding data received from Ollama for text: {cleaned_text[:50]}...")
            print(f"Ollama response object: {response}")
            return []
    except Exception as e:
        # Capturar excepciones generales de la API de OpenAI/Ollama
        print(f"Error getting embedding from Ollama (via OpenAI client) for text: {text[:50]}...")
        print(f"Error: {e}")
        print(f"Please ensure Ollama is running, '{EMBEDDING_MODEL}' is available,")
        print(f"and the base URL '{OLLAMA_BASE_URL}' is correct.")
        return []

# --- Inicialización ---
qdrant_client_instance = get_qdrant_client()
ensure_qdrant_collection(qdrant_client_instance)

# --- Funciones de conveniencia ---
def get_configured_qdrant_client():
    """Devuelve la instancia del cliente Qdrant ya configurada para on-disk."""
    return qdrant_client_instance

def ensure_all_collections():
    """Asegura que todas las colecciones definidas existan en Qdrant."""
    client = get_configured_qdrant_client()
    for _, collection_name in COLLECTIONS.items():
        ensure_qdrant_collection(client, collection_name)
    return client

if __name__ == '__main__':
    print("Testing Ollama embedding function (via OpenAI client)...")
    sample_text = "Este es un texto de prueba usando el cliente OpenAI."
    embedding = get_embedding_from_ollama(sample_text)
    if embedding:
        print(f"Generated embedding for '{sample_text}':")
        print(f"Dimension: {len(embedding)}")
    else:
        print("Failed to generate embedding.")

    print("\nTesting Qdrant client (on-disk)...")
    client = get_configured_qdrant_client()
    try:
        collections_info = client.get_collections()
        print("Successfully connected to Qdrant (on-disk).")
        print(f"Available collections: {collections_info.collections}")
        # Verificar que la carpeta exista
        db_path = os.path.abspath(QDRANT_DB_PATH)
        if os.path.exists(db_path) and os.path.isdir(db_path):
            print(f"Qdrant database directory exists at: {db_path}")
        else:
             print(f"Warning: Qdrant database directory not found at: {db_path}")
    except Exception as e:
        print(f"Failed to interact with Qdrant (on-disk): {e}")
