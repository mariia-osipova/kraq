import json
import os
import sys
from typing import List, Dict
from tqdm import tqdm
import uuid

# Añadir src al path para importar módulos correctamente si se ejecuta como script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar configuración y funciones de Qdrant/Ollama
from src.DataBase.qdrant_config import (
    get_configured_qdrant_client,
    get_embedding_from_ollama,
    COLLECTIONS,
    VECTOR_DIMENSION,
    DISTANCE_METRIC
)
from qdrant_client.http.models import PointStruct, VectorParams

# Definir nombre de colección específico para preguntas aleatorias
COLLECTION_NAME_RANDOM = COLLECTIONS["random"]

def load_json_file(file_path: str) -> List[Dict]:
    """Carga datos desde un archivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_collection(client):
    """Asegura que la colección específica para preguntas aleatorias exista."""
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if COLLECTION_NAME_RANDOM not in collection_names:
        print(f"Creando colección '{COLLECTION_NAME_RANDOM}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME_RANDOM,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC)
        )
        print(f"Colección '{COLLECTION_NAME_RANDOM}' creada exitosamente.")
    else:
        print(f"Colección '{COLLECTION_NAME_RANDOM}' ya existe.")

def save_to_qdrant(questions: List[Dict]):
    """Genera embeddings con Ollama y guarda las preguntas aleatorias/naive en Qdrant."""
    print(f"\nGuardando {len(questions)} preguntas aleatorias en Qdrant (Collection: {COLLECTION_NAME_RANDOM})...")
    client = get_configured_qdrant_client()
    
    # Asegurar que la colección exista
    ensure_collection(client)

    batch_size = 50
    points_batch = []
    processed_count = 0
    failed_embeddings = 0

    for i, question in enumerate(tqdm(questions, desc="Procesando preguntas")):
        if 'question' not in question or not question['question']:
            continue
            
        # Generar embedding usando Ollama
        embedding = get_embedding_from_ollama(question['question'])
        
        if not embedding:
            print(f"Warning: No se pudo generar embedding para: {question['question'][:50]}...")
            failed_embeddings += 1
            continue
            
        # Generar ID único usando UUID
        unique_id = str(uuid.uuid4())  # Usar UUID en lugar de "random:X"
        
        # Preparar payload para Qdrant
        payload = {
            'question': question['question'],
            'type': 'naive',  # Todas son de tipo 'naive'
            'random_index': i,  # Guardar el índice original como parte del payload
            'fragments': question.get('fragments', []),  # Archivos de origen
        }
        
        # Crear PointStruct para Qdrant
        point = PointStruct(
            id=unique_id,
            vector=embedding,
            payload=payload
        )
        
        points_batch.append(point)
        processed_count += 1
        
        # Insertar lote en Qdrant cuando alcance el tamaño deseado
        if len(points_batch) >= batch_size:
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME_RANDOM,
                    points=points_batch,
                    wait=True
                )
            except Exception as e:
                print(f"\nError durante Qdrant upsert: {e}")
            points_batch = []
    
    # Insertar el último lote si existe
    if points_batch:
        try:
            client.upsert(
                collection_name=COLLECTION_NAME_RANDOM,
                points=points_batch,
                wait=True
            )
        except Exception as e:
            print(f"\nError durante Qdrant upsert final: {e}")
    
    print(f"\nCarga de preguntas aleatorias completada:")
    print(f"Total de preguntas procesadas exitosamente: {processed_count}")
    print(f"Total de preguntas omitidas por fallo de embedding: {failed_embeddings}")

def main():
    """Función principal para cargar archivos JSON y guardarlos en Qdrant."""
    output_dir = os.path.join(project_root, 'output')
    
    if not os.path.isdir(output_dir):
        print(f"Error: El directorio '{output_dir}' no existe.")
        return
    
    # Buscar el archivo de preguntas naive/aleatorias
    naive_file = os.path.join(output_dir, "naive_questions.json")
    
    if not os.path.exists(naive_file):
        # Intentar con el archivo de progreso si el final no existe
        naive_file = os.path.join(output_dir, "naive_questions_base_progress.json")
        
        if not os.path.exists(naive_file):
            print(f"No se encontró ningún archivo de preguntas aleatorias en '{output_dir}'.")
            print("Busque 'naive_questions_base_final.json' o 'naive_questions_base_progress.json'")
            return
    
    print(f"Cargando archivo: {naive_file}")
    try:
        questions = load_json_file(naive_file)
        total_questions = len(questions)
        print(f"Se cargaron {total_questions} preguntas aleatorias.")
        
        # Limitar a las primeras 17378 preguntas
        max_questions = 17378
        if total_questions > max_questions:
            questions = questions[:max_questions]
            print(f"Se procesarán solo las primeras {max_questions} preguntas.")
            
        save_to_qdrant(questions)
    except Exception as e:
        print(f"Error procesando archivo {naive_file}: {e}")

if __name__ == "__main__":
    main()
