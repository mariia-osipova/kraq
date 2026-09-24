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

# Definir nombre de colección específico para este tipo de preguntas
COLLECTION_NAME_BASE = COLLECTIONS["base"]

def load_json_file(file_path: str) -> List[Dict]:
    """Carga datos desde un archivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_collection(client):
    """Asegura que la colección específica para el modelo base exista."""
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if COLLECTION_NAME_BASE not in collection_names:
        print(f"Creando colección '{COLLECTION_NAME_BASE}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME_BASE,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC)
        )
        print(f"Colección '{COLLECTION_NAME_BASE}' creada exitosamente.")
    else:
        print(f"Colección '{COLLECTION_NAME_BASE}' ya existe.")

def save_to_qdrant(questions_list: List[Dict]):
    """Genera embeddings con Ollama y guarda las preguntas en Qdrant."""
    print(f"\nGuardando preguntas generadas con el modelo base en Qdrant (Collection: {COLLECTION_NAME_BASE})...")
    client = get_configured_qdrant_client()
    
    # Asegurar que la colección exista
    ensure_collection(client)

    batch_size = 50
    points_batch = []
    processed_count = 0
    failed_embeddings = 0

    for community in tqdm(questions_list, desc="Procesando comunidades"):
        community_id = community.get('community_id', 'unknown')
        level = community.get('level', 0)
        
        if 'questions' not in community:
            print(f"Warning: No se encontraron preguntas para la comunidad {community_id}")
            continue
            
        for q_idx, question in enumerate(community['questions']):
            # Verificar que el texto de la pregunta exista
            if 'question' not in question or not question['question']:
                continue
                
            # Generar embedding usando Ollama
            embedding = get_embedding_from_ollama(question['question'])
            
            if not embedding:
                print(f"Warning: No se pudo generar embedding para: {question['question'][:50]}...")
                failed_embeddings += 1
                continue
                
            # Generar ID único usando UUID
            unique_id = str(uuid.uuid4())
            
            # Preparar payload para Qdrant
            payload = {
                'question': question['question'],
                'level': level,
                'type': question.get('type', 'unknown'),
                'summary': community.get('summary', ''),
                'explanation': question.get('finding_explanation', ''),
                'community_id': community_id
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
                        collection_name=COLLECTION_NAME_BASE,
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
                collection_name=COLLECTION_NAME_BASE,
                points=points_batch,
                wait=True
            )
        except Exception as e:
            print(f"\nError durante Qdrant upsert final: {e}")
    
    print(f"\nCarga de modelo base completada:")
    print(f"Total de preguntas procesadas exitosamente: {processed_count}")
    print(f"Total de preguntas omitidas por fallo de embedding: {failed_embeddings}")

def main():
    """Función principal para cargar archivos JSON y guardarlos en Qdrant."""
    output_dir = os.path.join(project_root, 'output')
    json_files = []
    
    if not os.path.isdir(output_dir):
        print(f"Error: El directorio '{output_dir}' no existe.")
        return
    
    # Buscar archivos JSON específicos del modelo base
    for file in os.listdir(output_dir):
        # Solo incluir archivos que contengan específicamente community_questions_base_main_only_level_
        if 'community_questions_base_main_only_level_' in file and file.endswith('.json'):
            try:
                level = int(file.split('_')[-1].replace('.json', ''))
                json_files.append((os.path.join(output_dir, file), level))
            except ValueError:
                print(f"Warning: No se pudo extraer el nivel de: {file}")
    
    if not json_files:
        print(f"No se encontraron archivos 'community_questions_base_main_only_level_*.json' en '{output_dir}'.")
        return
    
    # Mostrar explícitamente qué archivos se están cargando
    print(f"Se cargarán {len(json_files)} archivos para el modelo BASE:")
    for path, level in json_files:
        print(f" - {os.path.basename(path)} (Nivel {level})")
    
    json_files.sort(key=lambda x: x[1])  # Ordenar por nivel
    all_questions = []
    
    print(f"Encontrados {len(json_files)} archivos JSON para procesar.")
    for file_path, level in json_files:
        print(f"Cargando archivo: {file_path} (Nivel {level})")
        if os.path.exists(file_path):
            try:
                data = load_json_file(file_path)
                all_questions.extend(data)
                print(f" -> {len(data)} comunidades cargadas desde este archivo.")
            except Exception as e:
                print(f"Error procesando archivo {file_path}: {e}")
        else:
            print(f"Warning: El archivo {file_path} ya no existe.")
    
    if not all_questions:
        print("No se procesaron preguntas válidas desde los archivos JSON.")
        return
    
    print(f"\nTotal de comunidades extraídas de los JSON: {len(all_questions)}")
    save_to_qdrant(all_questions)

if __name__ == "__main__":
    main()
