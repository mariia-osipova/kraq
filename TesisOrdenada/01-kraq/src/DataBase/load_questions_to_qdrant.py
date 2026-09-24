import json
import os
import sys
from typing import List, Dict
# from sentence_transformers import SentenceTransformer # Ya no se usa
# import numpy as np # Ya no se usa
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
# Importar PointStruct para la inserción en Qdrant
from qdrant_client.http.models import PointStruct, VectorParams

COLLECTION_NAME_FINETUNED = COLLECTIONS["finetuned"]  # Use the centralized definition

def load_json_file(file_path: str) -> List[Dict]:
    """Carga datos desde un archivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_questions(data: List[Dict], level: int) -> List[Dict]:
    """Procesa los datos crudos del JSON para extraer información relevante de las preguntas."""
    processed_questions = []
    for community in data:
        if 'questions' not in community:
            continue
        for question in community['questions']:
            question_data = {
                'community_id': community['community_id'],
                'level': level,
                'question_type': question['type'],
                'question_text': question['question'].strip('\"'),
                'summary': community.get('summary', ''),
            }
            if question['type'] == 'finding' and 'finding_explanation' in question:
                question_data['explanation'] = question['finding_explanation']
            processed_questions.append(question_data)
    return processed_questions

def ensure_collection(client):
    """Asegura que la colección específica para el modelo finetuneado exista."""
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if COLLECTION_NAME_FINETUNED not in collection_names:
        print(f"Creando colección '{COLLECTION_NAME_FINETUNED}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME_FINETUNED,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=DISTANCE_METRIC)
        )
        print(f"Colección '{COLLECTION_NAME_FINETUNED}' creada exitosamente.")
    else:
        print(f"Colección '{COLLECTION_NAME_FINETUNED}' ya existe.")

def save_to_qdrant(questions: List[Dict]):
    """Genera embeddings con Ollama y guarda las preguntas en Qdrant."""
    # Ya no se carga el modelo SentenceTransformer aquí
    # print("Cargando modelo de embeddings...") # El embedding se hace por llamada API

    # Contar tipos de preguntas (igual que antes)
    question_types = {}
    type_counters = {}
    for q in questions:
        q_type = q['question_type']
        question_types[q_type] = question_types.get(q_type, 0) + 1
        type_counters[q_type] = 0

    print("\nDistribución de tipos de preguntas:")
    for q_type, count in question_types.items():
        print(f"- {q_type}: {count}")

    print(f"\nGuardando {len(questions)} preguntas en Qdrant (Collection: {COLLECTION_NAME_FINETUNED})...")
    # Obtener cliente Qdrant
    client = get_configured_qdrant_client()
    ensure_collection(client)  # Asegurar que la colección existe

    batch_size = 50 # Reducir un poco el batch size puede ser útil si la generación de embeddings es lenta
    points_batch = [] # Lote de puntos para Qdrant
    processed_count = 0
    failed_embeddings = 0

    for i in tqdm(range(len(questions)), desc="Processing questions"):
        question = questions[i]

        # 1. Generar embedding usando Ollama
        embedding = get_embedding_from_ollama(question['question_text'])

        if not embedding:
            print(f"Warning: Failed to generate embedding for question: {question['question_text'][:50]}...")
            failed_embeddings += 1
            continue # Saltar esta pregunta si no se pudo generar el embedding

        # 2. Generar ID único usando UUID en lugar de formato de string
        unique_id = str(uuid.uuid4())

        # 3. Preparar payload (metadatos) para Qdrant
        payload = {
            'question': question['question_text'],
            'level': question['level'],
            'type': question['question_type'],
            'model_type': 'finetuned',  # Identificador para modelo finetuneado
            'summary': question['summary'],
            'explanation': question.get('explanation', ''),
            'community_id': question['community_id']
        }

        # 4. Crear PointStruct para Qdrant
        point = PointStruct(
            id=unique_id,
            vector=embedding,
            payload=payload
        )
        points_batch.append(point)
        processed_count +=1

        # 5. Insertar lote en Qdrant cuando alcance el tamaño deseado o al final
        if len(points_batch) >= batch_size or i == len(questions) - 1:
            if points_batch: # Asegurarse de que haya puntos para insertar
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME_FINETUNED,
                        points=points_batch,
                        wait=True # Esperar a que la operación se complete
                    )
                    # print(f"Upserted batch of {len(points_batch)} points.") # Descomentar para debug
                except Exception as e:
                    print(f"\nError during Qdrant upsert: {e}")
                    # Podrías añadir lógica para reintentar o guardar los puntos fallidos
                points_batch = [] # Limpiar lote

    print(f"\nProcesamiento completado.")
    print(f"Total de preguntas procesadas exitosamente: {processed_count}")
    print(f"Total de preguntas omitidas por fallo de embedding: {failed_embeddings}")

    print("\nResumen final de carga (basado en IDs generados):")
    for q_type, counter in type_counters.items():
        # Nota: Este contador refleja cuántos IDs se generaron, no necesariamente cuántos se insertaron
        # si hubo fallos de embedding para ese tipo.
        print(f"- {q_type}: {counter} IDs generados")


def main():
    """Función principal para cargar y procesar archivos JSON y guardarlos en Qdrant."""
    output_dir = 'output'
    json_files = []

    if not os.path.isdir(output_dir):
        print(f"Error: El directorio '{output_dir}' no existe.")
        return

    for file in os.listdir(output_dir):
        # Modificar para excluir explícitamente los archivos del modelo base
        if (file.startswith('community_questions_main_only_level_') and 
            not 'base_level_' in file and  # Excluir archivos del modelo base
            file.endswith('.json')):
            try:
                level = int(file.split('_')[-1].replace('.json', ''))
                json_files.append((os.path.join(output_dir, file), level))
            except ValueError:
                print(f"Warning: Could not parse level from filename: {file}")

    if not json_files:
        print(f"No se encontraron archivos 'community_questions_main_only_level_*.json' (excluyendo base) en '{output_dir}'.")
        return

    json_files.sort(key=lambda x: x[1]) # Ordenar por nivel
    all_questions = []

    print(f"Encontrados {len(json_files)} archivos JSON para procesar.")
    for file_path, level in json_files:
        print(f"Cargando archivo: {file_path} (Nivel {level})")
        if os.path.exists(file_path):
            try:
                data = load_json_file(file_path)
                processed = process_questions(data, level)
                all_questions.extend(processed)
                print(f" -> {len(processed)} preguntas procesadas desde este archivo.")
            except Exception as e:
                print(f"Error procesando archivo {file_path}: {e}")
        else:
            print(f"Warning: El archivo {file_path} ya no existe.")

    if not all_questions:
        print("No se procesaron preguntas válidas desde los archivos JSON.")
        return

    print(f"\nTotal de preguntas extraídas de los JSON: {len(all_questions)}")
    save_to_qdrant(all_questions)
    # El resumen final ahora se imprime dentro de save_to_qdrant

if __name__ == "__main__":
    main()
