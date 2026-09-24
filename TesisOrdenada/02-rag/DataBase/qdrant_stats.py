from collections import Counter
import json
import os
import sys
from typing import Dict
from tqdm import tqdm

# Añadir src al path para importar módulos correctamente si se ejecuta como script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar configuración y cliente de Qdrant
from qdrant_config import (
    get_configured_qdrant_client,
    COLLECTION_NAME,
    COLLECTIONS,
    VECTOR_DIMENSION,
    DISTANCE_METRIC
)
# Importar modelos necesarios para scroll
from qdrant_client.http.models import ScrollRequest, FieldCondition, MatchValue

def get_qdrant_statistics(collection_name=None) -> Dict | None:
    """Obtiene estadísticas de una colección Qdrant específica y las guarda en un archivo JSON."""
    collection_name = collection_name or COLLECTION_NAME
    collection_key = None
    
    # Find the key for this collection (for naming the output file)
    for key, name in COLLECTIONS.items():
        if name == collection_name:
            collection_key = key
            break
            
    print(f"=== Obteniendo estadísticas de Qdrant para '{collection_name}' ===")

    try:
        # Conectar a Qdrant
        print(f"\nConectando a Qdrant y accediendo a la colección '{collection_name}'...")
        client = get_configured_qdrant_client()

        # Comprobar que la colección existe
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        if collection_name not in collection_names:
            print(f"Error: La colección '{collection_name}' no existe en Qdrant.")
            print(f"Colecciones disponibles: {collection_names}")
            return None
            
        # Obtener estadísticas básicas de la colección
        collection_info = client.get_collection(collection_name=collection_name)
        total_points = collection_info.points_count

        # En las versiones recientes de Qdrant, la estructura ha cambiado
        # Intentar obtener información sobre vectores de diferentes maneras
        try:
            # Intentar obtener la configuración de vectores mediante la API moderna
            vector_config = client.get_collection_config(collection_name=collection_name)
            
            # Extraer información sobre la dimensión y métrica
            if hasattr(vector_config, 'params'):
                vector_dimension = vector_config.params.get('dimension', VECTOR_DIMENSION)
                distance_metric = vector_config.params.get('metric', str(DISTANCE_METRIC))
            elif hasattr(vector_config, 'vectors'):
                # Si hay múltiples configuraciones de vectores, tomar la primera
                first_vector_config = next(iter(vector_config.vectors.values()))
                vector_dimension = first_vector_config.size
                distance_metric = first_vector_config.distance
            else:
                # Si no podemos obtener la información, usar los valores predeterminados
                print("No se pudo obtener información de vectores, usando valores predeterminados")
                vector_dimension = VECTOR_DIMENSION
                distance_metric = DISTANCE_METRIC
        except Exception as e:
            print(f"Error al obtener configuración de vectores: {e}")
            print("Usando valores predeterminados de configuración")
            vector_dimension = VECTOR_DIMENSION
            distance_metric = DISTANCE_METRIC

        print(f"\nTotal de puntos encontrados: {total_points}")
        if total_points == 0:
            print("La colección está vacía. No hay estadísticas detalladas para generar.")
            return {
                "total_preguntas": 0,
                "preguntas_por_nivel": {},
                "preguntas_por_tipo": {},
                "preguntas_con_explicacion": 0,
                "dimension_vectores": vector_dimension,
                "metrica_similitud": str(distance_metric)
            }

        # Inicializar contadores
        level_counter = Counter()
        type_counter = Counter()
        questions_with_explanation = 0

        # Usar scroll para iterar sobre todos los puntos y obtener sus payloads
        print("\nRecuperando metadatos usando scroll...")
        batch_size = 256 # Tamaño del lote para scroll
        next_page_offset = None # Para manejar la paginación de scroll

        # Usar tqdm para mostrar progreso basado en el total de puntos
        with tqdm(total=total_points, desc="Scrolling points") as pbar:
            while True:
                # Simplificado: el scroll_filter=None es suficiente, no necesitamos ScrollFilter
                results, next_page_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None, # Sin filtros, obtener todo
                    limit=batch_size,
                    offset=next_page_offset,
                    with_payload=True, # ¡Importante! Obtener el payload (metadatos)
                    with_vectors=False # No necesitamos los vectores aquí
                )

                if not results: # Si no hay más resultados, hemos terminado
                    break

                # Procesar cada punto en el lote
                for record in results:
                    payload = record.payload # El payload contiene los metadatos

                    # Contar preguntas por nivel
                    level = payload.get('level', 'unknown')
                    # Asegurarse de que el nivel sea un tipo adecuado para ordenar (ej. int o str consistente)
                    try:
                        level_counter[int(level)] += 1
                    except (ValueError, TypeError):
                        level_counter[str(level)] += 1 # Tratar como string si no es int

                    # Contar preguntas por tipo
                    type_counter[payload.get('type', 'unknown')] += 1

                    # Contar preguntas con explicación (verificar si existe y no está vacío)
                    if payload.get('explanation'):
                        questions_with_explanation += 1

                    pbar.update(1) # Actualizar barra de progreso

                if next_page_offset is None: # Si Qdrant no devuelve un offset, terminamos
                     # Esto puede ocurrir si el último lote llenó exactamente 'limit'
                     # o si el número total de puntos es múltiplo de 'limit'.
                     # Verificar si ya procesamos todos los puntos contados inicialmente.
                     if pbar.n >= total_points:
                         break
                     else:
                         # Puede ser un caso raro o un error, intentar continuar una vez más
                         # Si se queda atascado, se saldrá por la condición `if not results` arriba
                         print(f"Warning: next_page_offset is None but processed points ({pbar.n}) < total_points ({total_points}). Attempting next scroll.")
                         # El offset None debería funcionar para obtener la siguiente página en versiones recientes
                         pass


        # Preparar estadísticas
        estadisticas = {
            "total_preguntas": total_points,
            "preguntas_por_nivel": dict(sorted(level_counter.items())),
            "preguntas_por_tipo": dict(sorted(type_counter.items())), # Ordenar también por tipo
            "preguntas_con_explicacion": questions_with_explanation,
            "dimension_vectores": vector_dimension, # Usar el valor real de la colección
            "metrica_similitud": str(distance_metric).upper() # Usar el valor real y en mayúsculas
        }

        # Asegurar que el directorio de salida exista
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"qdrant_stats_{collection_key or 'custom'}.json")

        # Guardar estadísticas en un archivo
        print(f"\nGuardando estadísticas en {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(estadisticas, f, ensure_ascii=False, indent=2)
        print("Estadísticas guardadas exitosamente.")

        # Imprimir estadísticas
        print("\n=== Estadísticas de la Base de Datos (Qdrant) ===")
        print(f"Total de preguntas almacenadas: {estadisticas['total_preguntas']}")
        print(f"Dimensión de los vectores: {estadisticas['dimension_vectores']}")
        print(f"Métrica de similitud: {estadisticas['metrica_similitud']}")

        print("\nDistribución por nivel:")
        for level, count in estadisticas['preguntas_por_nivel'].items():
            print(f"  Nivel {level}: {count} preguntas")

        print("\nDistribución por tipo:")
        for qtype, count in estadisticas['preguntas_por_tipo'].items():
            print(f"  {qtype}: {count} preguntas")

        print(f"\nPreguntas con explicación: {estadisticas['preguntas_con_explicacion']}")

        return estadisticas

    except Exception as e:
        print(f"\nError al obtener estadísticas de Qdrant para '{collection_name}':")
        print(f"- Mensaje: {str(e)}")
        print(f"- Tipo: {type(e)}")
        import traceback
        print(f"- Traceback:\n{traceback.format_exc()}")
        return None

def print_sample_questions(collection_name=None, num_samples=3):
    """
    Recupera y muestra algunas preguntas de ejemplo de la colección especificada.
    
    Args:
        collection_name: Nombre de la colección de la que obtener ejemplos
        num_samples: Número de ejemplos a mostrar
    """
    collection_name = collection_name or COLLECTION_NAME
    
    print(f"\n=== Ejemplos de preguntas de la colección '{collection_name}' ===")
    
    try:
        client = get_configured_qdrant_client()
        
        # Verificar que la colección existe
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        if collection_name not in collection_names:
            print(f"Error: La colección '{collection_name}' no existe en Qdrant.")
            return
        
        # Obtener estadísticas de la colección
        collection_info = client.get_collection(collection_name=collection_name)
        total_points = collection_info.points_count
        
        if total_points == 0:
            print("La colección está vacía, no hay ejemplos para mostrar.")
            return
        
        print(f"Recuperando {num_samples} preguntas de muestra de un total de {total_points}...")
        
        # Usar scroll para obtener algunas preguntas aleatorias
        # La distribución no será perfectamente aleatoria pero es suficiente para ejemplos
        samples_found = 0
        sample_step = max(1, total_points // (num_samples * 3))  # Usar un paso para distribuir las muestras
        
        results, _ = client.scroll(
            collection_name=collection_name,
            limit=total_points,  # Usamos un límite grande para poder saltar
            with_payload=True,
            with_vectors=False  # No necesitamos los vectores
        )
        
        # Tomar muestras distribuidas
        sampled_points = []
        if results:
            indices = [i * sample_step for i in range(min(num_samples, len(results) // sample_step))]
            # Si no hay suficientes puntos con el método de paso, tomar los primeros
            if not indices:
                indices = list(range(min(num_samples, len(results))))
            
            sampled_points = [results[i] for i in indices if i < len(results)]
        
        if not sampled_points:
            print("No se pudieron recuperar preguntas de muestra.")
            return
        
        # Mostrar cada pregunta de muestra
        for i, point in enumerate(sampled_points, 1):
            payload = point.payload
            question_text = payload.get('question', 'No encontrado')
            question_type = payload.get('type', 'desconocido')
            level = payload.get('level', 'No especificado')
            
            print(f"\n--- Ejemplo {i} ---")
            print(f"ID: {point.id}")
            print(f"Pregunta: {question_text}")
            print(f"Tipo: {question_type}")
            
            # Mostrar nivel si está disponible
            if level != 'No especificado':
                print(f"Nivel: {level}")
            
            # Mostrar explicación si está disponible
            if 'explanation' in payload and payload['explanation']:
                # Truncar explicaciones largas
                explanation = payload['explanation']
                if len(explanation) > 150:
                    explanation = explanation[:147] + "..."
                print(f"Explicación: {explanation}")
            
            # Mostrar otros campos relevantes según el tipo de colección
            if collection_name == COLLECTIONS["random"] and 'fragments' in payload:
                if payload['fragments']:
                    print(f"Fragmentos: {len(payload['fragments'])} archivos utilizados")
    
    except Exception as e:
        print(f"Error al recuperar preguntas de muestra: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def get_all_collections_statistics() -> Dict:
    """Obtiene estadísticas para todas las colecciones definidas y muestra ejemplos."""
    results = {}
    
    client = get_configured_qdrant_client()
    existing_collections = [col.name for col in client.get_collections().collections]
    
    for key, collection_name in COLLECTIONS.items():
        if collection_name in existing_collections:
            print(f"\n\n==== Procesando colección '{collection_name}' ({key}) ====")
            stats = get_qdrant_statistics(collection_name)
            if stats:
                results[key] = stats
                # Añadir el muestreo después de obtener las estadísticas
                if stats["total_preguntas"] > 0:
                    print_sample_questions(collection_name)
                
    # Save combined stats
    if results:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "qdrant_stats_all.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nEstadísticas combinadas guardadas en {output_file}")
        
    return results

if __name__ == "__main__":
    # Si se proporciona --samples como argumento, solo mostrar ejemplos
    if len(sys.argv) > 1 and sys.argv[1] == "--samples":
        if len(sys.argv) > 2 and sys.argv[2] in COLLECTIONS:
            collection_name = COLLECTIONS[sys.argv[2]]
            print_sample_questions(collection_name, num_samples=5)
        else:
            for key, name in COLLECTIONS.items():
                print(f"\n\n==== Ejemplos de la colección '{name}' ({key}) ====")
                print_sample_questions(name, num_samples=3)
    # Si se proporciona el nombre de una colección, obtener estadísticas y ejemplos de esa colección
    elif len(sys.argv) > 1:
        collection_key = sys.argv[1]
        if collection_key in COLLECTIONS:
            stats = get_qdrant_statistics(COLLECTIONS[collection_key])
            if stats and stats["total_preguntas"] > 0:
                print_sample_questions(COLLECTIONS[collection_key])
        else:
            print(f"Error: Colección '{collection_key}' no reconocida.")
            print(f"Colecciones disponibles: {list(COLLECTIONS.keys())}")
    # De lo contrario, obtener estadísticas de todas las colecciones
    else:
        get_all_collections_statistics()
