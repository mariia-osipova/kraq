# src/Benchmarks/calculate_questions_similarity_base.py
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List
# Eliminar: from sentence_transformers import SentenceTransformer

# Añadir src al path para importar módulos correctamente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar los módulos de Qdrant y BERTScore
from src.DataBase.qdrant_config import get_configured_qdrant_client, get_embedding_from_ollama, COLLECTIONS
from src.utils.bert_score_utils import calculate_bert_score, get_bert_score_statistics, get_top_bottom_scores

# Nombre de la colección específica para preguntas del modelo base
COLLECTION_NAME = COLLECTIONS["base"]

def load_benchmark_questions():
    """Carga las preguntas desde el archivo de benchmark local."""
    benchmark_path = os.path.join(project_root, "src", "Benchmarks", "qa_benchmark.json")
    
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"No se encontró el archivo de benchmark en {benchmark_path}")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    # Extraer solo las preguntas
    questions = [item['question'] for item in qa_data]
    
    return questions

def find_most_similar_question(query_question: str, client) -> Dict:
    """Busca la pregunta más similar en la colección de Qdrant."""
    print(f"\nBuscando pregunta similar a: {query_question}")
    
    try:
        # Usar Ollama para generar embeddings
        query_vector = get_embedding_from_ollama(query_question)
        
        if not query_vector:
            print(f"Error: No se pudo generar embedding para la consulta: {query_question}")
            return None
        
        # Realizar búsqueda en Qdrant
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,
            with_payload=True
        )
        
        if search_result:
            match = search_result[0]
            return {
                "pregunta_original": query_question,
                "pregunta_cercana": match.payload["question"],
                "similitud_coseno": match.score,
                "qdrant_id": match.id,
                "tipo_pregunta": match.payload.get("type", "desconocido"),
                "nivel": match.payload.get("level", "desconocido")
            }
        else:
            print(f"No se encontraron resultados similares")
            return None
            
    except Exception as e:
        print(f"Error general:")
        print(f"- Mensaje: {str(e)}")
        print(f"- Tipo: {type(e)}")
        import traceback
        print(f"- Traceback:\n{traceback.format_exc()}")
        return None

def calculate_and_save_proximities():
    """Calcula la similitud entre preguntas del benchmark y las almacenadas en Qdrant (modelo base)."""
    print("=== Iniciando proceso de similitud (Modelo Base) ===")
    
    # Ya no es necesario cargar el modelo SentenceTransformer
    print("\nUtilizando Ollama para embeddings con el mismo modelo que se usó para almacenar las preguntas.")
    
    print("\nConectando a Qdrant...")
    client = get_configured_qdrant_client()
    
    # Comprobar que la colección existe
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Error: La colección '{COLLECTION_NAME}' no existe en Qdrant.")
            print(f"Colecciones disponibles: {collection_names}")
            return None
    except Exception as e:
        print(f"Error conectando con Qdrant: {e}")
        return None
    
    print("\nCargando preguntas del benchmark...")
    benchmark_questions = load_benchmark_questions()
    print(f"Cargadas {len(benchmark_questions)} preguntas del benchmark")
    
    batch_size = 3
    resultados = []
    
    try:
        for i in range(0, len(benchmark_questions), batch_size):
            batch = benchmark_questions[i:i+batch_size]
            print(f"\nProcesando lote {i//batch_size + 1}/{len(benchmark_questions)//batch_size + 1}")
            
            for question in batch:
                resultado = find_most_similar_question(question, client)
                if resultado:
                    resultados.append(resultado)
                    print(f"Similitud encontrada: {resultado['similitud_coseno']:.4f}")
            
            time.sleep(0.5)  # Para no saturar el servidor Qdrant
    
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario. Guardando resultados parciales...")
    finally:
        if resultados:
            similitudes = [r['similitud_coseno'] for r in resultados]
            estadisticas = {
                "total_preguntas": len(resultados),
                "similitud_media": float(np.mean(similitudes)),
                "similitud_mediana": float(np.median(similitudes)),
                "desviacion_estandar": float(np.std(similitudes)),
                "similitud_maxima": float(np.max(similitudes)),
                "similitud_minima": float(np.min(similitudes)),
                "percentil_25": float(np.percentile(similitudes, 25)),
                "percentil_75": float(np.percentile(similitudes, 75))
            }
            
            # Añadir cálculo de BERTScore
            preguntas_originales = [r['pregunta_original'] for r in resultados]
            preguntas_similares = [r['pregunta_cercana'] for r in resultados]
            
            print("\nCalculando BERTScore...")
            _, _, bert_f1 = calculate_bert_score(preguntas_originales, preguntas_similares)
            
            estadisticas_bert = get_bert_score_statistics(bert_f1)
            top_bottom_bert = get_top_bottom_scores(preguntas_originales, preguntas_similares, bert_f1)
            
            # Datos específicos de colección
            info_coleccion = {
                "nombre_coleccion": COLLECTION_NAME,
                "tipo_modelo": "base"
            }
            
            # Estadísticas por tipo de pregunta (main, finding, etc.)
            if resultados and 'tipo_pregunta' in resultados[0]:
                tipos_pregunta = {}
                for r in resultados:
                    tipo = r.get('tipo_pregunta', 'desconocido')
                    if tipo not in tipos_pregunta:
                        tipos_pregunta[tipo] = []
                    tipos_pregunta[tipo].append(r['similitud_coseno'])
                
                estadisticas_por_tipo = {}
                for tipo, valores in tipos_pregunta.items():
                    if valores:
                        estadisticas_por_tipo[tipo] = {
                            "cantidad": len(valores),
                            "similitud_media": float(np.mean(valores)),
                            "similitud_mediana": float(np.median(valores)),
                            "similitud_maxima": float(np.max(valores)),
                            "similitud_minima": float(np.min(valores))
                        }
                info_coleccion["estadisticas_por_tipo"] = estadisticas_por_tipo
            
            # Modificar el resultado final para incluir BERTScore e info de colección
            resultado_final = {
                "info_coleccion": info_coleccion,
                "estadisticas_coseno": estadisticas,
                "estadisticas_bert": estadisticas_bert,
                "mejores_peores_bert": top_bottom_bert,
                "resultados": resultados
            }
            
            # Crear directorio si no existe
            os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
            
            # Guardar resultados
            output_path = os.path.join(project_root, "output", "similitudes_base.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(resultado_final, f, ensure_ascii=False, indent=2)
            
            print(f"\nResultados guardados en: {output_path}")
            
            # Imprimir estadísticas
            print("\n=== Estadísticas de Similitud (Modelo Base) ===")
            print(f"Similitud Media (Coseno): {estadisticas['similitud_media']:.4f}")
            print(f"BERTScore Media: {estadisticas_bert['media']:.4f}")
            print(f"Total de preguntas procesadas: {estadisticas['total_preguntas']}")
            
            return resultado_final
        return resultados

if __name__ == "__main__":
    print("Comenzando benchmark para modelo base (questions-index-base)")
    resultados = calculate_and_save_proximities()
    if isinstance(resultados, dict):  # Si tenemos el nuevo formato con estadísticas
        print("\nResultados de similitud (primeros 5):")
        for resultado in resultados['resultados'][:5]:
            print(f"Pregunta original: {resultado['pregunta_original']}")
            print(f"Pregunta similar: {resultado['pregunta_cercana']}")
            print(f"Similitud: {resultado['similitud_coseno']:.4f}")
            print(f"Tipo: {resultado.get('tipo_pregunta', 'N/A')}")
            print("-" * 40)
    else:  # Formato antiguo o error
        if resultados:
            print("\nResultados de similitud:")
            for resultado in resultados:
                print(f"Pregunta original: {resultado['pregunta_original']}")
                print(f"Pregunta similar: {resultado['pregunta_cercana']}")
                print(f"Similitud: {resultado['similitud_coseno']:.4f}")
                print("-" * 40)
