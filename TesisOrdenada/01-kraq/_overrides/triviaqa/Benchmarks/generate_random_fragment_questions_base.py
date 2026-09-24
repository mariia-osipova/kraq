import os
import random
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

# Añadir src al path para importar módulos correctamente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar los módulos necesarios
from src.DataBase.qdrant_config import get_configured_qdrant_client, get_embedding_from_ollama, COLLECTIONS
from src.utils.bert_score_utils import calculate_bert_score, get_bert_score_statistics, get_top_bottom_scores

# Tokenizer para contar tokens (sin cargar el modelo completo)
_tokenizer = None

def get_tokenizer():
    """Obtiene el tokenizer del modelo sin cargar el modelo completo."""
    global _tokenizer
    if _tokenizer is None:
        from src.QuestionGeneration.llm_inference_base import BASE_MODEL_ID
        print(f"Cargando tokenizer para {BASE_MODEL_ID}...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=True)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
        except Exception as e:
            print(f"Error cargando tokenizer: {e}")
            # Tokenizer de respaldo si no se puede cargar el específico
            _tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _tokenizer

def select_random_fragment(text: str, max_tokens: int = 300) -> str:
    """Selecciona un fragmento aleatorio del texto que no exceda el número máximo de tokens."""
    tokenizer = get_tokenizer()
    
    # Tokenizar el texto completo
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    
    if total_tokens <= max_tokens:
        return text
    
    # Calcular el tamaño de la ventana y elegir un punto de inicio aleatorio
    window_size = max_tokens
    max_start_idx = total_tokens - window_size
    start_token_idx = random.randint(0, max_start_idx)
    end_token_idx = start_token_idx + window_size
    
    # Obtener los tokens del fragmento seleccionado
    selected_tokens = tokens[start_token_idx:end_token_idx]
    
    # Decodificar los tokens de vuelta a texto
    fragment = tokenizer.decode(selected_tokens, skip_special_tokens=True)
    
    # Limpiar y ajustar el fragmento
    fragment = fragment.strip()
    
    # Intentar ajustar el fragmento para que comience y termine en puntos lógicos
    sentences = fragment.split('.')
    if len(sentences) > 1:
        # Si el primer fragmento está incompleto (menos de X caracteres), eliminarlo
        if len(sentences[0]) < 20:
            sentences = sentences[1:]
        # Si el último fragmento está incompleto, eliminarlo
        if len(sentences[-1]) < 20:
            sentences = sentences[:-1]
    
    return '.'.join(sentences).strip() + '.'

def load_random_fragments(num_fragments=5, max_tokens_per_fragment=300):
    """Carga fragmentos aleatorios de texto, limitando cada uno a max_tokens_per_fragment tokens."""
    input_dir = Path(project_root) / "input"
    all_fragments = list(input_dir.glob("*.txt"))
    
    if len(all_fragments) < num_fragments:
        raise ValueError(f"Not enough fragments found. Found {len(all_fragments)}")
    
    selected_fragments = random.sample(all_fragments, num_fragments)
    fragments_content = []
    
    for fragment_path in selected_fragments:
        try:
            with open(fragment_path, 'r', encoding='utf-8') as f:
                full_content = f.read().strip()
                
                # Seleccionar un fragmento que no exceda el límite de tokens
                selected_content = select_random_fragment(full_content, max_tokens_per_fragment)
                
                fragments_content.append({
                    'path': str(fragment_path),
                    'content': selected_content,
                    'is_truncated': len(full_content) != len(selected_content)
                })
        except Exception as e:
            print(f"Error leyendo {fragment_path}: {e}")
            continue
    
    return fragments_content

def get_qdrant_questions(collection_name):
    """Obtiene las preguntas almacenadas en la colección Qdrant especificada."""
    client = get_configured_qdrant_client()
    
    # Verificar que la colección existe
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    if collection_name not in collection_names:
        raise ValueError(f"La colección '{collection_name}' no existe. Colecciones disponibles: {collection_names}")
    
    # Obtener estadísticas de la colección
    collection_info = client.get_collection(collection_name=collection_name)
    total_vectors = collection_info.points_count
    
    questions = []
    question_ids = []
    
    print(f"Recuperando {total_vectors} preguntas de Qdrant ({collection_name})...")
    
    # Usar scroll para recuperar todos los puntos
    batch_size = 1000
    next_page_offset = None
    
    with tqdm(total=total_vectors, desc="Scrolling points") as pbar:
        while True:
            results, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                break
                
            for record in results:
                if 'question' in record.payload:
                    questions.append(record.payload['question'])
                    question_ids.append(record.id)
                pbar.update(1)
                
            if next_page_offset is None:
                break
    
    print(f"Recuperadas {len(questions)} preguntas de Qdrant")
    return questions, question_ids

def generate_and_analyze_questions(num_iterations=1000, collection_name=None, random_collection=None):
    """
    Compara preguntas de la colección random con las almacenadas en la colección base
    usando la búsqueda vectorial de Qdrant.
    """
    # Usar las colecciones específicas o las predeterminadas
    base_collection = collection_name or COLLECTIONS["base"]
    random_collection = random_collection or COLLECTIONS["random"]
    
    client = get_configured_qdrant_client()
    
    # Cargar solo las preguntas random
    print("\nCargando preguntas random de Qdrant...")
    random_questions, random_ids = get_qdrant_questions(random_collection)
    
    # Limitar a num_iterations
    if len(random_questions) > num_iterations:
        print(f"Seleccionando {num_iterations} preguntas aleatorias de {len(random_questions)} disponibles...")
        indices = random.sample(range(len(random_questions)), num_iterations)
        selected_random_questions = [random_questions[i] for i in indices]
        selected_random_ids = [random_ids[i] for i in indices]
    else:
        print(f"Usando todas las {len(random_questions)} preguntas random disponibles...")
        selected_random_questions = random_questions
        selected_random_ids = random_ids
    
    resultados_detallados = []
    
    print("\nAnalizando similitud entre preguntas random y base...")
    for i, (random_question, random_id) in enumerate(tqdm(zip(selected_random_questions, selected_random_ids), total=len(selected_random_questions), desc="Procesando preguntas")):
        print(f"\nProcesando pregunta random {i+1}/{len(selected_random_questions)}: {random_question}")
        
        # Obtener embedding para la pregunta random
        embedding = get_embedding_from_ollama(random_question)
        
        if not embedding:
            print(f"Error al generar embedding para: {random_question}")
            continue
        
        # Usar Qdrant para encontrar la pregunta más similar directamente
        search_result = client.search(
            collection_name=base_collection,
            query_vector=embedding,
            limit=1,  # Solo queremos la más similar
            with_payload=True
        )
        
        if not search_result:
            print(f"No se encontraron resultados para la pregunta: {random_question}")
            continue
        
        # Obtener la pregunta más similar y su puntuación
        most_similar_result = search_result[0]
        most_similar_question = most_similar_result.payload.get('question', 'Pregunta no encontrada')
        highest_similarity = most_similar_result.score
        
        # Calcular BERTScore
        try:
            precision, recall, f1 = calculate_bert_score([random_question], [most_similar_question])
            bert_score = float(f1[0])
        except Exception as e:
            print(f"Error al calcular BERTScore: {e}")
            bert_score = 0.0
        
        # Registrar resultado
        resultado = {
            "pregunta_random_id": random_id,
            "pregunta_random": random_question,
            "pregunta_similar": most_similar_question,
            "similitud_coseno": highest_similarity,
            "bert_score": bert_score
        }
        
        resultados_detallados.append(resultado)
        print(f"Similitud coseno: {highest_similarity:.4f}, BERTScore: {bert_score:.4f}")
        print(f"Pregunta más similar: {most_similar_question}")
    
    # Calcular estadísticas globales
    if resultados_detallados:
        similitudes_coseno = [r['similitud_coseno'] for r in resultados_detallados]
        bert_scores = [r['bert_score'] for r in resultados_detallados]
        
        estadisticas = {
            "total_comparaciones": len(resultados_detallados),
            "similitud_coseno_media": float(np.mean(similitudes_coseno)),
            "similitud_coseno_mediana": float(np.median(similitudes_coseno)),
            "similitud_coseno_max": float(np.max(similitudes_coseno)),
            "similitud_coseno_min": float(np.min(similitudes_coseno)),
            "bert_score_medio": float(np.mean(bert_scores)),
            "bert_score_mediana": float(np.median(bert_scores)),
        }
        
        # Guardar resultados
        resultado_final = {
            "estadisticas": estadisticas,
            "resultados": resultados_detallados
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
        
        # Guardar resultados
        output_path = os.path.join(project_root, "output", "analisis_preguntas_random_vs_base.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resultado_final, f, ensure_ascii=False, indent=2)
        
        print(f"\nResultados guardados en: {output_path}")
        
        # Imprimir estadísticas
        print("\n=== Estadísticas de Similitud ===")
        print(f"Total de comparaciones: {estadisticas['total_comparaciones']}")
        print(f"Similitud Coseno Media: {estadisticas['similitud_coseno_media']:.4f}")
        print(f"BERTScore Medio: {estadisticas['bert_score_medio']:.4f}")
        
        return resultado_final
    
    print("No se pudieron realizar comparaciones.")
    return None

if __name__ == "__main__":
    print("Comenzando análisis de preguntas random vs. base")
    generate_and_analyze_questions(num_iterations=300)
