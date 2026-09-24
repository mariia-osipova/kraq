import os
import sys
import random
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from transformers import AutoTokenizer  # Importar solo el tokenizer
import time
import signal
import datetime

# Añadir src al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar la función de generación del modelo base
from src.QuestionGeneration.llm_inference_base import generate_batch, _load_model_and_tokenizer


# Almacenar el tokenizer como variable global para no cargarlo repetidamente
_tokenizer = None

# Variables para gestionar interrupciones
interrupted = False

def signal_handler(sig, frame):
    """Maneja las señales de interrupción (Ctrl+C)"""
    global interrupted
    print("\nDetectada solicitud de interrupción. Guardando progreso y finalizando...")
    interrupted = True

# Registrar el manejador de señales
signal.signal(signal.SIGINT, signal_handler)

def get_tokenizer():
    """
    Obtiene el tokenizer del modelo de llm_inference_base.py para reutilizarlo.
    """
    global _tokenizer
    if _tokenizer is None:
        # Obtener el tokenizer de la función _load_model_and_tokenizer()
        # que ya está funcionando correctamente
        print("Reusando tokenizer del modelo base...")
        _, _tokenizer = _load_model_and_tokenizer()
    return _tokenizer

def select_random_fragment(text: str, max_tokens: int = 300) -> str:
    """
    Selecciona un fragmento aleatorio del texto que no exceda el número máximo de tokens.
    
    Args:
        text: Texto completo del que seleccionar un fragmento
        max_tokens: Número máximo de tokens permitidos
    
    Returns:
        Un fragmento del texto que no excede max_tokens
    """
    try:
        # Obtener el tokenizer sin cargar el modelo
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
        
        # Limpiar el fragmento
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
    except Exception as e:
        print(f"Error seleccionando fragmento aleatorio: {e}")
        # En caso de error, devolver un fragmento del texto (los primeros 200 caracteres)
        return text[:min(200, len(text))]

def load_random_fragments(min_fragments=2, max_fragments=5, max_tokens_per_fragment=200) -> List[Dict]:
    """
    Carga fragmentos aleatorios de texto desde el directorio input,
    asegurando que cada fragmento no exceda el límite de tokens.
    """
    try:
        input_dir = Path(project_root) / "input"
        all_fragment_files = list(input_dir.glob("*.*"))  # Usar glob más general para capturar todos los archivos de texto
        
        # Filtrar para incluir solo archivos de texto
        text_files = [f for f in all_fragment_files if f.suffix.lower() in ('.txt', '.md')]
        
        if not text_files:
            raise ValueError(f"No se encontraron archivos de texto en {input_dir}")
        
        num_fragments = random.randint(min_fragments, max_fragments)
        
        if len(text_files) < num_fragments:
            # Si no hay suficientes archivos, permitir repeticiones
            selected_files = random.choices(text_files, k=num_fragments)
        else:
            selected_files = random.sample(text_files, num_fragments)
        
        fragments_content = []
        
        for file_path in selected_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_content = f.read().strip()
                    
                    # Seleccionar un fragmento que no exceda el límite de tokens
                    selected_content = select_random_fragment(full_content, max_tokens_per_fragment)
                    
                    fragments_content.append({
                        'path': str(file_path),
                        'content': selected_content,
                        'is_truncated': len(full_content) != len(selected_content)
                    })
                    
            except Exception as e:
                print(f"Error leyendo {file_path}: {e}")
                continue
        
        return fragments_content
    except Exception as e:
        print(f"Error cargando fragmentos aleatorios: {e}")
        # En caso de error, devolver al menos un fragmento con contenido de prueba
        return [{'path': 'error.txt', 'content': 'Fragmento de prueba debido a un error.', 'is_truncated': True}]

def generate_questions_from_fragments_batch(fragments_batch: List[List[Dict]]) -> List[str]:
    """Genera preguntas para un lote de fragmentos usando el modelo base."""
    try:
        tokenizer = get_tokenizer()
        prompts = []
        
        for fragments in fragments_batch:
            # Combinar los fragmentos, indicando si alguno fue truncado
            combined_content = "\n\n".join([
                f"{f['content']}" + (" [Fragmento truncado]" if f.get('is_truncated', False) else "")
                for f in fragments
            ])
            
            # Crear la estructura de mensajes con roles
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates natural, conversational questions about documents."},
                {"role": "user", "content": f"""Given these random fragments, generate a natural, concise question that someone might ask about the themes or topics present in these passages. The question should:
- Be short and to the point
- Focus on a common theme or interesting connection between the fragments
- Be something a real person would naturally ask
- Not be too complex or academic

Story fragments:
{combined_content}

Generate only ONE concise question:"""}
            ]
            
            # Aplicar chat template al mensaje
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        
        # Generar preguntas en lote
        responses = generate_batch(prompts, batch_size=2)
        
        # Procesar respuestas
        questions = []
        for i, response in enumerate(responses):
            try:
                # Limpiar y simplificar la respuesta
                cleaned_response = response.strip()
                
                # Si comienza con algo como "Question:" o "Pregunta:", eliminarlo
                if ":" in cleaned_response[:20]:
                    cleaned_response = cleaned_response.split(":", 1)[1].strip()
                    
                # Truncar la respuesta en el primer signo de interrogación
                if "?" in cleaned_response:
                    cleaned_response = cleaned_response.split("?")[0] + "?"
                
                questions.append(cleaned_response)
            except Exception as e:
                print(f"Error procesando respuesta {i}: {e}")
                questions.append(None)
        
        return questions
    except Exception as e:
        print(f"Error en la generación de preguntas de fragmentos: {e}")
        print(traceback.format_exc())
        # Devolver lista de None del mismo tamaño que el lote de entrada
        return [None] * len(fragments_batch)

def save_checkpoint(questions, checkpoint_dir=None):
    """
    Guarda las preguntas actuales en un archivo de checkpoint.
    Mantiene solo los 3 checkpoints más recientes.
    """
    try:
        if checkpoint_dir is None:
            checkpoint_dir = Path(project_root) / "output" / "checkpoints"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Mantener solo los 3 checkpoints más recientes
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith("naive_questions_checkpoint_") and f.endswith(".json")]
        
        if len(checkpoint_files) >= 3:
            # Ordenar por fecha (formato YYYYMMdd_HHMMSS en el nombre del archivo)
            checkpoint_files.sort()
            # Eliminar los más antiguos, dejando solo los 2 más recientes
            for old_file in checkpoint_files[:-2]:
                try:
                    os.remove(os.path.join(checkpoint_dir, old_file))
                    print(f"Eliminado checkpoint antiguo: {old_file}")
                except:
                    pass
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"naive_questions_checkpoint_{timestamp}.json"
        
        # Convertir tipos NumPy a tipos Python nativos
        safe_questions = convert_numpy_types(questions)
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(safe_questions, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Checkpoint guardado: {len(questions)} preguntas en {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"\n⚠️ Error al guardar checkpoint: {e}")
        print("Intentando salvar las preguntas ya generadas...")
        try:
            # Intento de recuperación: guardar solo lo que se pueda serializar
            recoverable_list = []
            for question in questions:
                try:
                    # Verificar si este elemento se puede serializar
                    json.dumps(convert_numpy_types(question))
                    recoverable_list.append(question)
                except:
                    print(f"Omitiendo pregunta no serializable...")
            
            recovery_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            recovery_path = checkpoint_dir / f"naive_questions_recovery_{recovery_timestamp}.json"
            with open(recovery_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(recoverable_list), f, ensure_ascii=False, indent=2)
            
            print(f"✓ Recuperación guardada: {len(recoverable_list)} preguntas en {recovery_path}")
            return recovery_path
        except Exception as recovery_error:
            print(f"Error en la recuperación: {recovery_error}")
            return None

def load_checkpoint(checkpoint_path=None):
    """
    Carga las preguntas desde un archivo de checkpoint.
    Si no se especifica la ruta, busca el checkpoint más reciente.
    """
    try:
        checkpoint_dir = Path(project_root) / "output" / "checkpoints"
        
        if checkpoint_path is None:
            # Buscar el checkpoint más reciente
            if not checkpoint_dir.exists():
                return []
            
            checkpoints = list(checkpoint_dir.glob("naive_questions_checkpoint_*.json"))
            if not checkpoints:
                return []
            
            # Ordenar por fecha de modificación (más reciente primero)
            checkpoint_path = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        
        if not os.path.exists(checkpoint_path):
            print(f"No se encontró el archivo de checkpoint: {checkpoint_path}")
            return []
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        print(f"Cargadas {len(questions)} preguntas desde checkpoint: {checkpoint_path}")
        return questions
    except Exception as e:
        print(f"Error cargando checkpoint: {e}")
        return []

def convert_numpy_types(obj):
    """Convierte tipos NumPy a tipos Python nativos para serializarlos en JSON."""
    if isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)  # Convertir enteros NumPy a enteros Python
    elif isinstance(obj, np.floating):
        return float(obj)  # Convertir flotantes NumPy a flotantes Python
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())  # Convertir arrays NumPy a listas Python
    else:
        return obj

def generate_naive_questions(num_questions=None, batch_size=5, checkpoint_interval=20, resume_from_checkpoint=True):
    """
    Genera preguntas aleatorias basadas en fragmentos de documentos aleatorios.
    Guarda checkpoints periódicamente para poder reanudar el proceso.
    
    Args:
        num_questions: Número de preguntas a generar. Si es None, generará tantas como sea posible.
        batch_size: Tamaño del lote para la generación de preguntas.
        checkpoint_interval: Cada cuántos lotes guardar un checkpoint.
        resume_from_checkpoint: Si debe intentar reanudar desde el último checkpoint.
    """
    global interrupted
    interrupted = False
    start_time = time.time()
    
    # Crear directorio de checkpoints
    checkpoint_dir = Path(project_root) / "output" / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Intentar cargar desde checkpoint si se solicita
    questions = []
    if resume_from_checkpoint:
        questions = load_checkpoint()
        
    # Determinar el número de preguntas a generar
    input_dir = Path(project_root) / "input"
    all_fragment_files = list(input_dir.glob("*.*"))
    text_files = [f for f in all_fragment_files if f.suffix.lower() in ('.txt', '.md')]
    
    if num_questions is None:
        max_files = len(text_files)
        if max_files > 0:
            # Generar hasta 5 preguntas por archivo disponible
            num_questions = max_files * 5
        else:
            num_questions = 36  # Valor por defecto si no hay archivos
    
    # Ajustar el número de preguntas restantes si se cargó desde checkpoint
    remaining_questions = num_questions - len(questions)
    if remaining_questions <= 0:
        print(f"Ya se han generado {len(questions)} preguntas. No se generarán más.")
        return questions
    
    print(f"\nGenerando {remaining_questions} preguntas aleatorias adicionales...")
    
    fragments_list = []
    error_count = 0
    max_errors = 10  # Máximo número de errores permitidos
    
    # Generar lotes de fragmentos
    for _ in range(remaining_questions):
        print(f"\rGenerando fragmentos aleatorios {_ + 1} de {remaining_questions}", end='', flush=True)
        try:
            fragments = load_random_fragments()
            fragments_list.append(fragments)
        except Exception as e:
            print(f"Error generando fragmentos: {e}")
            error_count += 1
            if error_count >= max_errors:
                print(f"Demasiados errores ({error_count}). Deteniendo generación.")
                break
    
    # Procesar en lotes para eficiencia
    batch_count = 0
    for i in range(0, len(fragments_list), batch_size):
        if interrupted:
            print("Proceso interrumpido. Guardando progreso antes de finalizar...")
            save_checkpoint(questions, checkpoint_dir)
            break
            
        batch_count += 1
        batch = fragments_list[i:i+batch_size]
        print(f"\nProcesando lote {i//batch_size + 1} de {len(fragments_list)//batch_size + 1}...")
        
        try:
            # Generar preguntas para este lote
            batch_questions = generate_questions_from_fragments_batch(batch)
            
            # Guardar solo la pregunta y metadatos básicos, sin incluir los fragmentos completos
            for j, q in enumerate(batch_questions):
                if q:  # Solo añadir si tenemos una pregunta válida
                    try:
                        # Contamos cuántos fragmentos se usaron pero no los guardamos completos
                        num_fragments_used = len(batch[j])
                        
                        question_obj = {
                            'question': q,
                            'num_fragments': num_fragments_used,  # Solo guardamos el número de fragmentos
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'method': 'naive'
                        }
                        questions.append(question_obj)
                    except Exception as item_error:
                        print(f"Error al procesar pregunta {j} del lote: {item_error}")
                        continue
            
            # Guardar checkpoint periódicamente
            if batch_count % checkpoint_interval == 0:
                save_checkpoint(questions, checkpoint_dir)
                print(f"Checkpoint periódico: {len(questions)}/{num_questions} preguntas generadas")
        
        except Exception as batch_error:
            print(f"Error procesando lote {batch_count}: {batch_error}")
            print(traceback.format_exc())
            # Intentar guardar progreso antes de continuar
            save_checkpoint(questions, checkpoint_dir)
            # Continuar con el siguiente lote
    
    # Calcular estadísticas
    total_time = time.time() - start_time
    questions_this_session = len(questions) - (len(load_checkpoint()) if resume_from_checkpoint else 0)
    avg_time_per_question = total_time / questions_this_session if questions_this_session > 0 else 0
    
    print(f"\n=== Estadísticas de Generación ===")
    print(f"Preguntas generadas en esta sesión: {questions_this_session}")
    print(f"Total de preguntas disponibles: {len(questions)}")
    print(f"Tiempo total de esta sesión: {total_time:.2f} segundos")
    print(f"Tiempo promedio por pregunta en esta sesión: {avg_time_per_question:.2f} segundos")
    
    # Guardar las preguntas generadas en el archivo final
    output_path = Path(project_root) / "output" / "naive_questions.json"
    
    try:
        # Convertir tipos NumPy a tipos Python nativos
        safe_questions = convert_numpy_types(questions)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(safe_questions, f, ensure_ascii=False, indent=2)
        
        print(f"\nPreguntas guardadas en {output_path}")
    except Exception as e:
        print(f"Error guardando archivo final: {e}")
        # Guardar en un archivo alternativo en caso de error
        alt_path = Path(project_root) / "output" / f"naive_questions_recovery_final.json"
        try:
            # Intentar guardar con un proceso de recuperación
            recoverable_list = []
            for question in questions:
                try:
                    # Verificar si este elemento se puede serializar
                    json.dumps(convert_numpy_types(question))
                    recoverable_list.append(question)
                except:
                    pass
            
            with open(alt_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(recoverable_list), f, ensure_ascii=False, indent=2)
            
            print(f"Recuperación guardada: {len(recoverable_list)} preguntas en {alt_path}")
        except:
            print("No se pudo guardar ni siquiera el archivo de recuperación.")
    
    return questions

if __name__ == "__main__":
    print("Iniciando generación de preguntas naive con modelo base...")
    try:
        start_time = time.time()
        # Utilizar None para generar todas las preguntas posibles
        # Guardar checkpoint cada 20 lotes (100 preguntas si batch_size=5)
        questions = generate_naive_questions(19100, batch_size=5, checkpoint_interval=4)
        total_time = time.time() - start_time
        print(f"\nGeneradas {len(questions)} preguntas naive exitosamente")
        print(f"Tiempo total del proceso completo: {total_time:.2f} segundos")
        if questions:
            print(f"Tiempo promedio por pregunta: {total_time/len(questions):.2f} segundos")
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError durante la generación: {e}")
        print(traceback.format_exc()) 