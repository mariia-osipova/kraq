import os
import sys
import json
from tqdm import tqdm
import time
import datetime
import signal
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer

# Añadir src al path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar la función de generación del modelo base
from src.QuestionGeneration.llm_inference_base import generate_batch
# Inicializar el tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Variable para controlar interrupciones
interrupted = False

def signal_handler(sig, frame):
    """Maneja las señales de interrupción (Ctrl+C)"""
    global interrupted
    print("\nDetectada solicitud de interrupción. Se guardará el progreso en el próximo checkpoint...")
    interrupted = True

# Registrar el manejador de señales
signal.signal(signal.SIGINT, signal_handler)

# Función para convertir tipos NumPy a tipos Python nativos
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

def save_progress(questions_list: List[Dict], level: int, is_checkpoint=False):
    """Guarda el progreso actual en un archivo temporal."""
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear directorio de checkpoints si es necesario
    if is_checkpoint:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Mantener solo los 3 checkpoints más recientes para cada nivel
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith(f"community_questions_base_level_{level}_checkpoint_") and f.endswith(".json")]
        
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
        temp_path = os.path.join(checkpoint_dir, f"community_questions_base_level_{level}_checkpoint_{timestamp}.json")
    else:
        temp_path = os.path.join(output_dir, f"temp_questions_base_level_{level}.json")
    
    try:
        # Convertir tipos NumPy a tipos Python nativos
        safe_questions_list = convert_numpy_types(questions_list)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(safe_questions_list, f, ensure_ascii=False, indent=2)
        
        if is_checkpoint:
            print(f"\n✓ Checkpoint guardado: {len(questions_list)} comunidades en {temp_path}")
        else:
            print(f"\nProgreso guardado para nivel {level}: {len(questions_list)} comunidades")
        
        return temp_path
    except Exception as e:
        print(f"\n⚠️ Error al guardar progreso: {e}")
        print("Intentando salvar las preguntas ya generadas...")
        try:
            # Intento de recuperación: guardar solo lo que se pueda serializar
            recoverable_list = []
            for question in questions_list:
                try:
                    # Verificar si este elemento se puede serializar
                    json.dumps(convert_numpy_types(question))
                    recoverable_list.append(question)
                except:
                    print(f"Omitiendo pregunta no serializable...")
            
            recovery_path = os.path.join(output_dir, f"recovery_questions_base_level_{level}_{timestamp}.json")
            with open(recovery_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(recoverable_list), f, ensure_ascii=False, indent=2)
            
            print(f"✓ Recuperación guardada: {len(recoverable_list)} comunidades en {recovery_path}")
            return recovery_path
        except Exception as recovery_error:
            print(f"Error en la recuperación: {recovery_error}")
            return None

def load_checkpoint(level: int):
    """Carga el checkpoint más reciente para el nivel especificado."""
    checkpoint_dir = os.path.join(project_root, "output", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return []
    
    # Buscar todos los checkpoints para este nivel
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith(f"community_questions_base_level_{level}_checkpoint_") and f.endswith(".json")]
    
    if not checkpoint_files:
        return []
    
    # Ordenar por fecha (formato YYYYMMdd_HHMMSS en el nombre del archivo)
    checkpoint_files.sort(reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    
    try:
        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
            questions_list = json.load(f)
        print(f"Cargadas {len(questions_list)} comunidades desde checkpoint: {latest_checkpoint}")
        return questions_list
    except Exception as e:
        print(f"Error cargando checkpoint: {e}")
        return []

def process_llm_response(response: str) -> Optional[str]:
    """Procesa la respuesta del LLM para extraer la pregunta."""
    try:
        # Simplemente devolver la respuesta limpia
        return response.strip()
    except Exception as e:
        print(f"Error procesando respuesta del LLM: {e}")
        print(f"Respuesta problemática: {response[:200]}...")
        return None

def generate_questions_for_level(target_level: int, max_main_questions: int = None, max_finding_questions: int = None,
                               checkpoint_interval: int = 50, resume_from_checkpoint: bool = True) -> List[Dict]:
    global interrupted
    interrupted = False
    start_time = time.time()
    total_questions_generated = 0
    
    # Intentar cargar desde checkpoint si se solicita
    questions_list = []
    if resume_from_checkpoint:
        questions_list = load_checkpoint(target_level)
    
    parquet_path = os.path.join(project_root, "output", "community_reports.parquet")
    print(f"Leyendo datos desde: {parquet_path}")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el archivo: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Definir los prompts
    main_prompt = """Given this summary of a document collection, generate a natural question that a person might ask when looking for this information. The question should be:
    - Simple and straightforward
    - Written in conversational language
    - Focused on the main topic or event
    - Something a real person would ask when searching for information

Now generate a question for this summary:

{summary}
"""

    finding_prompt = """Given this summary of a document collection, generate a natural question that a person might ask when looking for this information. The question should be:
    - Simple and straightforward
    - Written in conversational language
    - Focused on the main topic or event
    - Something a real person would ask when searching for information

Now generate a question for this summary:

Finding:
{finding}
"""
    
    print(f"\n{'='*80}")
    print(f"Procesando nivel {target_level}...")
    df_nivel = df[df['level'] == target_level].copy()
    
    if len(df_nivel) == 0:
        raise ValueError(f"No se encontraron datos para el nivel {target_level}")
    
    # Guardar los índices originales antes de resetear
    df_nivel['original_index'] = df_nivel.index
    
    # Usar todos los datos si max_main_questions es None
    df_main = df_nivel if max_main_questions is None else df_nivel.head(max_main_questions)
    df_main = df_main.reset_index(drop=True)
    
    if not questions_list:
        questions_list = []
    
    # Estadísticas
    level_summaries = len(df_main)
    level_questions = sum(len(q["questions"]) for q in questions_list) if questions_list else 0
    level_main_questions = sum(1 for q in questions_list for question in q["questions"] if question["type"] == "main") if questions_list else 0
    level_finding_questions = sum(1 for q in questions_list for question in q["questions"] if question["type"] == "finding") if questions_list else 0
    
    # Identificar comunidades ya procesadas
    processed_community_ids = [q["community_id"] for q in questions_list]
    
    print(f"Procesando {level_summaries} resúmenes principales para nivel {target_level}")
    if processed_community_ids:
        print(f"Se encontraron {len(processed_community_ids)} comunidades ya procesadas en checkpoints anteriores")
    
    # Procesar preguntas principales (solo las que faltan)
    main_messages_list = []
    community_ids = []
    summaries = []
    
    for _, row in df_main.iterrows():
        if str(row['original_index']) not in processed_community_ids and pd.notna(row['summary']):
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates natural, conversational questions about documents."},
                {"role": "user", "content": f"""Given this summary of a document collection, generate a natural question that a person might ask when looking for this information. The question should be:
- Simple and straightforward
- Written in conversational language
- Focused on the main topic or event
- Something a real person would ask when searching for information

Summary:
{row['summary']}

Generate only ONE question."""}
            ]
            
            main_messages_list.append(messages)
            community_ids.append(str(row['original_index']))
            summaries.append(str(row['summary']))
    
    # Generar preguntas principales
    if main_messages_list:
        print(f"\nGenerando {len(main_messages_list)} preguntas principales nuevas...")
        
        # Procesar en lotes más pequeños para permitir checkpoints
        batch_size = min(10, checkpoint_interval)
        batch_count = 0
        
        for i in range(0, len(main_messages_list), batch_size):
            if interrupted:
                break
                
            batch_count += 1
            batch_messages = main_messages_list[i:i+batch_size]
            batch_community_ids = community_ids[i:i+batch_size]
            batch_summaries = summaries[i:i+batch_size]
            
            print(f"\nProcesando lote de preguntas principales {i//batch_size + 1}/{(len(main_messages_list)-1)//batch_size + 1}...")
            
            try:
                # Convertir mensajes a prompts usando apply_chat_template
                main_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in batch_messages]
                main_responses = generate_batch(main_prompts, batch_size=2)
                
                for community_id, summary, response in zip(batch_community_ids, batch_summaries, main_responses):
                    try:
                        questions_for_community = {
                            "community_id": community_id,
                            "level": int(target_level),  # Asegurar que sea int Python
                            "summary": summary,
                            "questions": []
                        }
                        
                        if response:
                            questions_for_community["questions"].append({
                                "type": "main",
                                "question": response.strip()
                            })
                            level_main_questions += 1
                            level_questions += 1
                        
                        if questions_for_community["questions"]:
                            questions_list.append(questions_for_community)
                    except Exception as item_error:
                        print(f"Error procesando pregunta principal para comunidad {community_id}: {item_error}")
                        print("Continuando con la siguiente pregunta...")
                        continue
                
                # Guardar checkpoint cada cierto número de lotes o si se acumulan suficientes preguntas nuevas
                if batch_count % 3 == 0 or level_questions % checkpoint_interval < batch_size:
                    save_progress(questions_list, target_level, is_checkpoint=True)
                    
            except Exception as batch_error:
                print(f"Error procesando lote de preguntas principales: {batch_error}")
                print("Guardando progreso e intentando continuar con el siguiente lote...")
                save_progress(questions_list, target_level, is_checkpoint=True)
            
            if interrupted:
                print("Proceso interrumpido. Finalizando después de guardar checkpoint...")
                save_progress(questions_list, target_level, is_checkpoint=True)
                break
    
    # Si fue interrumpido durante las preguntas principales, retornar
    if interrupted:
        return questions_list
    
    # Procesar preguntas de hallazgos (sin límite si max_finding_questions es None)
    print(f"\nBuscando hallazgos para generar preguntas...")
    
    # Agrupar los hallazgos ya procesados por comunidad
    processed_findings = {}
    for q in questions_list:
        community_id = q["community_id"]
        finding_explanations = [question["finding_explanation"] for question in q["questions"] 
                              if "type" in question and question["type"] == "finding" and "finding_explanation" in question]
        processed_findings[community_id] = finding_explanations
    
    finding_messages_list = []
    finding_explanations = []
    finding_community_ids = []
    
    try:
        # Recolectar todos los hallazgos disponibles que aún no han sido procesados
        for _, row in df_nivel.iterrows():
            community_id = str(row['original_index'])
            # Obtener los hallazgos ya procesados para esta comunidad
            community_processed_findings = processed_findings.get(community_id, [])
            
            # Verificar si 'findings' está en las columnas y no es None/NaN
            if 'findings' in row.index and row['findings'] is not None:
                findings = row['findings']
                if isinstance(findings, np.ndarray):
                    findings = findings.tolist()
                
                if findings:
                    for finding in findings:
                        if isinstance(finding, dict) and 'explanation' in finding and pd.notna(finding['explanation']):
                            # Verificar si este hallazgo ya fue procesado
                            if str(finding['explanation']) not in community_processed_findings:
                                messages = [
                                    {"role": "system", "content": "You are a helpful assistant that generates natural, conversational questions about specific findings or details in documents."},
                                    {"role": "user", "content": f"""Generate a natural follow-up question based on this specific finding. The question should be:
- Simple and direct
- Related to the specific detail or subtopic
- Written as if someone is trying to learn more about this specific aspect

Finding:
{finding['explanation']}

Generate only ONE question."""}
                                ]
                                
                                finding_messages_list.append(messages)
                                finding_explanations.append(str(finding['explanation']))
                                finding_community_ids.append(community_id)
                                
                                # Solo limitar si max_finding_questions está definido
                                if max_finding_questions is not None and len(finding_messages_list) >= max_finding_questions:
                                    break
                    
                    # Solo limitar si max_finding_questions está definido
                    if max_finding_questions is not None and len(finding_messages_list) >= max_finding_questions:
                        break
    except Exception as find_error:
        print(f"Error recopilando hallazgos: {find_error}")
        print("Continuando con los hallazgos que ya se recopilaron...")
    
    # Generar preguntas de hallazgos en lotes para permitir checkpoints
    if finding_messages_list:
        print(f"\nGenerando {len(finding_messages_list)} preguntas de hallazgos nuevas...")
        
        # Procesar en lotes para permitir checkpoints
        batch_size = min(10, checkpoint_interval)  # Usar lotes más pequeños para checkpoints más frecuentes
        
        for i in range(0, len(finding_messages_list), batch_size):
            if interrupted:
                break
                
            batch_messages = finding_messages_list[i:i+batch_size]
            batch_explanations = finding_explanations[i:i+batch_size]
            batch_community_ids = finding_community_ids[i:i+batch_size]
            
            print(f"\nProcesando lote de hallazgos {i//batch_size + 1}/{(len(finding_messages_list)-1)//batch_size + 1}...")
            
            try:
                # Convertir mensajes a prompts usando apply_chat_template
                finding_prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
                                  for messages in batch_messages]
                finding_responses = generate_batch(finding_prompts, batch_size=2)
                
                for community_id, explanation, response in zip(batch_community_ids, batch_explanations, finding_responses):
                    try:
                        community_entry = next((q for q in questions_list if q["community_id"] == community_id), None)
                        
                        if community_entry is None:
                            community_entry = {
                                "community_id": community_id,
                                "level": int(target_level),  # Asegurar que sea int Python
                                "summary": str(df_nivel[df_nivel['original_index'] == int(community_id)]['summary'].iloc[0]),
                                "questions": []
                            }
                            questions_list.append(community_entry)
                        
                        finding_question = process_llm_response(response)
                        if finding_question:
                            community_entry["questions"].append({
                                "type": "finding",
                                "finding_explanation": explanation,
                                "question": finding_question
                            })
                            level_finding_questions += 1
                            level_questions += 1
                    except Exception as item_error:
                        print(f"Error procesando pregunta de hallazgo: {item_error}")
                        print("Continuando con la siguiente pregunta...")
                        continue
                
                # Guardar checkpoint después de cada lote
                save_progress(questions_list, target_level, is_checkpoint=True)
                
            except Exception as batch_error:
                print(f"Error procesando lote de hallazgos: {batch_error}")
                print("Guardando progreso e intentando continuar con el siguiente lote...")
                save_progress(questions_list, target_level, is_checkpoint=True)
            
            if interrupted:
                print("Proceso interrumpido. Finalizando después de guardar checkpoint...")
                save_progress(questions_list, target_level, is_checkpoint=True)
                break
    
    # Imprimir estadísticas
    print(f"\nEstadísticas del nivel {target_level}:")
    print(f"- Resúmenes procesados: {level_summaries}")
    print(f"- Total preguntas generadas: {level_questions}")
    print(f"  - Preguntas principales: {level_main_questions}")
    print(f"  - Preguntas de hallazgos: {level_finding_questions}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Estadísticas de Tiempo:")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    if level_questions > 0:
        print(f"Tiempo promedio por pregunta: {total_time/level_questions:.2f} segundos")
    print(f"{'='*80}\n")
    
    return questions_list

def save_questions_for_level(target_level: int, max_main_questions: int = None, max_finding_questions: int = None,
                           checkpoint_interval: int = 50, resume_from_checkpoint: bool = True):
    start_time = time.time()
    mensaje_limites = "Iniciando generación de TODAS las preguntas posibles para nivel"
    if max_main_questions is not None and max_finding_questions is not None:
        mensaje_limites = f"Iniciando generación de preguntas para nivel {target_level} ({max_main_questions} principales y {max_finding_questions} de hallazgos)..."
    else:
        mensaje_limites = f"Iniciando generación de TODAS las preguntas posibles para nivel {target_level}..."
    
    if resume_from_checkpoint:
        mensaje_limites += " (reanudando desde checkpoint si existe)"
    
    print(mensaje_limites)
    
    try:
        questions = generate_questions_for_level(
            target_level, 
            max_main_questions, 
            max_finding_questions, 
            checkpoint_interval=checkpoint_interval,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convertir tipos NumPy a tipos Python nativos antes de guardar
        safe_questions = convert_numpy_types(questions)
        
        output_path = os.path.join(output_dir, f"community_questions_base_level_{target_level}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(safe_questions, f, ensure_ascii=False, indent=2)
        print(f"✓ Nivel {target_level}: {len(questions)} comunidades guardadas en {output_path}")
        
        total_time = time.time() - start_time
        print(f"\nTiempo total del proceso completo: {total_time:.2f} segundos")
        
        return True, questions
    except Exception as e:
        print(f"\n⚠️ Error durante la generación de preguntas para nivel {target_level}: {e}")
        print("Se intentará continuar con el siguiente nivel.")
        return False, []

if __name__ == "__main__":
    # Procesar todos los niveles disponibles secuencialmente
    try:
        # Primero, cargar el dataset para identificar los niveles disponibles
        parquet_path = os.path.join(project_root, "output", "community_reports.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"No se encontró el archivo: {parquet_path}")
        
        df = pd.read_parquet(parquet_path)
        available_levels = sorted(df['level'].unique())
        
        print(f"Se encontraron {len(available_levels)} niveles para procesar: {available_levels}")
        
        for level in available_levels:
            print(f"\n{'='*80}")
            print(f"PROCESANDO NIVEL {level}")
            print(f"{'='*80}\n")
            
            # Usar None para generar todas las preguntas posibles
            # Con checkpoints cada 50 preguntas y recuperación automática
            success, _ = save_questions_for_level(
                target_level=level,
                checkpoint_interval=50,
                resume_from_checkpoint=True
            )
            
            if success:
                print(f"\n{'='*80}")
                print(f"COMPLETADO NIVEL {level}")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*80}")
                print(f"NIVEL {level} COMPLETADO PARCIALMENTE CON ERRORES")
                print(f"{'='*80}\n")
    
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError durante la generación: {e}")