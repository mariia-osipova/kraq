import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
from typing import List
import bitsandbytes as bnb

# --- CONFIGURACIÓN ---
# Construir ruta relativa al modelo LoRA desde este script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
DEFAULT_PEFT_MODEL_PATH = os.path.join(project_root, "llama3-8b-qlora-finetuned")

# Parámetros configurables (pueden sobrescribirse con variables de entorno)
PEFT_MODEL_PATH = os.getenv("PEFT_MODEL_PATH", DEFAULT_PEFT_MODEL_PATH)
MAX_NEW_TOKENS = 25  # Adjusted to limit the number of new tokens generated
GENERATION_BATCH_SIZE = int(os.getenv("GENERATION_BATCH_SIZE", 2)) # Tamaño del lote para inferencia

# Configuración específica de GPU y cuantización
CUDA_DEVICE = "cuda:1"  # Forzar uso de GPU1
COMPUTE_DTYPE = torch.float16
device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Configuración de cuantización 4-bit
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Usar nf4 para mejor precisión
)

# --- CARGA DEL MODELO Y TOKENIZER (Singleton Pattern Básico) ---
model = None
tokenizer = None

def _load_model_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        print("Modelo y tokenizer ya cargados.")
        return model, tokenizer

    print(f"Cargando modelo LoRA desde: {PEFT_MODEL_PATH}")
    if not os.path.isdir(PEFT_MODEL_PATH):
        raise FileNotFoundError(f"Directorio del modelo LoRA no encontrado en: {PEFT_MODEL_PATH}")

    print("Cargando configuración PEFT...")
    try:
        config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
    except Exception as e:
        print(f"Error cargando PeftConfig desde {PEFT_MODEL_PATH}: {e}")
        print("Asegúrate de que adapter_config.json exista y sea válido.")
        raise

    print(f"Cargando modelo base ({config.base_model_name_or_path})...")
    # Cargar modelo base con cuantización 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=quantization_config,
        device_map={"": CUDA_DEVICE},  # Forzar todo al CUDA_DEVICE especificado
        torch_dtype=COMPUTE_DTYPE,
        trust_remote_code=True
    )

    print("Cargando pesos LoRA...")
    model = PeftModel.from_pretrained(
        base_model, 
        PEFT_MODEL_PATH,
        torch_dtype=COMPUTE_DTYPE,
        device_map={"": CUDA_DEVICE}  # Asegurar que los adaptadores también van a GPU1
    )
    model.eval()

    # Verificar que el modelo está en la GPU correcta
    current_device = next(model.parameters()).device
    print(f"Modelo cargado en dispositivo: {current_device}")
    if str(current_device) != CUDA_DEVICE:
        print(f"Warning: El modelo está en {current_device} en lugar de {CUDA_DEVICE}")

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    # Asegurarse de que el pad token esté configurado (importante para batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id # Sincronizar con el modelo

    # Mostrar información de memoria GPU
    if torch.cuda.is_available():
        gpu_id = int(CUDA_DEVICE.split(':')[1])
        print(f"\nEstado de memoria GPU {gpu_id}:")
        print(f"Memoria total: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        print(f"Memoria reservada: {torch.cuda.memory_reserved(gpu_id) / 1e9:.2f} GB")
        print(f"Memoria en uso: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")

    print("Modelo y tokenizer cargados exitosamente.")
    return model, tokenizer

# --- FUNCIÓN DE GENERACIÓN EN BATCH ---
def generate_batch(prompts: List[str], batch_size: int = GENERATION_BATCH_SIZE) -> List[str]:
    """
    Genera respuestas para una lista de prompts usando el modelo cargado,
    procesando en lotes.
    """
    loaded_model, loaded_tokenizer = _load_model_and_tokenizer()
    if not loaded_model or not loaded_tokenizer:
        raise RuntimeError("El modelo o el tokenizer no se pudieron cargar.")

    all_outputs = []
    print(f"Generando respuestas para {len(prompts)} prompts en lotes de {batch_size}...")

    # Configuración de generación
    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=loaded_tokenizer.pad_token_id,
        eos_token_id=loaded_tokenizer.eos_token_id
    )

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        print(f"Procesando lote {i//batch_size + 1}...")

        try:
            inputs = loaded_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=loaded_model.config.max_position_embeddings - MAX_NEW_TOKENS
            ).to(device)  # Usar el device global

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=COMPUTE_DTYPE):  # Usar autocast para optimizar memoria
                outputs = loaded_model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            input_ids_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_ids_len:]
            batch_outputs = loaded_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Truncate each result at the first question mark
            truncated_outputs = [result.strip().split("?")[0] + "?" for result in batch_outputs]
            all_outputs.extend(truncated_outputs)

            # Liberar memoria CUDA después de cada lote
            del outputs, inputs
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"\nError: Memoria GPU insuficiente en lote {i//batch_size + 1}.")
            print("Intentando liberar memoria y reducir tamaño de lote...")
            torch.cuda.empty_cache()
            # Procesar este lote con tamaño reducido
            if batch_size > 1:
                print("Reintentando con batch_size=1...")
                for single_prompt in batch_prompts:
                    result = generate_batch([single_prompt], batch_size=1)
                    all_outputs.extend(result)
            else:
                print("No se puede reducir más el batch_size. Saltando prompt...")
                all_outputs.extend(["Error: Memoria insuficiente"] * len(batch_prompts))

        except Exception as e:
            print(f"\nError en lote {i//batch_size + 1}: {e}")
            all_outputs.extend([f"Error: {str(e)}"] * len(batch_prompts))

        # Mostrar uso de memoria después de cada lote
        if torch.cuda.is_available():
            gpu_id = int(CUDA_DEVICE.split(':')[1])
            print(f"Memoria GPU en uso: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")

    print("Generación completada.")
    return all_outputs

# --- EJEMPLO DE USO (para probar este script directamente) ---
if __name__ == "__main__":
    # Cargar modelo al ejecutar el script
    try:
        _load_model_and_tokenizer()
        print("\n--- Probando generación en batch ---")
        test_prompts = [
            "Formato JSON esperado:\n{\"preguntas\": [{\"tipo\": \"definicion\", \"pregunta\": \"¿Qué es X?\"}] }\n\nContexto: La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía.\nGenera una pregunta de definición sobre la fotosíntesis en formato JSON:",
            "Formato JSON esperado:\n{\"preguntas\": [{\"tipo\": \"comparacion\", \"pregunta\": \"¿Diferencia entre A y B?\"}] }\n\nContexto: Los perros son mamíferos domesticados. Los lobos son mamíferos salvajes relacionados con los perros.\nGenera una pregunta de comparación entre perros y lobos en formato JSON:",
            "Formato JSON esperado:\n{\"preguntas\": [{\"tipo\": \"causa_efecto\", \"pregunta\": \"¿Por qué ocurre Y cuando pasa X?\"}] }\n\nContexto: El calentamiento global provoca el derretimiento de los glaciares.\nGenera una pregunta de causa-efecto sobre el calentamiento global en formato JSON:",
        ]
        results = generate_batch(test_prompts, batch_size=2)

        for prompt, result in zip(test_prompts, results):
             print("-" * 40)
             # print(f"Prompt:\n{prompt}") # Descomentar si quieres ver el prompt completo
             print(f"Respuesta Generada:\n{result}")
             print("-" * 40)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Verifica la ruta del modelo LoRA en la variable PEFT_MODEL_PATH.")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Limpiar memoria GPU al finalizar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_id = int(CUDA_DEVICE.split(':')[1])
            print(f"\nMemoria GPU final en uso: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")
