import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
import os
from typing import List
import bitsandbytes as bnb

# --- CONFIGURACIÓN ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Configuración del modelo
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 250))
GENERATION_BATCH_SIZE = int(os.getenv("GENERATION_BATCH_SIZE", 2))

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
    bnb_4bit_quant_type="nf4",
)

# --- CARGA DEL MODELO Y TOKENIZER (Singleton Pattern) ---
model = None
tokenizer = None

def _load_model_and_tokenizer():
    global model, tokenizer
    if model is not None and tokenizer is not None:
        return model, tokenizer

    print(f"Cargando modelo base: {BASE_MODEL_ID}")
    
    print("Cargando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map={"": CUDA_DEVICE},
        torch_dtype=COMPUTE_DTYPE,
        trust_remote_code=True
    )
    model.eval()

    # Verificar dispositivo del modelo
    current_device = next(model.parameters()).device
    print(f"Modelo cargado en dispositivo: {current_device}")
    if str(current_device) != CUDA_DEVICE:
        print(f"Warning: El modelo está en {current_device} en lugar de {CUDA_DEVICE}")

    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Importante para la generación en batch
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Tokenizer cargado exitosamente.")
    print(f"Configurando tokenizer.padding_side = {tokenizer.padding_side}")
    print(f"Configurando model.config.pad_token_id = {model.config.pad_token_id}")

    # Mostrar información de memoria GPU
    if torch.cuda.is_available():
        gpu_id = int(CUDA_DEVICE.split(':')[1])
        print(f"\nEstado de memoria GPU {gpu_id}:")
        print(f"Memoria total: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
        print(f"Memoria reservada: {torch.cuda.memory_reserved(gpu_id) / 1e9:.2f} GB")
        print(f"Memoria en uso: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")

    return model, tokenizer

def generate_batch(prompts: List[str], batch_size: int = GENERATION_BATCH_SIZE) -> List[str]:
    """Genera respuestas para una lista de prompts usando el modelo base."""
    loaded_model, loaded_tokenizer = _load_model_and_tokenizer()
    if not loaded_model or not loaded_tokenizer:
        raise RuntimeError("El modelo o el tokenizer no se pudieron cargar.")

    all_outputs = []
    print(f"Generando respuestas para {len(prompts)} prompts en lotes de {batch_size}...")

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
            # Actualizamos el formato para Llama-3.1-Instruct
            inputs = loaded_tokenizer(
                batch_prompts,  # Prompts ya vienen con formato [INST]...[/INST]
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=loaded_model.config.max_position_embeddings - MAX_NEW_TOKENS
            ).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=COMPUTE_DTYPE):
                outputs = loaded_model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            input_ids_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_ids_len:]
            batch_outputs = loaded_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_outputs.extend(batch_outputs)

            # Liberar memoria CUDA
            del outputs, inputs
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"\nError: Memoria GPU insuficiente en lote {i//batch_size + 1}.")
            print("Intentando liberar memoria y reducir tamaño de lote...")
            torch.cuda.empty_cache()
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

        # Mostrar uso de memoria
        if torch.cuda.is_available():
            gpu_id = int(CUDA_DEVICE.split(':')[1])
            print(f"Memoria GPU en uso: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")

    return all_outputs

if __name__ == "__main__":
    # Test del script
    try:
        print("\n=== Iniciando prueba de generación con modelo base ===")
        test_prompts = [
            "Formato JSON esperado:\n{\"preguntas\": [{\"tipo\": \"definicion\", \"pregunta\": \"¿Qué es X?\"}] }\n\nContexto: La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía.\nGenera una pregunta de definición sobre la fotosíntesis en formato JSON:",
            "Formato JSON esperado:\n{\"preguntas\": [{\"tipo\": \"comparacion\", \"pregunta\": \"¿Diferencia entre A y B?\"}] }\n\nContexto: Los perros son mamíferos domesticados. Los lobos son mamíferos salvajes relacionados con los perros.\nGenera una pregunta de comparación entre perros y lobos en formato JSON:"
        ]

        results = generate_batch(test_prompts, batch_size=2)
        
        print("\n=== Resultados de la prueba ===")
        for i, (prompt, result) in enumerate(zip(test_prompts, results), 1):
            print(f"\nPrueba {i}:")
            print("-" * 40)
            print(f"Respuesta:\n{result}")
            print("-" * 40)

    except Exception as e:
        print(f"\nError durante la prueba: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_id = int(CUDA_DEVICE.split(':')[1])
            print(f"\nMemoria GPU final en uso: {torch.cuda.memory_allocated(gpu_id) / 1e9:.2f} GB")
