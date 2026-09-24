import torch
import asyncio
import time
import os
import psutil
import GPUtil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURACIÓN ---
model_name = "meta-llama/Llama-3.1-8B-Instruct"
cuda_device = "cuda:0"
compute_dtype = torch.float16
num_parallel_calls = 10  # Cambiá este número para testear distintos niveles de concurrencia
max_new_tokens = 100
prompt = "Explícame qué es la inteligencia artificial en lenguaje sencillo."

# --- CARGA DEL MODELO Y TOKENIZER ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": cuda_device},
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)
model.eval()

# --- FUNCIÓN DE INFERENCIA ---
def run_inference(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(cuda_device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=compute_dtype):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# --- ENVOLTORIO PARA USAR EN ASYNCIO ---
async def async_infer(executor, prompt_text):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, run_inference, prompt_text)

# --- MAIN ---
async def main():
    print(f"Test de {num_parallel_calls} inferencias en paralelo con GPU {cuda_device}")
    start = time.time()

    with ThreadPoolExecutor(max_workers=num_parallel_calls) as executor:
        tasks = [async_infer(executor, prompt) for _ in range(num_parallel_calls)]
        results = await asyncio.gather(*tasks)

    end = time.time()
    duration = end - start

    for i, r in enumerate(results):
        print(f"\n--- Resultado #{i+1} ---\n{r[:300]}...\n")

    print(f"\nTiempo total: {duration:.2f} segundos")

    # Uso de GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.id == 0:
            print(f"\nUso de GPU (ID {gpu.id}):")
            print(f"  Memoria usada: {gpu.memoryUsed:.2f} MB")
            print(f"  Memoria total: {gpu.memoryTotal:.2f} MB")
            print(f"  Carga de GPU: {gpu.load*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
