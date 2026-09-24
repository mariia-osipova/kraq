# modelos/

## `llama3-8b-qlora-finetuned/`

El adapter **QLoRA de fθ**, el generador de preguntas KRAQ del Capítulo 3 (§3.2.5).

- **Modelo base:** `meta-llama/Llama-3.1-8B-Instruct`, cargado en 4 bits.
- **Formato:** adapter PEFT (`adapter_model.safetensors`, 109 MB + tokenizer).
- **Se carga desde:** `../01-kraq/src/QuestionGeneration/llm_inference.py`, que lo busca en
  la raíz del proyecto o en `$PEFT_MODEL_PATH`.

```python
config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, ...)
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
```

### Procedencia

Los 4 zips de KRAQ (`Local_{BioASQ,hotpot,Pubhealth,TriviaQA}_*.zip`) traían cada uno una
copia. Se verificó por CRC que las cuatro son **byte-idénticas** (`908f92dd`) y se conservó
una sola. Los `checkpoint-*/` intermedios (incluido `checkpoint-7500`) quedaron en los zips.

### ⚠️ El código de entrenamiento no está

Está el adapter, están `adapter_config.json` y `training_args.bin` (que documentan la
configuración), pero **no el script que lo entrenó** ni los datos de Dolly-v2 / MusiQue
procesados. El adapter es reutilizable; fθ no es re-entrenable desde este repo.
Ver `../docs/06-riesgos-y-faltantes.md` §1.3.

Como registro del entrenamiento están los `trainer_state.json` de los dos checkpoints
(extraídos de `Local_BioASQ_*.zip`):

- `trainer_state_checkpoint-7500.json` — el **mejor** checkpoint (`best_metric` = eval loss
  0.7338), que es el adapter publicado acá.
- `trainer_state_checkpoint-7899.json` — el último paso (época 3.0).

Contienen la curva completa de loss/eval_loss cada 10 pasos y el schedule de learning
rate efectivo (decaimiento lineal desde 2e-4), paso a paso.
