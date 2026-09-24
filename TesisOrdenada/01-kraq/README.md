# 01-kraq — Capítulo 3: Knowledge-Retrieval-Augmented Questions

Genera preguntas sintéticas a partir de las comunidades de un grafo de conocimiento
construido con GraphRAG, y las indexa en Qdrant para que los métodos de `../02-rag/` las
usen en recuperación.

## Contenido

```
src/
├── QuestionGeneration/     generación de preguntas por comunidad
│   ├── generate_community_questions.py        fθ = LLaMA-3.1-8B + QLoRA  ← la propuesta
│   ├── generate_community_questions_base.py   LLaMA-3.1-8B sin fine-tune ← ablation
│   ├── generate_community_questions_nofind.py       idem, sin usar los findings del reporte
│   ├── generate_community_questions_base_nofind.py
│   ├── generate_naive_questions.py            generación directa, sin comunidades
│   ├── llm_inference.py       carga el adapter QLoRA con PEFT en 4-bit
│   └── llm_inference_base.py  carga el modelo base sin adapter
├── Benchmarks/             evaluación de la calidad de las preguntas
│   ├── calculate_questions_similarity_{finetuned,base,random}.py  BERTScore
│   ├── generate_random_fragment_questions_{base,finetuned}.py     baseline sin grafo
│   ├── entity_overlap_{finetuned,base,random}.py   solapamiento de entidades
│   ├── analyze_relevance_threshold.py
│   └── count_questions.py
├── DataBase/               carga de preguntas en Qdrant
│   ├── qdrant_config.py                    conexión y nombres de colección
│   ├── load_questions_to_qdrant.py         → questions-index-finetuned
│   ├── load_questions_to_qdrant_base.py    → questions-index-base
│   ├── load_questions_to_qdrant_random.py  → questions-index-random
│   ├── qdrant_stats.py / clear_qdrant_collection.py
├── Analysis/               lectura de los parquets y de LanceDB que produce GraphRAG
└── utils/
    ├── bert_score_utils.py          ⚠️ ver docs/06 §2.1 (lang="es")
    └── test_parallel_inference.py

configs/<dataset>/    settings.yaml de GraphRAG + los 13 prompts + Descripcion.txt
preprocesamiento/     dataset crudo → corpus de texto + set de preguntas de evaluación
_overrides/<dataset>/ los 9 archivos que cambian según el dataset
```

## El sufijo del archivo dice qué variante es

| Sufijo | Generador | Colección Qdrant | Rol en la tesis |
|---|---|---|---|
| `_finetuned` (o sin sufijo) | fθ: LLaMA-3.1-8B + adapter QLoRA | `questions-index-finetuned` | la propuesta |
| `_base` | LLaMA-3.1-8B Instruct, sin fine-tune | `questions-index-base` | ablation "instruct" |
| `_random` | fragmentos al azar, sin grafo | `questions-index-random` | ablation "random" |

`_nofind` = variante que ignora los *findings* del community report y usa sólo el resumen.

## Orden de ejecución

1. `preprocesamiento/process_*.py` → corpus + preguntas de evaluación
2. `graphrag index --root <dataset>` con `configs/<dataset>/settings.yaml`
3. `src/QuestionGeneration/generate_community_questions.py` (guarda checkpoints por nivel)
4. `src/DataBase/load_questions_to_qdrant.py`
5. `src/Benchmarks/calculate_questions_similarity_finetuned.py` para las métricas del Cap. 3

Detalle completo en `../docs/02-como-reproducir.md`.

## Notas

- El adapter de fθ está en `../modelos/llama3-8b-qlora-finetuned`. `llm_inference.py` lo
  busca en la raíz del proyecto; se puede apuntar con `PEFT_MODEL_PATH`.
- Los `prompts/` de cada dataset son los **defaults de GraphRAG** (modalidad
  *NoPromptTuning*). Son idénticos entre los 4 datasets.
- El corpus que indexó GraphRAG está en `../04-datos/corpus/<dataset>/graphrag-input/`;
  copiarlo (o symlinkearlo) como `input/` junto al `settings.yaml` del dataset.
- En `logs-indexado/<dataset>/` están los logs reales de la corrida de indexado
  (`indexing-engine.log` + `logs.json`) de bioasq, hotpot y pubhealth — sirven para
  confirmar versión de graphrag, tiempos y errores. El zip de triviaqa no traía logs.
