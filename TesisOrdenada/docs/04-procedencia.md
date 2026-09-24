# Procedencia: de dónde salió cada cosa

Todo el contenido de `TesisOrdenada/` viene de `../BackupTesis/`. Nada fue reescrito ni
regenerado; las únicas modificaciones (claves removidas, rutas absolutas) están listadas
en «Modificaciones aplicadas», más abajo. `../BackupTesis/` queda intacto como respaldo.

## Fuentes

| Origen en BackupTesis | Tamaño | Qué aportó | Destino |
|---|---|---|---|
| `Speculative_Rag_..._hotpot_solver.zip` | 765 MB | árbol **canónico** de `02-rag` (71 `.py`, el más completo: único con `solver/` y `ablation_combined/`) | `02-rag/` |
| `Speculative_Rag_..._triviaQA_solver.zip` | 713 MB | overrides + resultados | `02-rag/_overrides/triviaqa/`, `03-resultados/triviaqa/` |
| `Speculative_Rag_..._pubhealth_solver.zip` | 344 MB | overrides + resultados | `02-rag/_overrides/pubhealth/`, `03-resultados/pubhealth/` |
| `Speculative_Rag_..._vsequential_BioASQ_solver.zip` | 538 MB | overrides + resultados | `02-rag/_overrides/bioasq/`, `03-resultados/bioasq/` |
| `Local_BioASQ_test1_NoPromptTuning.zip` | 3.0 GB | árbol **canónico** de `01-kraq/src` (27 `.py`, superset) + config + adapter | `01-kraq/` |
| `Local_hotpot_NoPromptTuning.zip` | 2.8 GB | overrides + config | `01-kraq/_overrides/hotpot/`, `01-kraq/configs/hotpot/` |
| `Local_Pubhealth_NoPromptTuning.zip` | 1.4 GB | overrides + config | `01-kraq/_overrides/pubhealth/`, `01-kraq/configs/pubhealth/` |
| `Local_TriviaQA_Mistral7Bv0.1_300.zip` | 1.4 GB | overrides + config | `01-kraq/_overrides/triviaqa/`, `01-kraq/configs/triviaqa/` |
| `Graphrag_BioASQ/.../Processing_dataset/` | (descomprimido) | scripts de preprocesamiento de datasets | `01-kraq/preprocesamiento/` |
| `Local_*.zip` → `output/community_questions_*.json`, `naive_questions.json` | 119 MB | las preguntas KRAQ generadas por fθ | `04-datos/kraq-preguntas-generadas/<ds>/` |
| `Speculative_Rag_*_solver.zip` → `DataBase/qa_benchmark.json` | 40–650 KB | sets de evaluación | `02-rag/DataBase/` y `_overrides/<ds>/DataBase/` |
| `ResultadosTesis.txt` | 2.6 KB | números publicados, anotados por el autor | `03-resultados/ResultadosTesis.txt` |
| `Local_*.zip` → `output/*.parquet` | 670 MB | grafos de conocimiento de GraphRAG (entidades, relaciones, comunidades, reportes, text units, documentos) | `04-datos/grafos-graphrag/<ds>/` |
| `Local_*.zip` → `input/` | ~60 MB | corpus que indexó GraphRAG | `04-datos/corpus/<ds>/graphrag-input/` |
| `Speculative_Rag_*_solver.zip` → `DataBase/{input,chunks}/` | ~370 MB | corpus del pipeline RAG (pubhealth y triviaqa; bioasq es idéntico a su graphrag-input; el de hotpot era una copia del de TriviaQA y no se copió) | `04-datos/corpus/<ds>/rag-{input,chunks}/` |
| `Local_*.zip` → `logs/` | ~60 MB | logs reales del indexado de GraphRAG (bioasq, hotpot, pubhealth; triviaqa no traía) | `01-kraq/logs-indexado/<ds>/` |
| `Local_BioASQ_*.zip` → `checkpoint-{7500,7899}/trainer_state.json` | 277 KB | registro del entrenamiento del QLoRA (curvas de loss, schedule) | `modelos/llama3-8b-qlora-finetuned/` |

`MateriasOptativas.txt` no se copió: es la lista de materias del plan de estudios, no tiene
que ver con el código.

## Criterio de selección del canónico

- **`02-rag` → hotpot.** Tiene 71 archivos `.py` contra 60–63 de los otros tres, y es el
  único que trae `solver/` (5 archivos), `ablation_combined/` (3) y `solver_vbusqueda_EM/`.
  Es también el dataset del que salen los ablations de α y n del Cap. 4.
- **`01-kraq` → BioASQ.** 27 archivos `.py` contra 23–26; contiene todo lo que tienen los
  otros tres más `count_questions.py`.

## Deduplicación

- **Adapter QLoRA:** los 4 zips KRAQ traían una copia de `llama3-8b-qlora-finetuned/`
  (109 MB de `adapter_model.safetensors` + 17 MB de tokenizer). Se verificó por CRC que las
  4 son **byte-idénticas** (`908f92dd`) y se conservó una sola, en `modelos/`.
  Los `checkpoint-*/` intermedios no se copiaron.
- **Prompts de GraphRAG:** los 13 archivos de `prompts/` son idénticos en los 4 datasets
  (verificado con `diff -rq`). Se copian igual una vez por dataset dentro de
  `configs/<ds>/prompts/` porque GraphRAG los espera junto a su `settings.yaml`.
- **Código:** 32 de los archivos de `02-rag` y 17 de los de `01-kraq` eran idénticos en los
  4 forks. Se guardan una vez.

## Lo que se dejó afuera a propósito

| Qué | Por qué |
|---|---|
| `cache/` de GraphRAG (2 GB+ por dataset) | artefacto derivado, se regenera |
| `output/lancedb/` (~1 GB por dataset) | embeddings derivados; sólo hacen falta para `graphrag query` |
| `qdrant_db/`, `qdrant_client/collection/` | índices, se regeneran |
| Respuestas crudas de cada corrida (`.json` por pregunta) | cientos de MB; los resúmenes sí están |
| `checkpoint-*/` del adapter (pesos y optimizer) | pesos intermedios del fine-tuning; sí se rescataron sus `trainer_state.json` |
| `__pycache__/` | binarios de Python |
| `.env` con claves reales | se reemplazó por `.env.example` |
| `Graphrag_BioASQ/` descomprimida (5.6 GB) | variante anterior con 7 `.py`, subconjunto del zip `Local_BioASQ_test1` (27 `.py`); sólo se rescataron sus scripts de preprocesamiento |
| `Tesis/`, `Tesis - copia*/`, `TesisTests/`, `Tesis_Speculative_Backup/` | fase piloto con la API de OpenAI sobre corpus de Harry Potter; **no** produjeron ningún número de la tesis |

## Modificaciones aplicadas

### Seguridad (2 líneas)

```
02-rag/TraditionalRag/utils/pinecone_connection.py:22
02-rag/_overrides/bioasq/Speculative_Rag_Modified/utils/pinecone_connection.py:22

-   api_key = os.environ.get("PINECONE_API_KEY", "<clave de Pinecone en claro>")
+   api_key = os.environ["PINECONE_API_KEY"]  # ver .env.example (clave removida del codigo)
```

Se verificó con un grep de patrones de credenciales que **no queda ninguna** en este
árbol. Las claves originales siguen en `../BackupTesis/` y en los `.zip`.

### Rutas absolutas de la máquina original (3 archivos)

El código traía rutas `/home/<usuario>/Documents/Teo Gutter/Tesis/...` que no existen en
ninguna otra máquina. Se cambiaron a relativas, dejando la nota `# antes: ruta absoluta`:

- `02-rag/DataBase/chunkit.py:45-46` y `02-rag/_overrides/pubhealth/DataBase/chunkit.py:45-46`
  → `input/` y `chunks/` relativos al propio archivo.
- `02-rag/Speculative_Rag_Modified/analyze_benchmark_times.py:169` → default relativo a `02-rag/`.
- `01-kraq/logs-indexado/*/`: los logs de GraphRAG traían la ruta de la máquina original
  en cada línea; se reemplazó el home original por `/home/usuario` (sólo cambia la ruta,
  no el contenido del log).

### Archivos derivados (1)

`02-rag/DataBase/qa_benchmark_hotpot_reconstruido.json` **no existía en el backup**: se
generó extrayendo `id`/`question`/`reference_answer` de los resultados publicados de hotpot.
El motivo y la verificación están en `02-rag/DataBase/README.md`.

Nada más fue tocado. `requirements.txt` es nuevo (el original no tenía).

## Cómo se generó esto

El inventario de qué archivo es idéntico, cuál difiere y en qué datasets se calculó por
hash MD5 y está en `_reportes/inventario_rag.json` y `_reportes/inventario_kraq.json`.
Los `diff -u` completos están en `diffs/`.
