# Diferencias por dataset

El árbol canónico está configurado para **HotPotQA** (`02-rag`) y **BioASQ** (`01-kraq`).
`_overrides/<dataset>/` contiene sólo los archivos que cambian, en la misma ruta relativa.
Los `diff -u` completos están en `diffs/`.

## Resumen

| | idénticos en los 4 | difieren | propios de un dataset |
|---|---|---|---|
| `02-rag` | 32 | 20 | 20 |
| `01-kraq` | 17 | 9 | 0 |

## Las diferencias que importan

### 1. Prompts adaptados a la tarea del dataset ⚠️

**Esta es la diferencia sustantiva.** No es configuración: cambia qué se le pide al modelo.

HotPotQA / TriviaQA / BioASQ usan el prompt genérico de QA:

```
Response to the instruction. Also provide a concise rationale that justifies the response.
###Instruction:
{"response": "your response here", "rationale": "your rationale here"}
```

PubHealth, que es verificación de afirmaciones, usa:

```
Given the following medical claim and evidence, determine whether it is true, false,
or a mixture of both. Then, provide a brief rationale explaining your reasoning.
###Claim:
{"response": "<true/false/mixture>", "rationale": "<your explanation>"}
```

Afecta a: `SpeculativeRag/utils/rag.py`, `Speculative_Rag_Modified/utils/rag.py`,
`TraditionalRag/traditional_rag.py`, `TraditionalRag_CombinedRetrieve/traditional_rag_combined_retrieve.py`.

PubHealth además saca el `"You can only use the evidence provided to answer the question"`
del system prompt.

### 2. Hiperparámetros por dataset

En los `run_*.py` y en los defaults de los benchmarks:

| Dataset | k | m | top_k |
|---|---|---|---|
| HotPotQA | 4 | 8 | 10 |
| TriviaQA | 2 | 5 | 10 |
| PubHealth | 2 | 5 | 10 |
| BioASQ | 5 | 10 | 18 |

(fuente: `03-resultados/ResultadosTesis.txt`; los defaults que están escritos en el código
no siempre coinciden — ver `docs/06-riesgos-y-faltantes.md`).

### 3. Consulta de ejemplo en los `run_*.py`

Cosmética pero delata el dataset: BioASQ `"Can sorafenib activate AMPK?"`, PubHealth
`"If Congress does not pass the renewal of the payroll tax cut..."`, TriviaQA
`"Three-in-One Pill Shows Promise in Beating High Blood Pressure"`.

### 4. Ruido de refactor

`device='cuda:1'` explícito vs. implícito en `SentenceTransformer`, `batch_size` del
InBedder (30 en hotpot vs 15 en pubhealth), un `import csv` de más, comentarios. En
`DataBase/index_questions_qdrant.py` pubhealth usa `question_id = i` (índice) en vez de
`question_entry["id"]`, y `BATCH_SIZE` 3000 vs 100.

⚠️ En `Speculative_Rag_Modified/utils/inbedder.py` la diferencia **sí afecta a la medición**:
hotpot devuelve `total_elapsed` (tiempo de pared de todo el encode) y pubhealth/bioasq
devuelven `batch_processing_time` (suma del tiempo de los batches). Son magnitudes
distintas y ambas alimentan las tablas de latencia del Cap. 5.

## Archivos propios de un solo dataset

**BioASQ** trae un módulo entero que no existe en los otros:
`TraditionalRag_SimilarRetrieve/` (9 archivos) — recupera usando **sólo** la pregunta
similar, sin mezclar con la consulta original. Es el caso α=0 del Cap. 4, implementado
como módulo aparte antes de que existiera la versión con α. Además:
`bioasq_solver_combined_retrieve.py`, `evaluate_exact_answers_llm.py`,
`exact_match_Traditional.py`.

**TriviaQA:** `combined_solver.py`, `time_comparison.py`, `time_comparison_ablation.py`,
`ablation_vanterior.py` (versión anterior del ablation).

**PubHealth:** `exact_match_discrepancies.py` (×2) — analiza casos donde EM y LLM-as-judge
discrepan.

**triviaqa / pubhealth / bioasq:** `solver.py` en la raíz. En hotpot esto evolucionó al
paquete `solver/` con 5 archivos, que es la versión más desarrollada.

## Listado completo

### `02-rag` — archivos que difieren

| Archivo | Difiere en |
|---|---|
| `DataBase/chunkit.py` | pubhealth |
| `DataBase/clear_qdrant_collection.py` | bioasq |
| `DataBase/index_questions_qdrant.py` | pubhealth, bioasq |
| `SpeculativeRag/Benchmark/benchmark_speculative.py` | triviaqa, pubhealth, bioasq |
| `SpeculativeRag/Benchmark/exact_match.py` | triviaqa, pubhealth |
| `SpeculativeRag/run_speculative_rag.py` | bioasq |
| `SpeculativeRag/utils/rag.py` | pubhealth |
| `Speculative_Rag_Modified/BenchMarks/benchmark_modified_train.py` | triviaqa, pubhealth, bioasq |
| `Speculative_Rag_Modified/BenchMarks/exact_match.py` | triviaqa |
| `Speculative_Rag_Modified/run_modified_rag.py` | pubhealth |
| `Speculative_Rag_Modified/utils/inbedder.py` | pubhealth, bioasq |
| `Speculative_Rag_Modified/utils/rag.py` | triviaqa, pubhealth |
| `TraditionalRag/Benchmark/benchmark_traditional_train.py` | triviaqa, pubhealth, bioasq |
| `TraditionalRag/Benchmark/exact_match.py` | pubhealth |
| `TraditionalRag/run_traditional_rag.py` | pubhealth |
| `TraditionalRag/traditional_rag.py` | triviaqa, pubhealth, bioasq |
| `TraditionalRag_CombinedRetrieve/Benchmark/benchmark_traditional_combined_question.py` | triviaqa, pubhealth, bioasq |
| `TraditionalRag_CombinedRetrieve/Benchmark/exact_match.py` | triviaqa, pubhealth |
| `TraditionalRag_CombinedRetrieve/traditional_rag_combined_retrieve.py` | triviaqa, pubhealth |
| `ablation.py` | triviaqa, pubhealth, bioasq |

### `01-kraq` — archivos que difieren

| Archivo | Difiere en |
|---|---|
| `Benchmarks/analyze_relevance_threshold.py` | triviaqa |
| `Benchmarks/entity_overlap_{base,finetuned,random}.py` | hotpot, triviaqa |
| `Benchmarks/generate_random_fragment_questions_{base,finetuned}.py` | triviaqa |
| `DataBase/load_questions_to_qdrant_random.py` | triviaqa |
| `QuestionGeneration/generate_naive_questions.py` | hotpot, pubhealth |
| `QuestionGeneration/llm_inference_base.py` | hotpot, pubhealth, triviaqa |

### Configuración de GraphRAG por dataset

| Dataset | LLM de indexado | Embeddings | Llamadas concurrentes |
|---|---|---|---|
| BioASQ | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | `nomic-embed-text` | 20 |
| HotPotQA | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | `nomic-embed-text` | — |
| PubHealth | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | `nomic-embed-text` | — |
| TriviaQA | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | `nomic-embed-text` | 15 |

