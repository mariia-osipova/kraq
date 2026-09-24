# Cómo reproducir

> Todo esto corre local. La tesis lo ejecutó sobre una **RTX 3090 de 24 GB**. Varios scripts
> tienen `cuda:1` hardcodeado (InBedder, BERTScore, SentenceTransformer): si tenés una sola
> GPU hay que cambiarlo a `cuda:0`. Ver `docs/06-riesgos-y-faltantes.md`.

## 0. Dependencias y servicios

```bash
pip install -r requirements.txt     # Python 3.10+
pip install graphrag                # indexado del Cap. 3
```

```bash
# LLM — vLLM en el puerto 8000
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
    --quantization awq --port 8000

# Embeddings — Ollama en el puerto 11434
ollama serve
ollama pull nomic-embed-text
```

Los `base_url` están hardcodeados como `http://localhost:8000/v1` y
`http://localhost:11434/v1` en cada módulo. Qdrant se usa **embebido en disco**
(`AsyncQdrantClient(path=...)`), no hace falta levantar un servidor.

## 1. Corpus y preguntas del dataset

**Atajo:** el corpus ya procesado de la corrida original está en
`04-datos/corpus/<dataset>/` (ver su README) — con eso se saltea este paso y se garantiza
usar exactamente los mismos documentos.

Para regenerarlo desde el dataset crudo, `01-kraq/preprocesamiento/` arma el corpus y el
set de evaluación. Los scripts están escritos para BioASQ; para los otros datasets hay que
ajustar el parser (ver nota abajo).

```bash
cd 01-kraq/preprocesamiento
python process_bioasq.py      # dataset crudo → processed_documents.json + processed_questions.json
python corpus_creation.py     # → un .txt por documento, para el input/ de GraphRAG
python create_questions.py    # → questions.json (set de evaluación)
```

## 2. Indexado con GraphRAG (Capítulo 3)

```bash
# la config de cada dataset está en 01-kraq/configs/<dataset>/
graphrag index --root <carpeta_del_dataset>
```

Produce `output/` con `entities.parquet`, `relationships.parquet`, `communities.parquet`
y `community_reports.parquet`. Los `prompts/` incluidos son los **defaults de GraphRAG**
(modalidad *NoPromptTuning* de la Tab. 3.1).

## 3. Generación de preguntas KRAQ

> Las preguntas ya generadas para los 4 datasets están en `04-datos/`. Este paso sólo hace
> falta para regenerarlas o para un dataset nuevo.

```bash
cd 01-kraq
# fθ = LLaMA-3.1-8B + adapter QLoRA (../modelos/llama3-8b-qlora-finetuned)
export PEFT_MODEL_PATH=../modelos/llama3-8b-qlora-finetuned
python src/QuestionGeneration/generate_community_questions.py      # variante finetuned
python src/QuestionGeneration/generate_community_questions_base.py # variante sin fine-tune
python src/Benchmarks/generate_random_fragment_questions_base.py   # baseline random
```

Guardan checkpoints por nivel de comunidad, así que una corrida interrumpida se retoma.

## 4. Indexado de chunks y preguntas en Qdrant

```bash
cd 02-rag
# atajo: los chunks originales ya están en ../04-datos/corpus/<ds>/rag-chunks/
# (copiarlos a DataBase/chunks/); si no, generarlos:
python DataBase/chunkit.py                # parte el corpus en chunks
python DataBase/index_chunks_qdrant.py    # → colección "chunks"
python DataBase/index_questions_qdrant.py # → colección "questions-index-finetuned"
                                          #   lee las preguntas KRAQ de ../04-datos/ y el set
                                          #   de evaluación qa_benchmark.json (ver DataBase/README.md:
                                          #   para hotpot usar qa_benchmark_hotpot_reconstruido.json)
python DataBase/qdrant_stats.py           # verificar qué quedó indexado
```

Las variantes `base` / `random` se cargan con los scripts homónimos de
`01-kraq/src/DataBase/load_questions_to_qdrant_{base,random}.py`.

## 5. Correr los métodos

Cada método tiene un `run_*.py` que ejecuta **una** consulta con prints de debug (útil para
ver el pipeline paso a paso) y un `Benchmark/benchmark_*.py` que corre el set completo.

```bash
cd 02-rag
python TraditionalRag/run_traditional_rag.py
python SpeculativeRag/run_speculative_rag.py
python Speculative_Rag_Modified/run_modified_rag.py
python TraditionalRag_CombinedRetrieve/run_traditional_rag_similar_question.py

# benchmarks completos
python Speculative_Rag_Modified/BenchMarks/benchmark_modified_train.py --limit 300
python TraditionalRag/Benchmark/benchmark_traditional_train.py --limit 300
```

Los benchmarks **guardan resultados parciales cada N preguntas** y llevan
`skipped_questions.txt`; una corrida cortada se retoma sin perder lo hecho.

> ⚠️ **Los defaults del código no son los hiperparámetros de la tesis.**
> `benchmark_modified_train.py` tiene `k=5, m=10, top_k=20` por default, que no corresponde
> a ninguna corrida publicada. Los valores reales por dataset están en
> `docs/03-resultados-y-trazabilidad.md` y hay que pasarlos explícitamente.

## 6. Evaluación

```bash
cd 02-rag/solver
python solver_dual_combined_exact_match.py   # traditional vs combined, misma semilla
python solver_exact_match.py                 # busca una semilla y evalúa
python analyze_results.py                    # consolida
cd ../solver_exact_match_runs
python llm_evaluator_parallel.py             # LLM-as-judge sobre las respuestas guardadas
```

## 7. Ablations

```bash
cd 02-rag
python ablation.py                                    # barrido k × top_k (Cap. 5)
python ablation_combined/benchmark_ablation_combined.py   # barrido α × n (Cap. 4)
python ablation_combined/consolidate_ablation_results.py  # → CSV consolidado
```

`benchmark_ablation_combined.py` barre `alpha ∈ {0, 0.25, 0.5, 0.75}` × `n ∈ {1,2,3,4}`
= 16 experimentos (línea 582 en adelante).

## Cambiar de dataset

El árbol canónico está configurado para **HotPotQA**. Para otro dataset:

```bash
cp -r 02-rag/_overrides/pubhealth/. 02-rag/     # o triviaqa / bioasq
cp -r 01-kraq/_overrides/pubhealth/. 01-kraq/src/
```

Eso sobrescribe los archivos que cambian (prompts adaptados a la tarea, hiperparámetros,
nombres de colección). Hacelo sobre una copia, o con git, para poder volver.
