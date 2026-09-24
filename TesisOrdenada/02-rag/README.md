# 02-rag — Capítulos 4 y 5: los métodos de RAG y su evaluación

Los cuatro métodos que compara la tesis, los benchmarks, los ablations y la evaluación.

> Los nombres de directorio son los originales del backup, **a propósito**: los `import`
> del código dependen de ellos (`from Speculative_Rag_Modified.utils.inbedder import InBedder`).
> Renombrar carpetas rompería el árbol. La traducción a nombres legibles está abajo.

## Los cuatro métodos

| Carpeta | Nombre en la tesis | Capítulo | Archivo principal |
|---|---|---|---|
| `TraditionalRag/` | RAG tradicional (baseline) | — | `traditional_rag.py` |
| `TraditionalRag_CombinedRetrieve/` | **Combined Retrieve RAG** | **4** | `traditional_rag_combined_retrieve_flexible.py` |
| `SpeculativeRag/` | Speculative RAG (baseline, Wang et al.) | — | `main.py` |
| `Speculative_Rag_Modified/` | **Efficient Speculative RAG** | **5** | `modified_rag.py` |

Cada uno tiene la misma forma interna:

```
<Metodo>/
├── <metodo>.py          la implementación
├── run_<metodo>.py      corre UNA consulta con prints de debug (para entender el flujo)
├── setup.py
├── utils/
│   ├── rag.py           prompts del drafter y el verifier, cálculo de ρ
│   ├── inbedder.py      InBedder (RoBERTa enmascarado) — sólo en los Speculative
│   ├── clustering.py    K-means sobre los embeddings instruidos
│   └── models.py        dataclasses
└── Benchmark/
    ├── benchmark_*.py   corre el set completo, con checkpointing
    └── exact_match.py   métrica EM
```

### Los dos parámetros del Capítulo 4

`traditional_rag_combined_retrieve_flexible.py` (la versión parametrizada, la que usa el
ablation):

- `alpha` — proporción de los `top_k` documentos que se recuperan con el embedding de la
  **consulta original**; el resto se reparte entre las preguntas KRAQ similares.
  α=1 equivale a RAG tradicional, α=0 a recuperar sólo desde las preguntas.
- `num_similar_questions` (n) — cuántas preguntas KRAQ similares se traen de
  `questions-index-finetuned`.

`traditional_rag_combined_retrieve.py` es la misma lógica con α y n fijos.

### La diferencia del Capítulo 5

`modified_rag.py:232` — la instrucción que recibe el InBedder es la **pregunta KRAQ
similar**, no la consulta del usuario:

```python
instruction = f"{similar_question}"
...
doc_embeddings, batch_time = await inbedder.encode(documents, instruction, n_mask=3)
```

Todo lo demás (clustering en `k`, `m` drafts, verifier con ρ_Draft/ρ_SC/ρ_SR) es igual a
`SpeculativeRag/`. Comparar los dos archivos es la forma más rápida de ver la contribución.

Los `m` drafts se generan **secuencialmente** (una GPU sola), y el código estima la latencia
paralela como `max(draft_times)` / `max(verification_times)` — las ecuaciones 5.2 y 5.3 de
la tesis. Ver `../docs/01-arquitectura.md`.

## Infraestructura y evaluación

```
DataBase/                indexado en Qdrant
├── chunkit.py                  parte el corpus en chunks
├── index_chunks_qdrant.py      → colección "chunks"
├── index_questions_qdrant.py   → colección "questions-index-finetuned"
├── qdrant_config.py / qdrant_stats.py / clear_qdrant_collection.py

solver/                  evaluación (Exact Match + LLM-as-judge)
├── solver_dual_combined_exact_match.py   traditional vs combined, MISMA semilla ← Cap. 4
├── solver_exact_match.py                 busca semilla y evalúa
├── benchmark_{traditional,combined}_exact_match.py
└── analyze_results.py                    consolida
solver_exact_match_runs/llm_evaluator_parallel.py   LLM-as-judge en paralelo
solver_vbusqueda_EM/     variante anterior del solver (se conserva por trazabilidad)

ablation.py                              barrido k × top_k (Cap. 5)
ablation_combined/
├── benchmark_ablation_combined.py       barrido α × n, 16 celdas (Cap. 4)
├── consolidate_ablation_results.py      → CSV consolidado
└── re_evaluate_with_solver_prompt_parallel.py   re-evalúa respuestas ya generadas

_overrides/<dataset>/    lo que cambia por dataset (prompts adaptados, hiperparámetros)
```

## Antes de correr

- vLLM en `:8000` y Ollama en `:11434` (hardcodeados en cada módulo).
- Los hiperparámetros por default **no** son los de la tesis. Ver
  `../docs/03-resultados-y-trazabilidad.md`.
- Varios scripts fijan `cuda:1`. Ver `../docs/06-riesgos-y-faltantes.md` §2.5.
