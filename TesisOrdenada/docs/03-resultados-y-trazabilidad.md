# Resultados: cada número de la tesis y el archivo que lo generó

La fuente autoritativa de los números es `03-resultados/ResultadosTesis.txt`
(las notas del autor). **Todos fueron cruzados contra el PDF de la tesis y coinciden.**

---

## Capítulo 4 — Combined Retrieve RAG (Tab. de §4.2)

| Dataset | Traditional EM | Combined EM | Traditional LLM | Combined LLM |
|---|---|---|---|---|
| HotPotQA | 57.0 | **58.6** | 76.3 | **77.3** |
| TriviaQA | 88.6 | **89.0** | 93.6 | **94.6** |
| PubHealth | 65.5 | **66.2** | 65.5 | **66.2** |
| BioASQ | **69.6** | 67.5 | 78.8 | **79.5** |

- **PDF:** líneas 3104–3118 del texto extraído.
- **Generado por:** `02-rag/solver/solver_dual_combined_exact_match.py`
- **Evidencia en disco:** `03-resultados/hotpot/solver_exact_match_runs/llm_evaluation_summary.json`
  (300 preguntas: traditional 171 EM / 229 LLM-correct; combined 176 EM / 232 LLM-correct →
  0.57 / 0.7633 y 0.5867 / 0.7733) y `.../final_summary.json`.
- **Config:** `top_k=15`, `α=0.5`, `n=2`, 300 preguntas, `seed=1`.

> Nota del autor en `ResultadosTesis.txt`: *"Combined: (TENGO QUE MEJORAR ESTOS NUMEROS)"*.
> Los números que quedaron en la tesis son estos igual.

## Ablation de α y n (HotPotQA, top_k=15)

| α (n=2) | EM | LLM | | n (α=0.5) | EM | LLM |
|---|---|---|---|---|---|---|
| 0.25 | 0.560 | 75.00 | | 1 | 0.5867 | 77.00 |
| 0.50 | 0.586 | 77.33 | | 2 | 0.5867 | 77.33 |
| 0.75 | 0.596 | 77.33 | | 3 | 0.5867 | 77.33 |
| | | | | 4 | 0.5700 | 76.00 |

- **PDF:** líneas 3215–3216.
- **Generado por:** `02-rag/ablation_combined/benchmark_ablation_combined.py`
- **Evidencia:** `03-resultados/hotpot/ablation_combined/combined_rag_analysis_summary.csv`,
  `consolidated_combined_rag_results.csv`, y los 16 directorios
  `ablation_study_seed23_topk15_*/alpha_*_similar_*/`.

## Capítulo 5 — Efficient Speculative RAG

| Dataset | Spec EM | Effic EM | Spec LLM | Effic LLM | Spec s | Effic s | Config |
|---|---|---|---|---|---|---|---|
| HotPotQA | 44.3 | 44.0 | 48.3 | 49.0 | 3.01 | **2.93** | top_k=10, m=8, k=4 |
| TriviaQA | 77.6 | 75.3 | 82.0 | 82.0 | 3.91 | **3.51** | top_k=10, k=2, m=5 |
| PubHealth | 58.3 | 58.0 | 58.3 | 58.0 | 3.81 | **3.36** | top_k=10, k=2, m=5, 600q |
| BioASQ | 51.4 | 50.6 | 56.6 | 56.4 | 4.32 | **3.97** | top_k=18, k=5, m=10, 500q |

- **PDF:** líneas 4133–4136.
- **Generado por:** `02-rag/Speculative_Rag_Modified/BenchMarks/benchmark_modified_train.py`
  y `02-rag/SpeculativeRag/Benchmark/benchmark_speculative.py`.
- **Evidencia:** `03-resultados/<dataset>/ablation_runs/*/{modified,speculative}/.../stats*.json`.
- **Desglose de latencia por etapa:** `02-rag/Speculative_Rag_Modified/analyze_benchmark_times.py`;
  salida en `03-resultados/hotpot/time_comparison_report.md` y `time_comparison_data.csv`.

## Ablation de k y top_k (Cap. 5, HotPotQA, k=3)

| top_k | Speculative | Modified |
|---|---|---|
| 10 | 3.01 s | 2.93 s |
| 15 | 3.10 s | 2.99 s |
| 20 | 3.23 s | 2.99 s |

- **Generado por:** `02-rag/ablation.py`
- **Evidencia:** `03-resultados/hotpot/ablation_runs/k{3..6}_m8_topk{10,15,20,25}_*/`
  (16 combinaciones) + `ablation_report.md`.

## Capítulo 3 — KRAQ

**Ablation del generador de preguntas (BioASQ, top_k=15):**

| Variante | EM | LLM |
|---|---|---|
| instruct (sin fine-tune) | 65.1 | 75.7 |
| random (sin grafo) | 65.3 | 77.7 |
| **finetuned (fθ QLoRA)** | **67.5** | **79.5** |

- **Generado por:** `01-kraq/src/Benchmarks/calculate_questions_similarity_{base,random,finetuned}.py`
  (BERTScore) y las tres colecciones Qdrant correspondientes.

**Volumen de preguntas KRAQ generadas** (`ResultadosTesis.txt`, con distribución por nivel
de comunidad): TriviaQA 17.378 · PubHealth 10.412 · HotPotQA 25.660 · BioASQ 19.100.

**Preguntas evaluadas por experimento:** PubHealth 2400 KRAQ / 1000 Traditional / 600 Speculative ·
TriviaQA 300 los tres · HotPotQA 2000 KRAQ / 300 Combined / 300 Speculative ·
BioASQ 1000 KRAQ / 1000 Traditional / 500 Speculative.

---

## Qué hay en `03-resultados/`

Sólo los **resúmenes** (`summary.json`, `stats*.json`, `report.md`, `*.csv`) — 411 archivos,
5 MB. Las respuestas crudas pregunta-por-pregunta (cientos de MB por dataset) siguen en
`../BackupTesis/*.zip`, en las mismas rutas relativas.

| Carpeta | Archivos | De dónde |
|---|---|---|
| `03-resultados/hotpot/` | 159 | `Speculative_Rag_..._hotpot_solver.zip` |
| `03-resultados/bioasq/` | 139 | `..._vsequential_BioASQ_solver.zip` |
| `03-resultados/pubhealth/` | 74 | `..._pubhealth_solver.zip` |
| `03-resultados/triviaqa/` | 39 | `..._triviaQA_solver.zip` |
