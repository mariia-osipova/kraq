# Arquitectura: cómo encaja todo

## El pipeline completo, de punta a punta

```
                        ┌─────────────────────────────────────────┐
   corpus del dataset ─►│ 01-kraq  (Capítulo 3)                   │
   (TriviaQA/HotPot/    │                                         │
    PubHealth/BioASQ)   │  GraphRAG indexa el corpus              │
                        │    → entidades, relaciones, comunidades │
                        │    → community_reports.parquet          │
                        │                                         │
                        │  fθ (LLaMA-3.1-8B + QLoRA) genera       │
                        │  preguntas por comunidad                │
                        │    → questions.json                     │
                        └──────────────┬──────────────────────────┘
                                       │  se indexan con nomic-embed-text
                                       ▼
                        ┌─────────────────────────────────────────┐
                        │ Qdrant (en disco)                       │
                        │   colección "chunks"                    │
                        │   colección "questions-index-finetuned" │
                        │   colección "questions-index-base"      │
                        │   colección "questions-index-random"    │
                        └──────────────┬──────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
   │ TraditionalRag   │   │ TraditionalRag_    │   │ SpeculativeRag       │
   │ (baseline)       │   │ CombinedRetrieve   │   │ (baseline, Wang+)    │
   │                  │   │ ── Capítulo 4 ──   │   │                      │
   │ top_k docs de la │   │ α·top_k de la      │   │ recupera → InBedder  │
   │ consulta → LLM   │   │ consulta +         │   │ → k clusters →       │
   │                  │   │ (1-α)·top_k de las │   │ m drafts en paralelo │
   │                  │   │ n preguntas KRAQ   │   │ → verifier elige     │
   │                  │   │ similares          │   │                      │
   └──────────────────┘   └────────────────────┘   └──────────┬───────────┘
                                                              │
                                                   ┌──────────▼───────────┐
                                                   │ Speculative_Rag_     │
                                                   │ Modified             │
                                                   │ ── Capítulo 5 ──     │
                                                   │ igual, pero la       │
                                                   │ instrucción del      │
                                                   │ InBedder es la       │
                                                   │ pregunta KRAQ        │
                                                   │ similar, no la query │
                                                   └──────────────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       ▼
                        ┌─────────────────────────────────────────┐
                        │ 02-rag/solver/  — evaluación            │
                        │   Exact Match + LLM-as-judge            │
                        │   → 03-resultados/                      │
                        └─────────────────────────────────────────┘
```

## Infraestructura (§3.2.2 de la tesis)

Todo corre local, sobre una **RTX 3090 (24 GB)**:

| Rol | Servicio | Puerto | Modelo |
|---|---|---|---|
| LLM (drafter, verifier, indexado, judge) | vLLM | `8000` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |
| Embeddings | Ollama | `11434` | `nomic-embed-text:latest` |
| Vector store | Qdrant embebido (on-disk) | — | carpeta `qdrant_client/` o `qdrant_db/` |
| InBedder | HuggingFace local | — | RoBERTa `AutoModelForMaskedLM`, en `cuda:1` |
| fθ (generador de preguntas) | HuggingFace + PEFT, 4-bit | — | `modelos/llama3-8b-qlora-finetuned` sobre LLaMA-3.1-8B |

Ambos servicios se consumen con el cliente de OpenAI apuntado a `localhost`, por eso el
código usa `AsyncOpenAI(base_url=...)`. **No hay llamadas a la API de OpenAI en el camino
caliente** — las constantes `OPENAI_*` que quedan en `.env.example` son residuo de la fase
piloto y no se usan en estos experimentos.

## Los tres pipelines de KRAQ (para el ablation del Cap. 3)

`01-kraq/src/` tiene tres variantes en paralelo, y el sufijo del archivo dice cuál es:

| Sufijo | Qué genera las preguntas | Colección Qdrant | En la tesis |
|---|---|---|---|
| `_finetuned` | fθ = LLaMA-3.1-8B + adapter QLoRA | `questions-index-finetuned` | la propuesta |
| `_base` | LLaMA-3.1-8B Instruct sin fine-tune | `questions-index-base` | ablation "instruct" |
| `_random` | preguntas desde fragmentos random, sin grafo | `questions-index-random` | ablation "random" |

El ablation correspondiente (BioASQ, top_k=15) da: instruct LLM=75.7 / random LLM=77.7 /
finetuned LLM=79.5 (ver `03-resultados/ResultadosTesis.txt`).

Los archivos `*_nofind.py` son la variante que **no** usa los *findings* del community
report, sólo el resumen principal.

## Qué hace distinto cada método

**TraditionalRag** (`traditional_rag.py`) — embebe la consulta, trae `top_k` chunks de la
colección `chunks`, arma un prompt único y responde. Es el piso.

**TraditionalRag_CombinedRetrieve** (`traditional_rag_combined_retrieve_flexible.py`, Cap. 4) —
embebe la consulta, busca las `n` preguntas KRAQ más similares en
`questions-index-finetuned`, y reparte el presupuesto de `top_k` documentos:
`α·top_k` recuperados con el embedding de la consulta original y el resto repartido entre
los embeddings de las preguntas similares. Deduplica por id y arma un solo prompt.
La versión `_flexible` es la que parametriza α y n (la usa el ablation);
`traditional_rag_combined_retrieve.py` es la de α y n fijos.

**SpeculativeRag** (`main.py`) — reproduce Speculative RAG: recupera `top_k` documentos,
los reembebe con el **InBedder** usando la consulta como instrucción, hace K-means en `k`
clusters, muestrea `m` subconjuntos (uno por cluster), genera `m` drafts y el verifier los
puntúa con ρ_Draft / ρ_SC / ρ_SR para elegir uno.

Los `m` drafts se generan **secuencialmente**, no en paralelo — con una sola RTX 3090 no
había forma de correr `m` instancias del drafter a la vez. La tesis lo dice explícitamente
(§5.2.3) y el código lo refleja: mide el tiempo de **cada** draft y de **cada** verificación
por separado, y calcula la latencia paralela estimada como el máximo, que es exactamente
las ecuaciones 5.2 y 5.3:

```python
# modified_rag.py:329 y :417
parallel_draft_estimate  = max(draft_times)
parallel_verify_estimate = max(verification_times)
```

Las latencias publicadas del Cap. 5 salen de ahí. Es el punto donde código y tesis coinciden
con más precisión.

**Speculative_Rag_Modified** (`modified_rag.py`, Cap. 5) — idéntico salvo un punto:
en la línea del InBedder la instrucción **no** es la consulta del usuario sino la pregunta
KRAQ más similar recuperada de Qdrant (`modified_rag.py:232`, `instruction = f"{similar_question}"`).
La idea de la tesis es que esa instrucción es reutilizable entre consultas y por lo tanto
el embedding instruido se podría precomputar offline. **Ojo: en este código el InBedder
igual se ejecuta en línea** (`modified_rag.py:239`); el pre-cómputo se trata analíticamente
en §5.2.3–5.2.4 y nunca se implementó. Ver `docs/06-riesgos-y-faltantes.md`.

El módulo instrumenta el tiempo de cada etapa por separado (embedding / búsqueda de pregunta
similar / retrieval / InBedder / clustering / drafting / verification), que es de dónde salen
las tablas de latencia del Cap. 5.

## Evaluación

Dos métricas, siempre las dos:

- **Exact Match** — normaliza y compara contra la respuesta de referencia
  (`*/Benchmark/exact_match.py`).
- **LLM-as-judge** — un LLM decide si la respuesta es correcta
  (`solver/solver_exact_match.py`, `solver_exact_match_runs/llm_evaluator_parallel.py`).

`solver/solver_dual_combined_exact_match.py` corre traditional y combined sobre **el mismo
set de preguntas y la misma semilla**, que es lo que hace comparables los dos números del
Cap. 4.
