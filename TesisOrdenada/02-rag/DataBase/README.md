# DataBase — indexado en Qdrant y sets de evaluación

## Scripts

| | |
|---|---|
| `chunkit.py` | parte el corpus (`./input/`) en chunks (`./chunks/`). Rutas relativas a este archivo (antes eran absolutas de la máquina original). |
| `index_chunks_qdrant.py` | chunks → colección `chunks` |
| `index_questions_qdrant.py` | preguntas KRAQ (`../../04-datos/...`) → colección `questions-index-finetuned`; set de evaluación (`qa_benchmark.json`) → colección `questions-benchmark` |
| `qdrant_config.py`, `qdrant_stats.py`, `clear_qdrant_collection.py` | conexión, inspección, limpieza |

## Sets de evaluación (`qa_benchmark.json`) ⚠️

Los benchmarks **no leen este JSON**: leen la colección Qdrant `questions-benchmark`
(`get_questions_from_qdrant`). El JSON es lo que se cargó en esa colección.

| Dataset | Archivo | Registros | Estado |
|---|---|---|---|
| TriviaQA | `_overrides/triviaqa/DataBase/qa_benchmark.json` | 300 | ✓ original |
| PubHealth | `_overrides/pubhealth/DataBase/qa_benchmark.json` | 3000 | ✓ original |
| BioASQ | `_overrides/bioasq/DataBase/qa_benchmark.json` | 1000 | ✓ original |
| **HotPotQA** | `qa_benchmark_hotpot_reconstruido.json` | 300 | **reconstruido** (ver abajo) |
| — | `qa_benchmark_OJO_es_triviaqa.json` | 300 | archivo que venía en el fork hotpot; es byte-idéntico al de TriviaQA (CRC `020beea5`) |

**Qué pasó con hotpot:** el fork hotpot se creó copiando el de TriviaQA y el
`qa_benchmark.json` en disco nunca se actualizó — contiene preguntas de TriviaQA
(`"What does a seismologist study?"`). Pero la colección Qdrant `questions-benchmark` de
hotpot **sí** tenía las preguntas de HotPotQA: los resultados publicados
(`03-resultados/hotpot/solver_exact_match_runs/.../full_results.json`) son 300 preguntas
de HotPotQA (`"Morgan Paull played Dave Holden in a 1982 film..."`).

`qa_benchmark_hotpot_reconstruido.json` es exactamente esas 300 preguntas
(`id`, `question`, `answer`) extraídas de ese `full_results.json`. Es el set con el que se
midieron los números del Cap. 4 en hotpot. Para regenerar la colección:
renombrarlo a `qa_benchmark.json` y correr `index_questions_qdrant.py`.
