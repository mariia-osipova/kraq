# Riesgos conocidos y qué falta

Esto es una revisión honesta del código consolidado. Nada de acá invalida la tesis, pero
son las cosas que conviene tener contestadas antes de una defensa o de publicar el repo.

---

## 1. Inconsistencias entre la tesis y el código

### 1.1 Falta la corrida *PromptTuning*

La Tab. 3.1 de la tesis compara **NoPromptTuning vs PromptTuning**. Los cuatro zips KRAQ
son `NoPromptTuning` y los `prompts/` incluidos son los defaults de GraphRAG (verificado:
los 13 archivos son idénticos entre los 4 datasets). **La mitad de esa tabla no tiene
respaldo en este código.**

### 1.2 El pre-cómputo offline de embeddings no está implementado

Es la contribución conceptual del Cap. 5. En `02-rag/Speculative_Rag_Modified/modified_rag.py:239`
el InBedder se sigue ejecutando **en línea**:

```python
doc_embeddings, batch_time = await inbedder.encode(documents, instruction, n_mask=3)
```

Lo que la pregunta KRAQ similar aporta acá es la *instrucción* (línea 232), no un embedding
cacheado. La tesis lo trata como estimación analítica en §5.2.3–5.2.4, así que es coherente
— pero si alguien pregunta "¿y la versión precomputada?", la respuesta es que no se
implementó.

### 1.3 Falta el script de entrenamiento del adapter QLoRA

Está el resultado (`modelos/llama3-8b-qlora-finetuned/`, con `adapter_config.json` y
`training_args.bin`) pero no el código que lo entrenó, ni los datos de Dolly-v2 / MusiQue
procesados (§3.2.5). **fθ no es re-entrenable desde este repo**, sólo reutilizable.
Si ese script está en un Colab, vale la pena bajarlo.

---

## 2. Bugs y fragilidades en el código

### 2.1 BERTScore con `lang="es"` sobre datasets en inglés ⚠️

`01-kraq/src/utils/bert_score_utils.py:9`:

```python
P, R, F1 = score(candidates, references, lang="es", verbose=False, device=device)
```

Los cuatro datasets (TriviaQA, HotPotQA, PubHealth, BioASQ) son **en inglés**. El parámetro
`lang` selecciona el modelo subyacente de BERTScore, así que se está midiendo similitud de
texto inglés con un modelo elegido para español.

Esta función alimenta **todos** los números de relevancia del Cap. 3
(`calculate_questions_similarity_{base,random,finetuned}.py`, `generate_random_fragment_questions_*.py`,
`analyze_relevance_threshold.py`).

Atenuante importante: el error es **uniforme en las tres variantes**, así que la comparación
instruct vs random vs finetuned sigue siendo internamente consistente y el ordenamiento
probablemente se sostiene. Lo que no es directamente comparable con la literatura es el
valor absoluto.

### 2.2 Dos definiciones distintas de "tiempo del InBedder"

Ver `docs/05-diferencias-por-dataset.md` §4: hotpot mide tiempo de pared total del encode,
pubhealth/bioasq suman el tiempo de los batches. Ambos números van a las tablas de latencia
del Cap. 5. La diferencia debería ser chica, pero no es cero.

### 2.3 La separación rationale/response es heurística y degrada en silencio

En `SpeculativeRag/utils/rag.py:166` y `Speculative_Rag_Modified/utils/rag.py:166`, los
logprobs se parten en "los del rationale" y "los de la respuesta" buscando patrones de
substring entre los tokens:

```python
rationale_patterns = ['ration', 'rational', ' rationale']
response_patterns  = ['response', ' response', 'resp']
```

y tomando `min(indices)` de cada uno. Dos problemas:

- `'ration'` matchea también dentro del *texto* del rationale (`operation`, `rational`,
  `duration`, `narration`…), así que el corte puede caer en el lugar equivocado.
- Si no encuentra ambos campos, el fallback **parte los tokens por la mitad**
  (`mid_point = len(tokens) // 2`), y si algo falla el `except` deja
  `p_rationale = p_response = 0.5`.

Esto es bastante más robusto que la versión de la fase piloto (que directamente crasheaba),
pero la degradación es **silenciosa**: no hay log ni contador de cuántas veces se cayó al
fallback. Esos ρ_Draft / ρ_SC / ρ_SR son los que eligen el draft ganador, así que si el
fallback se dispara seguido, la selección se vuelve parcialmente aleatoria sin que nada lo
indique. Vale la pena instrumentarlo antes de sacar conclusiones finas del Cap. 5.

### 2.4 Vectores no recuperados se rellenan con ceros

`Speculative_Rag_Modified/modified_rag.py:210` y `:213` — si un documento no trae vector se
rellena con `np.zeros(768).tolist()`. Eso mete un punto en el origen del espacio de
embeddings y distorsiona el K-means en silencio. Convendría descartar el documento y
loguearlo.

### 2.5 `cuda:1` hardcodeado

Aparece en el InBedder, en `bert_score_utils.py` y en varios `SentenceTransformer(...)`.
En una máquina con una sola GPU hay que cambiarlo a `cuda:0`:

```bash
grep -rn "cuda:1" 01-kraq 02-rag
```

### 2.6 Typo en un nombre de colección

`02-rag/TraditionalRag_CombinedRetrieve/run_traditional_rag_similar_question.py:61` usa
`question_collection="question-index-finetuned"` (falta la **s**). El nombre correcto,
usado en los otros 24 lugares, es `questions-index-finetuned`. Es un script de demo de una
sola consulta, **no** el benchmark, así que no afecta ningún resultado publicado — pero
falla si lo corrés.

### 2.7 Defaults del código ≠ hiperparámetros publicados

`Speculative_Rag_Modified/BenchMarks/benchmark_modified_train.py:199-201` tiene
`k=5, m=10, top_k=20`, que no corresponde a ninguna corrida de la tesis. Los valores reales
están en `docs/03-resultados-y-trazabilidad.md` y hay que pasarlos explícitamente.

### 2.8 El `qa_benchmark.json` del fork hotpot era de TriviaQA (resuelto, documentado)

El fork hotpot arrastró el set de evaluación de TriviaQA en disco (byte-idéntico, CRC
`020beea5`). No afecta ningún número publicado — los benchmarks leen la colección Qdrant
`questions-benchmark`, que en hotpot sí tenía preguntas de HotPotQA, como confirman los
resultados guardados. Se reconstruyó el set real desde esos resultados. Detalle en
`02-rag/DataBase/README.md`. Es un buen ejemplo de por qué los forks por copia son peligrosos.

### 2.9 Restos de la fase piloto

`02-rag/TraditionalRag/utils/pinecone_connection.py` sigue apuntando por default al índice
`harry-potter-index` en Pinecone. En estos experimentos el vector store es Qdrant en disco;
el módulo quedó sin usar. Se puede borrar.

---

## 3. Seguridad

Las claves que estaban hardcodeadas **fueron removidas de este árbol** (`docs/04-procedencia.md`
detalla las dos líneas modificadas), pero:

- Siguen presentes en `../BackupTesis/` (`.env` y `pinecone_connection.py` originales) y
  **dentro de los `.zip`**.
- La misma clave de OpenAI y la misma de Pinecone (las del `.env` del backup) están
  además en `../Tesis/`, `../Tesis - copia*/`, `../TesisTests/` y `../Tesis_Speculative_Backup/`.

**Revocá ambas claves.** No alcanza con limpiarlas de este árbol: si alguna vez compartiste
un zip o subís cualquiera de las otras carpetas a GitHub, están expuestas.

---

## 4. Lo que sí está bien resuelto

Para que la lista de arriba no dé una impresión equivocada:

- **Checkpointing de corridas caras.** Los benchmarks detectan un `partial_results.json`
  previo y retoman desde ahí (`benchmark_modified_train.py:265`), y llevan
  `skipped_questions.txt`. La generación de preguntas KRAQ guarda por nivel de comunidad y
  además **captura la interrupción**: `generate_community_questions.py:30` instala un
  handler que avisa *"se guardará el progreso en el próximo checkpoint"* en vez de perder
  horas de generación. Corridas largas, caras y recuperables. Poca gente lo hace.
- **Instrumentación de latencias por etapa.** `modified_rag.py` cronometra embedding /
  búsqueda de pregunta similar / retrieval / InBedder / clustering / drafting /
  verification por separado. Es exactamente el desglose que necesita el Cap. 5, y estaba
  pensado desde el diseño, no agregado al final.
- **Honestidad sobre el paralelismo.** Los `m` drafts se generan secuencialmente porque hay
  una sola GPU, la tesis lo dice sin adornos (§5.2.3), y el código mide el tiempo de cada
  draft y cada verificación por separado para estimar la latencia paralela como el máximo
  (`modified_rag.py:329` y `:417`). Es la implementación literal de las ecuaciones 5.2 y 5.3.
  Código y tesis coinciden exactamente en el punto donde era más fácil exagerar.
- **Comparabilidad de los baselines.** `solver_dual_combined_exact_match.py` corre
  traditional y combined sobre el mismo set y la misma semilla. Es lo que hace legítima la
  comparación del Cap. 4.
- **Barridos completos.** 16 celdas de α×n para el Cap. 4 y 16 de k×top_k para el Cap. 5,
  con resultados guardados por celda.
