# _overrides

El árbol de al lado está configurado para el dataset **canónico**
(HotPotQA en `02-rag`, BioASQ en `01-kraq`).

Cada subcarpeta acá contiene **sólo los archivos que cambian** para otro dataset, en la
misma ruta relativa. Para cambiar de dataset se superponen:

```bash
# desde 02-rag/
cp -r _overrides/pubhealth/. .

# desde 01-kraq/
cp -r _overrides/pubhealth/. src/
```

Hacelo sobre una copia o con git, para poder volver al canónico.

Las diferencias **no son cosméticas**: incluyen prompts adaptados a la tarea de cada
dataset (PubHealth verifica afirmaciones true/false/mixture; los demás responden preguntas)
e hiperparámetros distintos.

- Qué cambia exactamente y por qué: `../../docs/05-diferencias-por-dataset.md`
- Los `diff -u` completos, archivo por archivo: `../../diffs/`
