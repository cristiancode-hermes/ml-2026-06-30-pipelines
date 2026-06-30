# Learnings — 2026-06-30

## Qué se hizo
Proyecto completo sobre **Pipelines en scikit-learn** usando el dataset Titanic. Se cubrió:
- Pipeline y ColumnTransformer para preprocesamiento mixto (numéricas + categóricas)
- SimpleImputer para rellenar nulos (median para numéricas, most_frequent para categóricas)
- StandardScaler y OneHotEncoder dentro del pipeline
- GridSearchCV con pipeline (optimizando parámetros del preprocesador y modelo simultáneamente)
- Comparativa Pipeline vs Manual para demostrar la ventaja
- Validación cruzada 5-fold con pipeline
- 7 visualizaciones generadas

## Conceptos nuevos
- [x] Pipeline: encadenar transformaciones + modelo en un solo objeto
- [x] ColumnTransformer: aplicar pipelines diferentes a grupos de columnas
- [x] SimpleImputer con distintas estrategias (median, most_frequent)
- [x] Pipeline + GridSearchCV con sintaxis `paso__parametro`
- [x] Optimizar parámetros del preprocesador durante la búsqueda
- [x] Extraer feature names después del pipeline
- [x] Uso de `handle_unknown='ignore'` en OneHotEncoder

## Dificultades encontradas
- F-strings con sintaxis compleja (split en múltiples líneas) dieron SyntaxError. Se resolvió calculando variables intermedias.
- El resumen final usaba `print("""...""")` con `{accuracy_pipe:.3f}` que se imprimía literal porque no era f-string. Se cambió a `print(f"""...""")`.

## Notas para el próximo proyecto
- Clustering con K-Means es el siguiente skill pendiente del Nivel 3
- Tema: aprendizaje no supervisado, método del codo, silueta
- Dataset sugerido: datos de clientes de supermercado (mall customers) o similar
- El pipeline aprendido hoy se puede reutilizar en proyectos de clustering con datos mixtos
