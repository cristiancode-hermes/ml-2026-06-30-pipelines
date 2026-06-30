# 🎯 Pipelines en scikit-learn — El arte de no olvidar nada

**Nivel:** 3 | **Skill:** Pipelines + ColumnTransformer + Preprocesamiento

## 📖 Qué vas a aprender

Imagina que estás preparando una receta de cocina compleja:
1. **Lavas las papas** (limpiamos los datos)
2. **Las pelas** (transformamos variables numéricas)
3. **Picas las verduras** (codificamos variables categóricas)
4. **Cocinas todo** (entrenas el modelo)

Sin Pipelines, harías cada paso manualmente y tendrías que acordarte de aplicar EXACTAMENTE las mismas transformaciones a los datos de entrenamiento y a los de prueba. Si se te olvida escalar los datos de prueba o aplicas un escalado diferente, el modelo falla en producción. 😱

**Pipelines** resuelve esto: encapsulas TODO el proceso (preprocesamiento + modelo) en un solo objeto. Un solo `.fit()` y un solo `.predict()` que aplica todo automáticamente. Nunca más se te olvidará escalar los datos de prueba.

**Hoy aprenderás:**
- `Pipeline`: encadenar transformaciones + modelo
- `ColumnTransformer`: aplicar transformaciones DIFERENTES a columnas numéricas y categóricas
- `SimpleImputer`: rellenar valores nulos automáticamente
- `OneHotEncoder`: convertir categorías a números
- `StandardScaler`: escalar variables numéricas
- Pasar el Pipeline completo a `GridSearchCV` para optimizar hiperparámetros
- Comparar con y sin pipeline para ver la diferencia

## 🧪 El Dataset: Titanic 🚢

Usaremos el dataset histórico del Titanic (891 pasajeros). ¿Quién sobrevivió? Tenemos:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `survived` | objetivo (0/1) | 0 = falleció, 1 = sobrevivió |
| `pclass` | categórica | 1ª, 2ª, 3ª clase |
| `sex` | categórica | male / female |
| `age` | numérica | edad (con **nulos** 😈) |
| `sibsp` | numérica | # hermanos/esposos a bordo |
| `parch` | numérica | # padres/hijos a bordo |
| `fare` | numérica | tarifa pagada |
| `embarked` | categórica | puerto de embarque (con **nulos**) |

Es perfecto para pipelines porque:
✅ Mezcla columnas numéricas y categóricas
✅ Tiene valores nulos que necesitan imputación
✅ Es un problema de clasificación real

---

## 🧪 Código paso a paso

### Paso 1: Cargar los datos y explorar

```python
import seaborn as sns

titanic = sns.load_dataset('titanic')
```

Primero cargamos el dataset con seaborn. Vamos a inspeccionar cuántos nulos hay y qué tipos de datos tenemos.

### Paso 2: Separar features (X) y target (y), y dividir en train/test

```python
from sklearn.model_selection import train_test_split

X = titanic.drop('survived', axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**`stratify=y`** es importante porque el Titanic tenía ~38% de supervivientes. Sin estratificar, podríamos terminar con una partición de prueba que no refleje esa proporción.

### Paso 3: Definir qué columnas son numéricas y cuáles categóricas

```python
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['pclass', 'sex', 'embarked']
```

Seleccionamos las columnas relevantes. Nota: `pclass` aunque es numérica (1, 2, 3), en realidad es categórica ordinal. Pero para simplificar, la tratamos como categórica nominal.

### Paso 4: Crear pipelines separados para numéricas y categóricas

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Rellenar nulos con la mediana
    ('scaler', StandardScaler())                    # Escalar a media=0, std=1
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Rellenar nulos con el valor más común
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Convertir categorías a vectores
])
```

**¿Por qué mediana para numéricas y most_frequent para categóricas?**
- La mediana es robusta a outliers, a diferencia del promedio
- En categóricas, el valor más frecuente es una opción razonable si no tenemos información adicional

### Paso 5: Combinar con ColumnTransformer

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])
```

`ColumnTransformer` es como un **embudo inteligente**: cada tipo de columna va a su pipeline específico, y al final todo se junta automáticamente.

### Paso 6: Pipeline COMPLETO (preprocesador + modelo)

```python
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

¡Este es el pipeline COMPLETO! Un solo objeto que:
1. Recibe datos CRUDOS del Titanic
2. Rellena nulos numéricos con la mediana
3. Escala las numéricas
4. Rellena nulos categóricos
5. One-hot encodea las categóricas
6. Entrena un Random Forest

### Paso 7: Entrenar con un solo `.fit()`

```python
full_pipeline.fit(X_train, y_train)
```

**Un solo `.fit()`** para todo el proceso. Comodidad total.

### Paso 8: Predecir con un solo `.predict()`

```python
y_pred = full_pipeline.predict(X_test)
```

**Un solo `.predict()`** que aplica automáticamente las MISMAS transformaciones que se aprendieron en `fit()`. Nunca se te olvidará escalar los datos de prueba.

### Paso 9: GridSearchCV con Pipeline

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'preprocessor__num__imputer__strategy': ['mean', 'median']
}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

**Nota los nombres con doble guion bajo:** `classifier__n_estimators` significa "dentro del paso `classifier` del pipeline, busca el parámetro `n_estimators`". Así puedes optimizar parámetros del preprocesador Y del modelo simultáneamente.

### Paso 10: La alternativa SIN pipeline (para comparar)

Hacemos todo manual para que veas por qué el pipeline es superior:
- Imputación separada en train y test
- Escalado separado
- One-hot encoding separado
- Mucho más código, más variables, más riesgo de errores

---

## 📊 Resultados

Al ejecutar `python project.py` verás:

```
=== PIPELINE AUTOMÁTICO (TODO EN UNO) ===
✅ Entrenando con un solo .fit()
✅ Prediciendo con un solo .predict()
Accuracy con Pipeline: 0.81

=== MANUAL (SIN PIPELINE) ===
🤯 Muchas líneas de código
🤯 Variables intermedias que se te pueden olvidar
Accuracy manual: 0.81 (mismo resultado, más trabajo)

=== GRIDSEARCH CON PIPELINE ===
Mejores parámetros: {classifier__max_depth: 5, classifier__n_estimators: 100, preprocessor__num__imputer__strategy: 'median'}
Mejor accuracy (CV): 0.823
```

También verás 5 visualizaciones:
1. **Mapa de nulos** — calor de dónde faltan datos
2. **Distribución de edad** por supervivencia
3. **Comparativa con/sin pipeline** (métricas lado a lado)
4. **Heatmap de GridSearch** — cómo afectan los parámetros
5. **Importancia de features** (¡tras el pipeline!)
6. **Matriz de confusión** del mejor modelo

---

## 💡 Conceptos clave

- **Pipeline**: encapsula una secuencia de transformaciones + un estimador final. Llamas `.fit()` UNA vez y hace todo.
- **ColumnTransformer**: aplica transformaciones DIFERENTES a grupos de columnas. Vital para datos mixtos (numéricas + categóricas).
- **SimpleImputer**: rellena valores nulos automáticamente con estrategias como media, mediana o moda.
- **OneHotEncoder**: convierte cada categoría en una columna binaria (0/1). Esencial para que los modelos entiendan datos categóricos.
- **StandardScaler**: centra y escala a varianza unitaria. Muchos modelos (SVM, KNN, regresión) requieren features en la misma escala.
- **Nombres con `__`**: en GridSearchCV con Pipeline, usas `paso__parametro` para acceder a parámetros anidados.
- **Ventaja clave del Pipeline**: garantiza que las transformaciones de entrenamiento sean EXACTAMENTE las mismas que en prueba. Previene el error más común en ML: data leakage.

## 🚀 Siguiente paso

**Clustering con K-Means** — El próximo skill pendiente del Nivel 3. Aprenderás a agrupar datos sin etiquetas, encontrar patrones ocultos, y usar el método del codo para elegir K. ¡ML no supervisado!
