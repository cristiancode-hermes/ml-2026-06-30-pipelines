"""
===========================================================
  🚢 PIPELINES EN SCIKIT-LEARN — Titanic Survival Prediction
===========================================================
  Aprenderás:
    - Pipeline: encadenar transformaciones + modelo
    - ColumnTransformer: tratar numéricas y categóricas diferente
    - SimpleImputer: rellenar nulos automáticamente
    - OneHotEncoder + StandardScaler en el pipeline
    - GridSearchCV con pipelines
    - Comparativa con/sin pipeline (para ver la magia)

  Autor: Hermes ML Teacher 🤖
===========================================================
"""

# ============================================================
# 📦 IMPORTS — Cargamos las librerías que necesitamos
# ============================================================
import warnings
warnings.filterwarnings('ignore')  # Limpiamos la salida de warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score
)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

import os
os.makedirs('docs', exist_ok=True)  # Carpeta para guardar los gráficos

# Semilla para reproducibilidad
SEED = 42

print("=" * 65)
print("  🚢 PIPELINES EN SCIKIT-LEARN — Titanic Survival Prediction")
print("=" * 65)
print()

# ============================================================
# 1️⃣ CARGAR DATOS — Usamos el dataset Titanic de seaborn
# ============================================================
# El dataset Titanic tiene datos REALES de 891 pasajeros.
# Es perfecto porque mezcla columnas numéricas (age, fare)
# con categóricas (sex, embarked) y además tiene VALORES NULOS.
print("📥 Cargando dataset Titanic...")
titanic = sns.load_dataset('titanic')
print(f"   → {titanic.shape[0]} pasajeros, {titanic.shape[1]} columnas")
print()

# ============================================================
# 2️⃣ EXPLORACIÓN RÁPIDA — Vemos qué tenemos
# ============================================================
print("🔍 Exploración rápida:")
print(f"   - Supervivientes: {titanic['survived'].sum()} ({titanic['survived'].mean()*100:.1f}%)")
print(f"   - Hombres: {(titanic['sex']=='male').sum()} | Mujeres: {(titanic['sex']=='female').sum()}")

# Mostramos los primeros registros para ver la pinta de los datos
print("\n📋 Primeros 5 registros:")
print(titanic.head().to_string())
print()

# ============================================================
# 3️⃣ ANÁLISIS DE VALORES NULOS — El enemigo a vencer
# ============================================================
# Antes de construir el pipeline, veamos cuántos nulos tenemos.
# Así apreciaremos el trabajo que hace SimpleImputer automáticamente.
print("🔎 Valores nulos por columna:")
nulos = titanic.isnull().sum()
nulos_pct = (nulos / len(titanic)) * 100
nulos_df = pd.DataFrame({'Nulos': nulos, '%': nulos_pct.round(1)})
print(nulos_df[nulos_df['Nulos'] > 0].to_string())
print(f"\n   🧩 Total de celdas nulas: {titanic.isnull().sum().sum()}")
print()

# ============================================================
# 4️⃣ VISUALIZACIÓN 1 — Mapa de valores nulos
# ============================================================
# Un heatmap que muestra DÓNDE están los nulos en el dataset.
# Las rayas blancas horizontales son pasajeros con datos faltantes.
print("📊 Generando visualización 1: Mapa de valores nulos...")
fig, ax = plt.subplots(figsize=(10, 6))

# Seleccionamos solo las columnas que nos interesan para el modelo
cols_interes = ['age', 'fare', 'sibsp', 'parch', 'pclass', 'sex', 'embarked', 'survived']
sns.heatmap(titanic[cols_interes].isnull(), cbar=False, yticklabels=False,
            cmap='viridis', ax=ax)
ax.set_title('🧩 Mapa de Valores Nulos en Titanic\n(Líneas amarillas = datos faltantes)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Columnas')
ax.set_ylabel('Pasajeros (891)')

plt.tight_layout()
plt.savefig('docs/01_mapa_nulos.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/01_mapa_nulos.png")
print()

# ============================================================
# 5️⃣ VISUALIZACIÓN 2 — Distribución de edad por supervivencia
# ============================================================
# Vemos cómo la edad influye en la supervivencia.
# Los niños pequeños tenían más probabilidades de sobrevivir.
print("📊 Generando visualización 2: Edad vs Supervivencia...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de edades por supervivencia
sns.histplot(data=titanic, x='age', hue='survived', kde=True,
             bins=30, palette=['#e74c3c', '#2ecc71'], alpha=0.6, ax=axes[0])
axes[0].set_title('Distribución de Edad por Supervivencia', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Edad')
axes[0].set_ylabel('Cantidad')
axes[0].legend(['Falleció (0)', 'Sobrevivió (1)'], title='Supervivencia')

# Boxplot para ver medianas y outliers
sns.boxplot(data=titanic, x='survived', y='age', palette=['#e74c3c', '#2ecc71'], ax=axes[1])
axes[1].set_title('Boxplot: Edad por Supervivencia', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Sobrevivió')
axes[1].set_ylabel('Edad')
axes[1].set_xticklabels(['No (0)', 'Sí (1)'])

plt.tight_layout()
plt.savefig('docs/02_edad_supervivencia.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/02_edad_supervivencia.png")
print()

# ============================================================
# 6️⃣ PREPARAR DATOS — Separamos features (X) y target (y)
# ============================================================
# 'survived' es lo que queremos predecir (0 = falleció, 1 = sobrevivió)
# Todo lo demás son características (features) que usamos para predecir
X = titanic.drop('survived', axis=1)  # Features (datos de entrada)
y = titanic['survived']               # Target (lo que queremos predecir)

print("📐 Dimensiones:")
print(f"   X (features): {X.shape}")
print(f"   y (target): {y.shape} con {y.sum()} supervivientes ({y.mean()*100:.1f}%)")
print()

# ============================================================
# 7️⃣ DIVIDIR EN TRAIN/TEST — Con estratificación
# ============================================================
# Train (80%): aprende patrones
# Test (20%): evaluamos qué tan bien aprendió
# stratify=y: mantenemos la misma proporción de supervivientes en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("✂️ División train/test:")
print(f"   Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.0f}%)")
print(f"   Test:  {len(X_test)} muestras ({len(X_test)/len(X)*100:.0f}%)")
print(f"   % supervivientes en train: {y_train.mean()*100:.1f}%")
print(f"   % supervivientes en test:  {y_test.mean()*100:.1f}%")
print()

# ============================================================
# 8️⃣ DEFINIR COLUMNAS POR TIPO
# ============================================================
# Separamos las columnas en dos grupos:
#   - Numéricas: edad, tarifa, hermanos, padres → se escalan
#   - Categóricas: clase, sexo, puerto → se convierten a one-hot
# NOTA: 'alive' y 'who' son redundantes con 'survived' y 'sex',
#        'deck' tiene demasiados nulos, 'adult_male' y 'alone' derivan de otras.
#        Las excluimos para no complicar.
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['pclass', 'sex', 'embarked']

print("📋 Columnas seleccionadas:")
print(f"   🔢 Numéricas: {numeric_features}")
print(f"   🏷️ Categóricas: {categorical_features}")
print()

# ============================================================
# 9️⃣ CREAR PIPELINE DE PREPROCESAMIENTO — ¡LA MAGIA EMPIEZA!
# ============================================================
# Vamos a construir el "embudo inteligente":
#
#   Pipeline Numérico (age, fare, sibsp, parch):
#     1. SimpleImputer(median) → rellena nulos con la mediana
#     2. StandardScaler() → centra y escala (media=0, std=1)
#
#   Pipeline Categórico (pclass, sex, embarked):
#     1. SimpleImputer(most_frequent) → rellena nulos con el valor más común
#     2. OneHotEncoder() → convierte cada categoría en columna 0/1
#
#   ColumnTransformer une ambos pipelines:
#     - Columnas numéricas → pipeline numérico
#     - Columnas categóricas → pipeline categórico
#     - El resto de columnas (si hay) se descartan

# --- Pipeline para columnas NUMÉRICAS ---
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # 🔸 SimpleImputer: rellena los NaN con la mediana de cada columna.
    #    Elegimos mediana porque es robusta a outliers.
    #    Si hubiera elegido 'mean', valores extremos sesgarían el relleno.
    ('scaler', StandardScaler())
    # 🔸 StandardScaler: transforma los datos a media=0, desviación=1.
    #    Fórmula: (x - media) / std
    #    ¿Por qué? Random Forest NO necesita escalado, pero es buena práctica
    #    y si después cambiamos a KNN o SVM, ya está listo.
])

# --- Pipeline para columnas CATEGÓRICAS ---
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # 🔸 SimpleImputer con 'most_frequent': rellena NaN con la moda
    #    (el valor que más se repite). Para categóricas tiene sentido.
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    # 🔸 OneHotEncoder: convierte 'male'/'female' en [1,0] / [0,1].
    #    handle_unknown='ignore': si en test aparece una categoría nueva
    #    que no vimos en train, no explota, solo pone todo en 0.
])

# --- UNIR TODO con ColumnTransformer ---
# ColumnTransformer aplica pipelines DIFERENTES a grupos de columnas.
# Es como tener una línea de ensamblaje con dos carriles separados.
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    # 🔸 Las columnas numéricas van al pipeline numérico
    ('cat', categorical_pipeline, categorical_features)
    # 🔸 Las columnas categóricas van al pipeline categórico
])

print("🔧 Estructura del preprocesador:")
print(f"   Pipeline Numérico ({numeric_features}):")
print(f"     ├── SimpleImputer(strategy='median')")
print(f"     └── StandardScaler()")
print(f"   Pipeline Categórico ({categorical_features}):")
print(f"     ├── SimpleImputer(strategy='most_frequent')")
print(f"     └── OneHotEncoder()")
print()

# ============================================================
# 🔟 PROBAR EL PREPROCESADOR — Ver cómo queda transformado
# ============================================================
# Antes de añadir el modelo, vamos a ver el resultado del preprocesador
# para entender qué hace internamente.
print("🧪 Probando el preprocesador (solo transformaciones)...")
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Obtenemos los nombres de las columnas resultantes
# Las numéricas mantienen su nombre, las categóricas se expanden
cat_encoder_fitted = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features_out = cat_encoder_fitted.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_features_out)

print(f"   Datos originales: {X_train.shape[1]} columnas")
print(f"   Datos transformados: {X_train_preprocessed.shape[1]} columnas")
print(f"   Columnas resultantes: {all_features}")
print(f"   ¿Nulos después del pipeline?: {np.isnan(X_train_preprocessed).sum()}")
print()

# ============================================================
# 1️⃣1️⃣ PIPELINE COMPLETO — ¡Preprocesador + Modelo en uno!
# ============================================================
# 🔥 ESTO ES LO MÁS IMPORTANTE DEL PROYECTO 🔥
#
# Creamos un SOLO objeto que hace TODO:
#   1. Rellena nulos numéricos con la mediana
#   2. Escala las variables numéricas
#   3. Rellena nulos categóricos con la moda
#   4. Convierte categorías a one-hot
#   5. Entrena un Random Forest
#
# Cuando llames .fit() en este pipeline, se ejecutan los pasos 1-4
# en los datos de entrenamiento, y luego entrena el modelo (paso 5).
# Cuando llames .predict(), aplica los pasos 1-4 a los NUEVOS datos
# usando las transformaciones APRENDIDAS en el fit().
# ¡Nunca se te olvidará escalar los datos de prueba! 🎉

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # 🔸 Primer paso: transformar los datos crudos
    ('classifier', RandomForestClassifier(random_state=SEED))
    # 🔸 Segundo paso: clasificar con Random Forest
    #    n_estimators=100 por defecto (100 árboles)
])

print("🤖 Pipeline completo creado:")
print(f"   {full_pipeline}")
print()

# ============================================================
# 1️⃣2️⃣ ENTRENAR — ¡Un solo .fit() hace TODO!
# ============================================================
print("🏋️ Entrenando el Pipeline completo...")
# 🔥 Aquí pasa todo: imputación → escalado → one-hot → entrenamiento
full_pipeline.fit(X_train, y_train)
print("   ✅ Pipeline entrenado con éxito!")
print()

# ============================================================
# 1️⃣3️⃣ PREDECIR — ¡Un solo .predict() aplica todo!
# ============================================================
# Al predecir, el pipeline AUTOMÁTICAMENTE:
#   - Usa la mediana que aprendió en train para rellenar nulos de test
#   - Usa la media y std que aprendió en train para escalar test
#   - Usa las categorías que vió en train para one-hot en test
# Todo consistente, todo automático. 🤯
print("🔮 Prediciendo con el Pipeline...")
y_pred = full_pipeline.predict(X_test)
print("   ✅ Predicciones listas!")
print()

# ============================================================
# 1️⃣4️⃣ EVALUAR — ¿Qué tal funciona?
# ============================================================
accuracy_pipe = accuracy_score(y_test, y_pred)
precision_pipe = precision_score(y_test, y_pred)
recall_pipe = recall_score(y_test, y_pred)
f1_pipe = f1_score(y_test, y_pred)

print("=" * 50)
print("  📊 RESULTADOS DEL PIPELINE")
print("=" * 50)
print(f"   🎯 Accuracy:  {accuracy_pipe:.4f} ({accuracy_pipe*100:.1f}%)")
print(f"   🎯 Precision: {precision_pipe:.4f}  (de los que predijo como supervivientes, "
      f"cuántos realmente lo eran)")
print(f"   🎯 Recall:    {recall_pipe:.4f}  (de los supervivientes reales, "
      f"cuántos atrapó)")
print(f"   🎯 F1-Score:  {f1_pipe:.4f}  (balance entre precisión y recall)")
print()

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
print("📋 Matriz de Confusión:")
print(f"               ┌─────────────────┐")
print(f"               │  VP={cm[1,1]:3d}   FP={cm[0,1]:3d}  │")
print(f"               │  FN={cm[1,0]:3d}   VN={cm[0,0]:3d}  │")
print(f"               └─────────────────┘")
print(f"   VP = Verdaderos Positivos (bien predichos como supervivientes)")
print(f"   VN = Verdaderos Negativos (bien predichos como fallecidos)")
print(f"   FP = Falsos Positivos (predijo superviviente, pero falleció)")
print(f"   FN = Falsos Negativos (predijo fallecido, pero sobrevivió)")
print()

# Reporte de clasificación completo
print("\n📋 Reporte de clasificación completo:")
print(classification_report(y_test, y_pred, target_names=['Falleció', 'Sobrevivió']))
print()

# ============================================================
# 1️⃣5️⃣ VISUALIZACIÓN 3 — Matriz de Confusión
# ============================================================
print("📊 Generando visualización 3: Matriz de Confusión...")
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Falleció', 'Sobrevivió'],
    cmap='Blues',
    ax=ax,
    colorbar=False
)
ax.set_title('🔢 Matriz de Confusión — Pipeline Automático', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/03_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/03_matriz_confusion.png")
print()

# ============================================================
# 1️⃣6️⃣ GRIDSEARCH CON PIPELINE — Optimizar hiperparámetros
# ============================================================
# 🚀 LO MEJOR DE LOS PIPELINES: puedes optimizar parámetros del
#    preprocesador Y del modelo en una sola búsqueda.
#
# Nota los nombres con '__' (doble guion bajo):
#   'classifier__n_estimators' significa:
#     paso 'classifier' → parámetro 'n_estimators'
#
# Podemos incluso optimizar parámetros del preprocesador:
#   'preprocessor__num__imputer__strategy':
#     paso 'preprocessor' → paso 'num' → paso 'imputer' → parámetro 'strategy'
print("🔍 Iniciando GridSearchCV con Pipeline...")
print("   (Buscando mejores parámetros para preprocesador + modelo)")

param_grid = {
    # Parámetros del Clasificador
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    # 🔸 n_estimators: cantidad de árboles en el bosque
    # 🔸 max_depth: profundidad máxima de cada árbol (None = sin límite)

    # Parámetros del PREPROCESADOR (¡sí, también se pueden optimizar!)
    'preprocessor__num__imputer__strategy': ['mean', 'median']
    # 🔸 ¿Media o mediana para rellenar edad? Que GridSearch lo decida.
}

# GridSearchCV con 5-fold cross-validation
# Va a probar: 2 (n_estimators) × 3 (max_depth) × 2 (imputer) = 12 combinaciones
# Cada combinación se evalúa con 5-fold CV: 12 × 5 = 60 entrenamientos
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,     # Usar todos los núcleos de la CPU
    verbose=1      # Mostrar progreso
)

n_estimators_vals = len(param_grid['classifier__n_estimators'])
max_depth_vals = len(param_grid['classifier__max_depth'])
imputer_vals = len(param_grid['preprocessor__num__imputer__strategy'])
combinaciones = n_estimators_vals * max_depth_vals * imputer_vals
print(f"   🔬 Explorando: {combinaciones} combinaciones")
print(f"   ⏱️  Cada una con validación cruzada de 5 folds")
print(f"   🔄 Total de entrenamientos: {combinaciones * 5}")
print()

grid_search.fit(X_train, y_train)

print(f"\n✅ GridSearch completado!")
print(f"\n🏆 MEJORES PARÁMETROS ENCONTRADOS:")
for param, value in grid_search.best_params_.items():
    print(f"   • {param}: {value}")
print(f"\n   📈 Mejor accuracy (validación cruzada): {grid_search.best_score_:.4f} "
      f"({grid_search.best_score_*100:.1f}%)")

# Evaluamos el mejor modelo en el conjunto de prueba
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print(f"\n   📊 Mejor modelo en TEST:")
print(f"      Accuracy: {accuracy_best:.4f} ({accuracy_best*100:.1f}%)")
print(f"      F1-Score: {f1_best:.4f}")
print()

# ============================================================
# 1️⃣7️⃣ VISUALIZACIÓN 4 — Heatmap de GridSearch
# ============================================================
# Mostramos cómo varía el accuracy según los parámetros
print("📊 Generando visualización 4: Heatmap de GridSearch...")
resultados_cv = pd.DataFrame(grid_search.cv_results_)

# Crear tabla pivote para el heatmap
param_combos = resultados_cv.groupby(
    ['param_classifier__max_depth', 'param_classifier__n_estimators',
     'param_preprocessor__num__imputer__strategy']
)['mean_test_score'].max().reset_index()
param_pivot = param_combos.pivot_table(
    index=['param_classifier__max_depth', 'param_preprocessor__num__imputer__strategy'],
    columns='param_classifier__n_estimators',
    values='mean_test_score'
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(param_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=1, ax=ax, cbar_kws={'label': 'Accuracy (CV)'})
ax.set_title('🎯 GridSearchCV: Accuracy por Combinación de Parámetros',
             fontsize=14, fontweight='bold')
ax.set_xlabel('n_estimators (cantidad de árboles)')
ax.set_ylabel('max_depth | imputer_strategy')

plt.tight_layout()
plt.savefig('docs/04_gridsearch_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/04_gridsearch_heatmap.png")
print()

# ============================================================
# 1️⃣8️⃣ VISUALIZACIÓN 5 — Importancia de Features
# ============================================================
# Una ventaja de Random Forest es que podemos ver qué features
# fueron más importantes para la clasificación.
# Pero con el pipeline, las features transformadas tienen nombres diferentes.
# Tenemos que obtener los nombres de las columnas transformadas.
print("📊 Generando visualización 5: Importancia de Features...")

# Extraer el clasificador del pipeline
rf_best = best_model.named_steps['classifier']

# Obtener nombres de features transformadas
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

importances = rf_best.feature_importances_
# Ordenamos de mayor a menor importancia
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(indices)), importances[indices][::-1], color='steelblue')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
ax.set_xlabel('Importancia Relativa', fontsize=12)
ax.set_title('🌟 Importancia de Features (Random Forest + Pipeline)',
             fontsize=14, fontweight='bold')

# Añadir valores en las barras
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{importances[indices[::-1][i]]:.3f}',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('docs/05_importancia_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/05_importancia_features.png")
print()

# ============================================================
# 1️⃣9️⃣ VISUALIZACIÓN 6 — Comparativa Pipeline vs Sin Pipeline
# ============================================================
# Vamos a hacer TODO EL PROCESO MANUALMENTE para que veas
# por qué el pipeline es tan valioso.
print("📊 Generando visualización 6: Comparativa Pipeline vs Manual...")

# --- Alternativa SIN Pipeline (todo manual) ---
# Seleccionamos las mismas columnas manualmente
X_train_manual = X_train[numeric_features + categorical_features].copy()
X_test_manual = X_test[numeric_features + categorical_features].copy()

# Imputación manual de numéricas
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
X_train_manual[numeric_features] = num_imputer.fit_transform(X_train_manual[numeric_features])
X_test_manual[numeric_features] = num_imputer.transform(X_test_manual[numeric_features])

# Escalado manual
scaler = StandardScaler()
X_train_manual[numeric_features] = scaler.fit_transform(X_train_manual[numeric_features])
X_test_manual[numeric_features] = scaler.transform(X_test_manual[numeric_features])

# Imputación manual de categóricas
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_manual[categorical_features] = cat_imputer.fit_transform(X_train_manual[categorical_features])
X_test_manual[categorical_features] = cat_imputer.transform(X_test_manual[categorical_features])

# One-hot encoding manual
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train_manual[categorical_features])
X_test_encoded = encoder.transform(X_test_manual[categorical_features])

# Unir numéricas + categóricas codificadas
X_train_final = np.hstack([X_train_manual[numeric_features].values, X_train_encoded])
X_test_final = np.hstack([X_test_manual[numeric_features].values, X_test_encoded])

# Entrenar modelo manual
rf_manual = RandomForestClassifier(random_state=SEED)
rf_manual.fit(X_train_final, y_train)
y_pred_manual = rf_manual.predict(X_test_final)

accuracy_manual = accuracy_score(y_test, y_pred_manual)

# --- Comparación visual ---
metrics_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Pipeline Automático 🚀': [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ],
    'Manual (sin Pipeline) 😰': [
        accuracy_score(y_test, y_pred_manual),
        precision_score(y_test, y_pred_manual),
        recall_score(y_test, y_pred_manual),
        f1_score(y_test, y_pred_manual)
    ]
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico de barras comparativo
x = np.arange(len(metrics_df['Métrica']))
width = 0.35
axes[0].bar(x - width/2, metrics_df['Pipeline Automático 🚀'], width,
            label='Pipeline 🚀', color='#2ecc71', alpha=0.8)
axes[0].bar(x + width/2, metrics_df['Manual (sin Pipeline) 😰'], width,
            label='Manual 😰', color='#e74c3c', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_df['Métrica'])
axes[0].set_ylabel('Puntaje')
axes[0].set_title('📊 Pipeline vs Manual — Mismas métricas', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 1)
# Añadir valores en barras
for i, v in enumerate(metrics_df['Pipeline Automático 🚀']):
    axes[0].text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(metrics_df['Manual (sin Pipeline) 😰']):
    axes[0].text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# Tabla de líneas de código
lines_data = {
    'Aspecto': ['Líneas de código', 'Variables temporales', 'Riesgo de error',
                'Reproducibilidad', 'Facilidad de mantenimiento'],
    'Pipeline 🚀': ['~5', '0 (ocultas)', 'Muy bajo', 'Automática', 'Excelente'],
    'Manual 😰': ['~25', '5-7', 'Alto (olvidar escalar)', 'Hay que gestionarla', 'Propensa a errores']
}
lines_df = pd.DataFrame(lines_data)
# Mostrar tabla como texto en el gráfico
axes[1].axis('tight')
axes[1].axis('off')
table = axes[1].table(cellText=lines_df.values, colLabels=lines_df.columns,
                       cellLoc='center', loc='center', colWidths=[0.3, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4a7c59')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f0f0f0')

axes[1].set_title('📋 ¿Por qué usar Pipeline?', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Pipeline Automático 🚀 vs Procesamiento Manual 😰',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/06_comparativa_pipeline_manual.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/06_comparativa_pipeline_manual.png")
print()

# ============================================================
# 2️⃣0️⃣ MOSTRAR COMPARATIVA EN CONSOLA
# ============================================================
print("=" * 65)
print("  📊 COMPARATIVA: Pipeline vs Manual")
print("=" * 65)
print()
print(f"  {'Métrica':<15} {'Pipeline 🚀':<15} {'Manual 😰':<15} {'Diferencia':<15}")
print(f"  {'─'*15} {'─'*15} {'─'*15} {'─'*15}")
for _, row in metrics_df.iterrows():
    diff = row['Pipeline Automático 🚀'] - row['Manual (sin Pipeline) 😰']
    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
    print(f"  {row['Métrica']:<15} {row['Pipeline Automático 🚀']:<15.4f} "
          f"{row['Manual (sin Pipeline) 😰']:<15.4f} {diff_str:<15}")
print()

print("  💡 Las métricas son SIMILARES porque el pipeline hace EXACTAMENTE")
print("     lo mismo que el proceso manual, pero con MUCHO MENOS código")
print("     y CON GARANTÍA de que no hay errores de consistencia.")
print()

# ============================================================
# VISUALIZACIÓN 7 — Validación Cruzada (comprobación extra)
# ============================================================
print("📊 Generando visualización 7: Validación Cruzada con Pipeline...")
# Demostramos que el pipeline funciona perfectamente con cross_val_score
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='accuracy')

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(1, 6), cv_scores, color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6'],
              alpha=0.8, edgecolor='white', linewidth=1.5)
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Media: {cv_scores.mean():.4f}')

for i, score in enumerate(cv_scores):
    ax.text(i + 1, score + 0.005, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylim(cv_scores.min() - 0.02, cv_scores.max() + 0.02)
ax.set_xlabel('Fold de Validación Cruzada', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('🎯 Validación Cruzada 5-Fold con Pipeline\n(El pipeline funciona con CV sin problemas)',
             fontsize=14, fontweight='bold')
ax.legend()
ax.set_xticks(range(1, 6))

plt.tight_layout()
plt.savefig('docs/07_validacion_cruzada.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: docs/07_validacion_cruzada.png")
print(f"   📊 Scores CV: {cv_scores}")
print(f"   📊 Media CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print()

# ============================================================
# 2️⃣2️⃣ ANÁLISIS DE ERRORES — ¿Dónde falla el modelo?
# ============================================================
# Veamos en qué casos específicos se equivoca el modelo
print("🔍 Analizando errores del mejor modelo...")
errores = X_test.copy()
errores['real'] = y_test.values
errores['predicho'] = y_pred_best
errores['acierto'] = errores['real'] == errores['predicho']

# Casos donde el modelo se equivocó
falsos_positivos = errores[(errores['real'] == 0) & (errores['predicho'] == 1)]
falsos_negativos = errores[(errores['real'] == 1) & (errores['predicho'] == 0)]

print(f"\n   ❌ Falsos Positivos (predijo superviviente, pero falleció): {len(falsos_positivos)}")
if len(falsos_positivos) > 0:
    print(f"      Edad promedio: {falsos_positivos['age'].mean():.1f} años")
    print(f"      Mujeres: {(falsos_positivos['sex']=='female').sum()}/{len(falsos_positivos)}")

print(f"\n   ❌ Falsos Negativos (predijo fallecido, pero sobrevivió): {len(falsos_negativos)}")
if len(falsos_negativos) > 0:
    print(f"      Edad promedio: {falsos_negativos['age'].mean():.1f} años")
    print(f"      Mujeres: {(falsos_negativos['sex']=='female').sum()}/{len(falsos_negativos)}")

print(f"\n   ✅ Total de aciertos: {(errores['acierto']).sum()}/{len(errores)} "
      f"({errores['acierto'].mean()*100:.1f}%)")
print()

# ============================================================
# 🧠 RESUMEN FINAL — Qué aprendiste hoy
# ============================================================
print("=" * 65)
print("  🧠 RESUMEN — Lo que aprendiste hoy")
print("=" * 65)
print(f"""
  📌 Pipeline: encapsula transformaciones + modelo en UN solo objeto.
     → .fit() entrena todo. .predict() aplica todo automáticamente.
     → Previene el error de olvidar transformar los datos de prueba.

  📌 ColumnTransformer: aplica pipelines DIFERENTES a grupos de columnas.
     → Numéricas: imputación + escalado
     → Categóricas: imputación + one-hot encoding
     → Ideal para datos del mundo real con tipos mixtos.

  📌 GridSearchCV + Pipeline: optimiza parámetros del preprocesador
     y del modelo en UNA sola búsqueda.
     → Usa 'paso__parametro' con doble guion bajo.

  📌 Beneficio clave: reproducibilidad y robustez.
     → El mismo pipeline funciona en desarrollo y producción.
     → Menos código, menos bugs, más confianza.

  Métricas obtenidas:
  → Accuracy: {accuracy_pipe:.3f} ({accuracy_pipe*100:.1f}%)
  → F1-Score: {f1_pipe:.3f}
""")
print()

print("=" * 65)
print(f"  🚢 Proyecto completado: Pipelines en scikit-learn")
print(f"  📁 Ruta: /opt/data/ml-projects/2026-06-30-pipelines/")
print(f"  📊 Visualizaciones generadas: 7")
print("=" * 65)
