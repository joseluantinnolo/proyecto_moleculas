# Predicción Topológica de Espectros Quirópticos en Helicenos mediante Redes Neuronales Informadas por la Física (PINN)

Este repositorio contiene la arquitectura de Machine Learning y la pipeline de procesamiento de datos desarrollada para el Trabajo de Fin de Grado en Física (Universidad de Córdoba). El modelo implementa un mecanismo de **Auto-Atención (Self-Attention)** para acoplar descriptores electrónicos y estéricos moleculares, optimizando la predicción morfológica del Dicroísmo Circular Electrónico (ECD) mediante restricciones energéticas y topológicas gaussianas.

## 🧬 Resumen de la Arquitectura
* **Entrada (64D):** Parámetros vectoriales por cada una de las 16 posiciones de sustitución del heliceno (Constantes de Hammett, Volumen de Van der Waals, Resonancia $R^+$ y Resonancia $R^-$).
* **Núcleo de Atención:** Capa de Auto-Atención Multi-Cabeza para mapear interacciones no locales entre sustituyentes.
* **Función de Pérdida PINN:** Pérdida mixta equilibrada ($\alpha=0.2, \beta=0.8$) que penaliza tanto las desviaciones en los parámetros estructurales gaussianos como el error cuadrático medio espectral.

## 🛠️ Estructura de la Pipeline de Ejecución
Para reproducir completamente los resultados del trabajo de investigación desde cero, ejecute los módulos en el siguiente orden estricto:

### Fase 1: Preparación y Purificación de Datos
1. **Unificación de bases de datos crudas y filtrado por corte energético batocrómico (650 nm):**
```bash
python src/data/01_unify_and_purify.py
```

2. **Extracción y codificación multivariable de descriptores químicos:**
```bash
python src/data/02_extract_features.py
```

3. **Deconvolución Híbrida mediante el Motor Atómico V2.5 (Ajuste de las 10 Gaussianas):**
```bash
python src/data/03_fit_gaussian_engine.py
```

### Fase 2: Modelado y Entrenamiento
4. **Validación Cruzada (Estudio K-Fold de robustez estructural):**
```bash
python src/train/01_evaluate_kfold.py
```

5. **Entrenamiento del Modelo Definitivo de Producción (100% de los datos químicos):**
```bash
python src/train/02_train_production.py
```