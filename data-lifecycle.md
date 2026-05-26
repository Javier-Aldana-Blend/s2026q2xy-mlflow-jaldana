# Conexión: Data Lifecycle → ML Lifecycle

Este documento hace explícita la relación entre el **ciclo de vida del dato** visto anteriormente y el **ciclo de vida de un modelo de Machine Learning**.

---

## El data lifecycle (repaso)

En sesiones anteriores vimos que los datos siguen un flujo estructurado:

```
Ingestión → Almacenamiento → Transformación → Consumo → Gobernanza
```

Cada etapa tiene actores, herramientas y responsabilidades definidas.

---

## El ML lifecycle

Cuando usamos datos para entrenar un modelo, ese flujo se extiende y especializa:

```
Ingestión → Exploración → Preprocesamiento → Entrenamiento
    → Evaluación → Registro → Despliegue → Inferencia → Monitoreo
```

---

## Comparación directa

| Etapa data lifecycle | Etapa ML lifecycle | ¿Qué cambia? |
|---|---|---|
| **Ingestión** | **Ingestión + Exploración** | Se añade EDA (Exploratory Data Analysis) para entender la distribución antes de transformar |
| **Transformación** | **Preprocesamiento + Feature engineering** | La transformación ahora debe ser reproducible y estar empaquetada junto al modelo (en este taller: `StandardScaler` dentro de un `Pipeline`) |
| **Almacenamiento** | **Registro del modelo** | No solo se almacenan datos, sino artefactos: el modelo, sus parámetros, métricas, versión de dependencias (MLflow Registry) |
| **Consumo** | **Despliegue + Inferencia** | El consumo ya no es una consulta SQL o un dashboard; es un endpoint REST que recibe datos nuevos y devuelve predicciones en tiempo real |
| **Gobernanza** | **Gobernanza + MLOps** | Se añaden lineaje del modelo, versionado, comparación de experimentos y transición entre stages (Staging → Production) |

---

## Qué se mantiene

- La necesidad de **reproducibilidad**: así como un pipeline de datos debe producir el mismo output dado el mismo input, un pipeline de ML debe poder reentrenarse y obtener resultados comparables.
- La importancia del **versionado**: tanto los datos como los modelos deben estar versionados para poder auditar y revertir cambios.
- La separación entre **entorno de desarrollo y producción**: lo que corre en el laptop del data scientist no es lo mismo que lo que va a producción — Docker resuelve esta brecha en ambos mundos.

---

## Qué cambia

- **El artefacto final no es un dato, es un modelo.** Un dataset transformado es estático; un modelo es una función que se ejecuta activamente sobre datos nuevos.
- **El consumo es activo.** Un dato se consulta; un modelo se invoca con inputs variables y produce outputs en tiempo real.
- **Los errores son más silenciosos.** Un dato mal ingresado falla rápido (constraint violation, schema error). Un modelo mal entrenado puede servir predicciones incorrectas durante semanas antes de ser detectado — de ahí la importancia del monitoreo.

---

## Qué se amplía

- **Tracking de experimentos:** no basta con versionar el código. Se deben versionar también los hiperparámetros, las métricas de evaluación y el dataset usado. MLflow Tracking cubre todo esto.
- **Ciclo de vida del modelo:** los modelos tienen etapas (desarrollo → staging → producción → retirado) que requieren gobernanza explícita. MLflow Registry gestiona estas transiciones con aliases.
- **Reproducibilidad del entorno:** en el data lifecycle, basta con versionar el código del pipeline. En ML, hay que versionar también el entorno de Python completo (versiones de sklearn, numpy, etc.) porque el comportamiento del modelo depende de ellos — de ahí el `requirements.txt` y el `MLmodel` con `conda.yaml`.

---

## Visualización del flujo completo (este taller)

```
Dataset breast_cancer (sklearn)
        │
        ▼ [Exploración local — Fase 1]
  Análisis exploratorio
  (shape, balance de clases, distribuciones)
        │
        ▼ [Docker — Fase 2 y 3]
  Preprocesamiento (StandardScaler)
  + Entrenamiento (RandomForestClassifier)
        │
        ├──► MLflow Tracking
        │    (accuracy_train, accuracy_test, artefactos)
        │
        ├──► MLflow Registry
        │    (versión 1 → alias "Staging")
        │
        ▼ [Docker — Fase 4]
  Serving REST (puerto 1234)
        │
        ▼
  Inferencia
  (curl / Python → array de probabilidades)
```

---

## Reflexión final

> El data lifecycle nos enseñó a llevar datos desde la fuente hasta el consumo de forma gobernada y reproducible.  
> El ML lifecycle extiende ese mismo principio: ahora el "consumo" no es un reporte o un dashboard, sino un modelo vivo que recibe datos en tiempo real y produce decisiones.  
> Las mismas preguntas aplican: ¿De dónde vienen los datos? ¿Están bien transformados? ¿Quién es responsable de cada etapa? ¿Cómo revertimos si algo falla?

La diferencia es que en ML, la respuesta a esas preguntas requiere herramientas adicionales: MLflow, registros de modelos, monitoreo de drift, y pipelines automatizados como los que provee Amazon SageMaker.
