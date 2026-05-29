# Taller Hands-On: MLflow + Docker — del dato al modelo desplegado

**Duración:** 2 horas  
**Nivel:** Introductorio / práctico  
**Repositorio:** `s2026q2xy-mlflow-jaldana`

---

## Objetivo

Aplicar los conceptos del data lifecycle —ya vistos previamente— al ciclo de vida de un modelo de Machine Learning. Usaremos un clasificador sobre el dataset `breast_cancer` de sklearn únicamente como vehículo pedagógico: no nos interesa el dominio clínico, sino cómo los datos se convierten en un modelo, cómo se versiona y cómo se despliega.

---

## Prerequisitos

Antes del taller verifica que tienes instalado:

- **Docker** y **Docker Compose** — para ejecutar todo el pipeline
- **Python 3.10+** — solo para exploración local de datos
- **pip** — para instalar dependencias locales de exploración

```bash
docker --version
docker compose version
python --version
```

---

## Estructura del taller

| Fase | Actividad | Tiempo |
|------|-----------|--------|
| 0 | Introducción y conexión con data lifecycle
| 1 | Exploración local del dataset
| 2 | Entrenamiento + serving sin registry (Docker)
| 3 | MLflow Registry y UI (Docker)
| 4 | Inferencia vía REST API
| 5 | Cierre y contextualización MLOps

---

## Fase 0 — Introducción (15 min)

### ¿Qué es MLflow?

MLflow es una plataforma open source para gestionar el ciclo de vida de modelos de ML. Cubre cuatro áreas:

| Componente | Qué hace |
|---|---|
| **Tracking** | Registra parámetros, métricas y artefactos de cada experimento |
| **Projects** | Empaqueta el código de forma reproducible |
| **Models** | Estandariza el formato del modelo para múltiples frameworks |
| **Registry** | Versionado centralizado y gestión del ciclo del modelo |

### ¿Por qué todo dentro de Docker?

El entorno de entrenamiento debe ser idéntico al entorno de serving. Sin contenedores, diferencias de versión de sklearn entre el data scientist y el ingeniero generan errores silenciosos en producción. Docker garantiza reproducibilidad completa.

### Pregunta de reflexión grupal

> Del data lifecycle que ya vimos (ingestión → transformación → consumo), ¿en qué fase situarían el entrenamiento de un modelo? ¿Y el serving?

Consulta el archivo [`data-lifecycle.md`](./data-lifecycle.md) para la conexión explícita entre ambos ciclos.

---

## Fase 1 — Exploración local del dataset (15 min)

En esta fase instalamos dependencias **solo localmente** para explorar los datos antes de meterlos al pipeline de Docker.

### Instalar dependencias

```bash
pip install scikit-learn pandas
```

### Explorar el dataset

Abre un intérprete de Python o un notebook y ejecuta:

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df['target'] = cancer['target']

print(f"Shape: {df.shape}")
print(f"Clases: {cancer['target_names']}")  # ['malignant' 'benign']
print(df.describe())
print(df['target'].value_counts())
```

### Preguntas para discutir

- ¿Cuántas observaciones y features tiene el dataset?
- ¿Está balanceado el target?
- ¿Qué tipo de preprocesamiento podría necesitar?

> **Nota pedagógica:** a partir de aquí, el dataset pasa a Docker. Todo lo que viene — entrenamiento, tracking, registro, serving — ocurre dentro de contenedores.

---

## Fase 2 — Entrenamiento + serving sin registry (25 min)

En este modo el modelo se guarda en disco local (directorio `clf-model/`) sin usar el MLflow Registry. Es el camino más simple para ver el ciclo completo.

### Diagrama del flujo

```
docker compose up
    │
    ├─► trainmodel:  clf-train.py → guarda modelo en clf-model/
    │
    └─► servemodel:  mlflow models serve -m clf-model -p 1234
```

### Ejecutar

```bash
docker compose -f docker-compose-no-registry.yml up --build
```

Docker construye la imagen, entrena el modelo y levanta el servidor de inferencia en el puerto `1234`. La salida del log muestra:

```
trainmodel  | Test data written to 'test.csv'
trainmodel  | Model saved at path: clf-model
servemodel  | Registered model server at: http://0.0.0.0:1234
```

### Inspeccionar el modelo guardado

Una vez que termine el entrenamiento, en otra terminal:

```bash
ls clf-model/
# MLmodel  conda.yaml  model.pkl  python_env.yaml  requirements.txt
```

El archivo `MLmodel` describe el modelo en el formato estándar de MLflow:

```bash
cat clf-model/MLmodel
```

### Hacer una predicción

```bash
./predict.sh test.csv
```

O directamente con curl:

```bash
curl http://localhost:1234/invocations \
  -H 'Content-Type: text/csv' \
  --data-binary @test.csv
```

Respuesta esperada — array de probabilidades (una por observación de test):

```json
{"predictions": [0.9823, 0.1204, 0.8765, ...]}
```

> **Nota:** Si abres `http://localhost:1234` en el navegador verás "Not Found" — eso es correcto.
> El serving de MLflow solo expone `POST /invocations`, no una UI navegable.
> El puerto `8000` es la UI; el puerto `1234` es exclusivamente para inferencia vía API.

### Preguntas para discutir

- ¿Qué problema tiene guardar el modelo solo en disco local en un equipo real de trabajo?
- ¿Cómo sabrías qué versión de este modelo está sirviendo en producción?

---

## Fase 3 — MLflow Registry y UI (25 min)

En este modo agregamos el **MLflow Registry** con una base de datos SQLite. Esto nos da trazabilidad completa: qué parámetros, qué métricas, qué versión del modelo.

### Diagrama del flujo

```
docker compose up
    │
    ├─► server:      MLflow UI en localhost:8000 (SQLite backend)
    │
    ├─► trainmodel:  clf-train-registry.py → registra modelo + métricas
    │                → asigna alias "Staging" a la versión más nueva
    │
    └─► servemodel:  mlflow models serve -m models:/clf-model@Staging -p 1234
```

### Ejecutar

```bash
docker compose -f docker-compose.yml up --build
```

### Explorar la UI

Abre el navegador en **http://localhost:8000**

Navega por:

1. **Experiments → my-experiment** — lista de runs con métricas
2. **Clic en un run** — ver `accuracy_train`, `accuracy_test`, artefactos
3. **Models → clf-model** — versiones registradas del modelo
4. **Alias "Staging"** — la versión actualmente en serving

### Qué hace el script `clf-train-registry.py`

El script ejecuta los siguientes pasos clave dentro de un `mlflow.start_run()`:

```python
mlflow.log_metric('accuracy_train', accuracy_train)
mlflow.log_metric('accuracy_test', accuracy_test)
mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
mlflow.register_model(model_uri=model_uri, name=artifact_path)
# ...
client.set_registered_model_alias(name=artifact_path, alias="Staging", version=...)
```

Cada vez que reentrenas, se crea una nueva versión. El alias `Staging` apunta siempre a la más reciente.

### Ejercicio: re-entrenar y comparar versiones

Sin detener los servicios, ejecuta un segundo entrenamiento desde otra terminal (dentro del contenedor):

```bash
docker compose run --rm trainmodel
```

Regresa a la UI en `localhost:8000` y observa:
- ¿Cuántas versiones existen ahora de `clf-model`?
- ¿A qué versión apunta el alias `Staging`?

---

## Fase 4 — Inferencia vía REST API (15 min)

El serving de MLflow expone un endpoint REST estándar que cualquier cliente puede consumir.

### Endpoint

```
POST http://localhost:1234/invocations
Content-Type: text/csv
Body: datos en formato CSV (sin columna target)
```

### Probar con curl

```bash
./predict.sh test.csv
```

### Probar desde Python

```python
import requests
import pandas as pd

df = pd.read_csv('test.csv')
csv_data = df.to_csv(index=False)

response = requests.post(
    'http://localhost:1234/invocations',
    headers={'Content-Type': 'text/csv'},
    data=csv_data
)
print(response.json())
```

### Interpretar los resultados

La respuesta es un array de probabilidades (entre 0 y 1). El modelo fue configurado para devolver la probabilidad de la clase positiva (benigno = 1).

### Preguntas para discutir

- ¿Qué ventaja tiene un REST API frente a compartir el archivo `.pkl` del modelo?
- ¿Quién puede consumir este endpoint? ¿Necesita saber Python?

---

## Fase 5 — Cierre y contextualización MLOps (15 min)

### Resumen del flujo que ejecutamos

```
Datos raw (sklearn)
    │
    ▼
Preprocesamiento + Entrenamiento (sklearn Pipeline)
    │
    ▼
MLflow Tracking (métricas, parámetros, artefactos)
    │
    ▼
MLflow Registry (versiones, aliases: Staging → Production)
    │
    ▼
Serving (REST API en Docker → cualquier cliente)
    │
    ▼
Inferencia (curl / Python / cualquier servicio)
```

### Este mismo flujo en AWS

En entornos empresariales el mismo patrón escala así:

| Local (este taller) | AWS |
|---|---|
| Docker local | Amazon SageMaker |
| SQLite + filesystem local | Amazon S3 + RDS |
| `localhost:8000` (MLflow UI) | MLflow gestionado en Databricks / SageMaker Experiments |
| `localhost:1234` (serving) | SageMaker Endpoints (con auto-scaling) |
| Manual `docker compose up` | Pipelines automatizados (Step Functions / CodePipeline) |

### Lecturas complementarias

- **Intro práctica a MLflow:** https://medium.com/@kevinnjagi83/mlflow-tutorial-63a9ab1a220d
- **MLOps en AWS + Azure:** https://docs.aws.amazon.com/es_es/prescriptive-guidance/latest/patterns/build-an-mlops-workflow-by-using-amazon-sagemaker-and-azure-devops.html
- **Repo completo MLflow + Docker:** https://github.com/mtpatter/mlflow-tutorial

### Limpieza

```bash
docker compose down
```

---

## Troubleshooting frecuente

| Problema | Causa probable | Solución |
|---|---|---|
| `port 8000 already in use` | Otro proceso usa el puerto | `lsof -i :8000` y mata el proceso |
| `servemodel` falla al iniciar | `trainmodel` no terminó | Esperar o revisar logs: `docker compose logs trainmodel` |
| `curl` devuelve error 422 | CSV mal formateado | Verificar que `test.csv` no tenga columna `target` |
| `clf-model` ya existe al reentrenar sin registry | El directorio persiste | `rm -rf clf-model/` antes de volver a correr |
