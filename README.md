# s2026q2xy-mlflow-jaldana

Repositorio del taller hands-on **MLflow + Docker** para el Semillero 2026 Q2.

Autor: Javier Aldana

---

## Descripción

Este repositorio contiene el material completo para el taller práctico sobre MLflow y Docker, enfocado en recorrer el ciclo de vida completo de un modelo de Machine Learning: desde los datos crudos hasta un endpoint REST de inferencia.

El taller usa un clasificador de sklearn sobre el dataset `breast_cancer` como vehículo pedagógico (no como dominio de estudio), para conectar los conceptos del data lifecycle ya vistos con el ciclo de vida de un modelo en producción.

---

## Materiales del taller

| Archivo | Descripción |
|---|---|
| [`taller.md`](./taller.md) | Guion completo del taller — fases, instrucciones paso a paso, preguntas de reflexión |
| [`data-lifecycle.md`](./data-lifecycle.md) | Conexión explícita entre el data lifecycle y el ML lifecycle |

---

## Código fuente

| Archivo | Descripción |
|---|---|
| [`clf-train.py`](./clf-train.py) | Entrena y guarda el modelo localmente (sin registry) |
| [`clf-train-registry.py`](./clf-train-registry.py) | Entrena, registra métricas y versiona el modelo en MLflow Registry |
| [`Dockerfile`](./Dockerfile) | Imagen Docker para todo el pipeline |
| [`docker-compose.yml`](./docker-compose.yml) | Pipeline completo con MLflow Registry (UI en puerto 8000) |
| [`docker-compose-no-registry.yml`](./docker-compose-no-registry.yml) | Pipeline simplificado sin registry |
| [`compose-server.yml`](./compose-server.yml) | Solo el servidor MLflow (sin entrenamiento) |
| [`predict.sh`](./predict.sh) | Script de inferencia vía curl |
| [`runServer.sh`](./runServer.sh) | Script para levantar el servidor MLflow |
| [`serveModel.sh`](./serveModel.sh) | Script para servir un modelo registrado |
| [`requirements.txt`](./requirements.txt) | Dependencias del entorno Docker |

---

## Prerequisitos

- Docker y Docker Compose
- Python 3.10+ (solo para la exploración local de datos en Fase 1)

---

## Ejecución rápida (TLDR)

### Con MLflow Registry (recomendado para el taller)

```bash
docker compose -f docker-compose.yml up --build
```

- MLflow UI disponible en: **http://localhost:8000**
- Inferencia disponible en: **http://localhost:1234**

### Sin registry

```bash
docker compose -f docker-compose-no-registry.yml up --build
```

- Inferencia disponible en: **http://localhost:1234**

### Hacer una predicción

```bash
./predict.sh test.csv
```

### Limpiar contenedores

```bash
docker compose down
```

---

## Estructura del taller (2 horas)

```
Fase 0 — Introducción y conexión con data lifecycle     (15 min)
Fase 1 — Exploración local del dataset                  (15 min)
Fase 2 — Entrenamiento + serving sin registry           (25 min)
Fase 3 — MLflow Registry y UI                           (25 min)
Fase 4 — Inferencia vía REST API                        (15 min)
Fase 5 — Cierre y contextualización MLOps               (15 min)
```

Ver guion detallado en [`taller.md`](./taller.md).

---

## Tutorial base

Este taller se basa en:  
**Machine Learning Model Serving for Newbies with MLflow** — Maria Patterson  
https://towardsdatascience.com/machine-learning-model-serving-for-newbies-with-mlflow-76f9f0ac3cb2?sk=3fabd570be956c5830591f9ac0fa7991

---

## Lecturas complementarias

- Intro práctica a MLflow: https://medium.com/@kevinnjagi83/mlflow-tutorial-63a9ab1a220d
- MLOps en AWS + Azure: https://docs.aws.amazon.com/es_es/prescriptive-guidance/latest/patterns/build-an-mlops-workflow-by-using-amazon-sagemaker-and-azure-devops.html
- Repo original del tutorial: https://github.com/mtpatter/mlflow-tutorial
