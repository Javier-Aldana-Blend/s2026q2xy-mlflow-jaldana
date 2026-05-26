Machine learning model serving for newbies with MLflow

A fully-reproducible, Dockerized, step-by-step tutorial for building an API for your sklearn model
Maria Patterson
Jul 11, 2021
6 min read
Keep those models flowin'! (Poorly drawn cartoon by author.)
[updated for latest mlflow version as of Sep 2024]

A common problem in machine learning is the fumbling handoff between the data scientists building machine learning models and the engineers trying to integrate these models into working software. The compute environment that data scientists are comfortable with doesn’t always slide nicely into production quality systems.

This model deployment problem has become so pervasive that a whole new term and subfield have emerged around finding solutions- MLOps, or the application of DevOps principles for pushing machine learning pipelines to production.
A simple recipe for model deployment

My new favorite tool for machine learning model deployment is MLflow, which calls itself an "open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry." MLflow works with pretty much every programming language you might use for machine learning, can run easily the same way on your laptop or in the cloud (with an awesome managed version integrated into Databricks), helps you version models (especially great for collaboration) and track model performance, and allows you to package up pretty much any model and serve it up so that you or anyone else can use it to make predictions by sending their own data through a REST API without running any code.

If it sounds really cool to you, that’s because it is. MLflow is so cool and does so many things that it took me forever to sift through all the documentation and tutorials to figure out how to actually use it. But I did (I think), and I do everything with Docker, so I thought I’d go ahead and add another tutorial to the mess. Since it’s all containerized with Docker, it should only take a couple of commands to get everything working.
Ingredients

All materials are available in my GitHub mlflow-tutorial repo. To follow along, clone the repo to your local environment. You can run the example with only Docker and Docker Compose on your system.

This GitHub repo walks through an example of training a classifier model with sklearn and serving the model with mlflow.

The repo has a few different components:

    A Dockerfile that can be used to build a Docker image for this tutorial
    A pip install requirements file for package versions
    Two example sklearn classifier model training scripts that build mlflow models either 1) using the mlflow registry (UI on localhost:8000) and a sqlite database or 2) saving the model locally without using the registry
    Two Docker Compose files to run the whole training-to-serving pipeline end-to-end either using the mlflow registry or saving the model locally
    A Docker Compose file that you can use standalone to run mlflow with a sqlite backend (UI on localhost:8000)
    A couple example shell scripts for running the mlflow server for the registry, serving a specified model, and making predictions with a csv of test data.

Directions (TLDR version)

To skip through and run all components with Docker Compose you can run this whole tutorial with the registry:

docker compose -f docker-compose.yml up --build

You can access the mlflow registry UI on your localhost at port 8000.

Or without the registry:

docker compose -f docker-compose-no-registry.yml up --build

The model will be served on port 1234 to access predictions by running the following script with a csv of test data:

./predict.sh test.csv

which runs the following curl command taking a csv as input:

curl http://localhost:1234/invocations 
-H 'Content-Type: text/csv' --data-binary @test.csv

And returns an array of predicted probabilities.
Directions (full version)

The first section saves the mlflow model locally to disk, and the second section shows how to use the mlflow registry for model tracking and versioning.
Training and serving an mlflow model (no registry)

The [clf-train.py](https://github.com/mtpatter/mlflow-tutorial/blob/main/clf-train.py) script uses the sklearn breast cancer dataset, trains a simple random forest classifier, and saves the model to local disk with mlflow. Adding the optional flag for writing output test data will split the training data first to add an example test data file.

To train the model, use the following command:

python clf-train.py clf-model --outputTestData test.csv

Below is a code snippet from the main script. The last line saves the model components locally to the clf-model directory.

Serve the model by running the following command:

mlflow models serve -m clf-model -p 1234 -h 0.0.0.0 --env-manager local

You can then make predictions by running the following script with a csv of test data:

./predict.sh test.csv

which runs the following curl command:

curl http://localhost:1234/invocations 
-H 'Content-Type: text/csv' --data-binary @test.csv

Training and serving models with an mlflow registry

Using an mlflow registry gives you the ability to do a lot of cool stuff:

    Have a central location for tracking and sharing different experiments
    Keep track and easily view model and code input parameters
    Log and view metrics (accuracy, recall, whatever)
    Version your models and compare performance of different models logged under the same experiment
    Collaborate with any other users who can access the registry for registering their models under the same experiment
    Easily transition models into and out of different versions (or "aliases") (e.g., Staging, Production), which means that, instead of referring to a specific numbered model, you can swap in your latest favorite model to a specific stage and downstream components can seamlessly use that new model by referring to that stage

First start an mlflow server to use a registry locally by running the following script:

./runServer.sh

which just runs the following command:

mlflow server 
 --backend-store-uri sqlite:///mlflow.db 
 --default-artifact-root ./mlflow-artifact-root 
 --host 0.0.0.0 
 --port 8000

That will run the mlflow UI visible in your browser at localhost:8000.

The [clf-train-registry.py](https://github.com/mtpatter/mlflow-tutorial/blob/main/clf-train-registry.py) script uses the sklearn breast cancer dataset, trains a simple random forest classifier, and saves and registers the model and metrics to an mlflow registry at a specified url. Adding the optional flag for writing output test data will split the training data first to add an example test data file.

To train the model, use the following command:

python clf-train-registry.py clf-model "http://localhost:8000" 
--outputTestData test.csv

Below is a code snippet from that script, which aliases the latest model to Staging.

Serve the newly transitioned Staging model to port 1234:

mlflow models serve -m models:/clf-model@Staging -p 1234 -h 0.0.0.0 --env-manager local

You can then make predictions by running the following script with a csv of test data:

./predict.sh test.csv

which again just runs the following curl command:

curl http://localhost:1234/invocations 
-H 'Content-Type: text/csv' --data-binary @test.csv

And that’s it!

A couple of things to note:

    If you run this tutorial with Docker Compose, the containers will access and write to your local mounted directory.
    You need to make sure that you the environment / version of sklearn when loading a model via the Python API as the same version as the one you used to save it (which is why I prefer to do everything in Docker).
    You can clean up running containers with docker compose down.

Hopefully this was a good introduction to productionizing a machine learning model. By swapping out the example training script, you should be able to adapt this tutorial for use with your own models.

In a follow-up post, I’ll walk through building your own custom mlflow model so that you can serve pretty much any function you like over a REST api. Stay tuned!