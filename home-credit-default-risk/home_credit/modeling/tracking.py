import mlflow
import mlflow.sklearn

def start_run(run_name=None):
    mlflow.start_run(run_name=run_name)

def log_params(params):
    mlflow.log_params(params)

def log_metrics(metrics):
    mlflow.log_metrics(metrics)

def log_model(model, name):
    mlflow.sklearn.log_model(model, name)

def end_run():
    mlflow.end_run() 