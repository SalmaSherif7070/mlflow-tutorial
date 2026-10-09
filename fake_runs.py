"""Log fake runs with hard-coded metrics so the UI shows a spread of accuracies.

Unlike train.py (which trains real models that all score ~1.000 on easy Iris
data), this script just logs made-up numbers for a handful of pretend models.
It's purely for seeing varied metrics — good, mediocre, and bad — in the UI.
"""

import sys

import mlflow

# MLflow prints status lines with emoji; make sure the Windows console can show them.
sys.stdout.reconfigure(encoding="utf-8")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-fake-demo")

# Hard-coded fake models: name, params, and the metrics we want to show in the UI.
fake_models = [
    {"name": "logreg",        "params": {"model_type": "logistic_regression", "C": 1.0},
     "accuracy": 0.97, "f1_macro": 0.96, "loss": 0.12},
    {"name": "random_forest", "params": {"model_type": "random_forest", "n_estimators": 100},
     "accuracy": 0.93, "f1_macro": 0.92, "loss": 0.25},
    {"name": "decision_tree", "params": {"model_type": "decision_tree", "max_depth": 4},
     "accuracy": 0.85, "f1_macro": 0.83, "loss": 0.41},
    {"name": "knn",           "params": {"model_type": "knn", "n_neighbors": 5},
     "accuracy": 0.78, "f1_macro": 0.75, "loss": 0.58},
    {"name": "naive_bayes",   "params": {"model_type": "naive_bayes"},
     "accuracy": 0.64, "f1_macro": 0.60, "loss": 0.77},
    {"name": "dummy_baseline","params": {"model_type": "dummy", "strategy": "most_frequent"},
     "accuracy": 0.33, "f1_macro": 0.17, "loss": 1.10},
]

print(f"Logging {len(fake_models)} fake runs...")
for m in fake_models:
    with mlflow.start_run(run_name=m["name"]):
        mlflow.log_params(m["params"])
        mlflow.log_metric("accuracy", m["accuracy"])
        mlflow.log_metric("f1_macro", m["f1_macro"])
        mlflow.log_metric("loss", m["loss"])
        print(f"  {m['name']:14s} accuracy={m['accuracy']:.2f}  f1_macro={m['f1_macro']:.2f}")

print("\nDone. Open http://127.0.0.1:5000 and pick the 'iris-fake-demo' experiment.")
