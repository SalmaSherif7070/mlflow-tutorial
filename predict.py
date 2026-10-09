"""Load the best logged model from MLflow and make predictions.

Run this after `train.py` has logged some runs. It asks the tracking server for
the run with the highest accuracy, loads that run's saved model, and predicts
the species for a few sample flowers.
"""

import sys

import mlflow
import pandas as pd
from sklearn.datasets import load_iris

# MLflow prints status lines with emoji; make sure the Windows console can show them.
sys.stdout.reconfigure(encoding="utf-8")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 1. Find the best run in the experiment (highest accuracy).
experiment = mlflow.get_experiment_by_name("iris-tutorial")
if experiment is None:
    sys.exit("No 'iris-tutorial' experiment found. Run `python train.py` first.")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)
if runs.empty:
    sys.exit("No runs found. Run `python train.py` first.")

best = runs.iloc[0]
run_id = best["run_id"]
print(
    f"Best run: {best['tags.mlflow.runName']} "
    f"(run_id={run_id[:8]}..., accuracy={best['metrics.accuracy']:.3f})"
)

# 2. Load that run's logged model.
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# 3. Predict on a few sample flowers.
iris = load_iris(as_frame=True)
samples = iris.data.iloc[[0, 50, 100]]  # one from each species
predictions = model.predict(samples)

label_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
results = samples.copy()
results["predicted"] = [label_names[p] for p in predictions]

print("\nPredictions:")
print(results.to_string(index=False))
