import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# MLflow prints status lines with emoji; make sure the Windows console can show them.
sys.stdout.reconfigure(encoding="utf-8")

# Point this script at the local tracking server you started with `mlflow server`.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-tutorial")

# 1. Load data
iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. A few hyperparameter sets to try. Each one becomes its own run in the UI.
experiments = [
    {"name": "rf-shallow", "n_estimators": 10, "max_depth": 1},
    {"name": "rf-baseline", "n_estimators": 100, "max_depth": 3},
    {"name": "rf-deep", "n_estimators": 200, "max_depth": 5},
    {"name": "rf-wide", "n_estimators": 300, "max_depth": 8},
]


def train_one(run_name, n_estimators, max_depth):
    """Train a single model and log it as one MLflow run."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
    }

    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        # Log what you tried...
        mlflow.log_params(params)
        # ...and how it did.
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)

        # Save the trained model as an artifact (with an input example for the
        # signature). cloudpickle is the simplest serializer and avoids extra
        # trust prompts for tree-based models like RandomForest.
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test.iloc[:5],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        print(f"  {run_name:12s} accuracy={accuracy:.3f}  f1_macro={f1:.3f}")


# 3. Train + log every configuration
print(f"Training {len(experiments)} models...")
for exp in experiments:
    train_one(exp["name"], exp["n_estimators"], exp["max_depth"])

print("\nDone. Open the UI at http://127.0.0.1:5000 and select the runs to Compare.")
