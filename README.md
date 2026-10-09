# MLflow Mini Tutorial — Iris Model

A tiny, end-to-end example: spin up a local MLflow tracking server, train a
simple Iris classifier, and view the results in the MLflow UI.

## What you'll learn

- How to run a local MLflow tracking **server** + **UI**
- How to log parameters, metrics, and a model from your training code
- How to log **several runs** and compare them in the browser
- How to **load a logged model back** and make predictions

## 1. Set up the environment

From this folder (`mlflow`), create and activate a virtual environment, then
install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Start the MLflow server + UI

Open a terminal and leave it running:

```powershell
mlflow server --host 127.0.0.1 --port 5000
```

This starts both the tracking server and the web UI. Open your browser at:

👉 **http://127.0.0.1:5000**

> It will look empty for now — that's expected. We haven't logged a run yet.
>
> MLflow stores run metadata in an `mlruns/` folder and artifacts locally by
> default. You can delete that folder anytime to start fresh.

## 3. Train the model (in a second terminal)

Keep the server running, open a **new** terminal in this folder, activate the
same venv, and run:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py
```

This trains **four** RandomForest models with different hyperparameters, so you
get four runs to compare. You should see output like:

```
Training 4 models...
  rf-shallow   accuracy=1.000  f1_macro=1.000
  rf-baseline  accuracy=1.000  f1_macro=1.000
  rf-deep      accuracy=1.000  f1_macro=1.000
  rf-wide      accuracy=1.000  f1_macro=1.000

Done. Open the UI at http://127.0.0.1:5000 and select the runs to Compare.
```

> Iris is an easy dataset, so all four may score 1.000 — that's expected. The
> point is to see multiple runs with different parameters show up in the UI.

## 4. Explore the UI

Refresh **http://127.0.0.1:5000**. Click the **iris-tutorial** experiment to see
all four runs in a table. Open any run (e.g. **rf-baseline**) to see:

- **Parameters** — `n_estimators`, `max_depth`, `random_state`
- **Metrics** — `accuracy`, `f1_macro`
- **Artifacts** — the saved `model/` folder

To compare: tick the checkboxes next to several runs, then click **Compare** to
see their parameters and metrics side by side.

## 5. Load the model and predict

With runs logged, load the best one back and make predictions:

```powershell
python predict.py
```

It asks the server for the run with the highest accuracy, loads that run's saved
model, and predicts the species of three sample flowers:

```
Best run: rf-wide (run_id=dda4ec37..., accuracy=1.000)

Predictions:
 sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  predicted
               5.1               3.5                1.4               0.2     setosa
               7.0               3.2                4.7               1.4 versicolor
               6.3               3.3                6.0               2.5  virginica
```

This is how you reuse a model after training: MLflow loads it from the run's
artifacts via a `runs:/<run_id>/model` URI — no need to retrain.

## 5b. See varied accuracies in the UI (fake demo)

Because Iris is easy, the real runs all score ~1.000. To see a visible spread of
good/mediocre/bad metrics in the UI, run the fake-run demo — it logs hard-coded
metrics for six pretend models (no real training):

```powershell
python fake_runs.py
```

This creates a separate **iris-fake-demo** experiment with accuracies ranging
from `0.33` (dummy baseline) to `0.97` (logistic regression). Open it in the UI,
switch the table to a **Chart** view, or **Compare** the runs to see the
differences clearly.

## 6. Experiment more (try it!)

Edit the `experiments` list in `train.py` (add configs or change `n_estimators`
/ `max_depth`), re-run `python train.py`, and watch the new runs appear in the
UI. Then run `python predict.py` again to load whichever run now scores best.

## That's it 🎉

You've run a local MLflow server, logged multiple experiment runs, compared them
in the UI, and loaded a logged model back for prediction — the core loop of
experiment tracking.
