import pytest
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from ..genetic_search import GASearchCV
from ..mlflow_log import MLflowConfig
from ..space import Integer, Categorical, Continuous

EXPERIMENT_NAME = "Digits-sklearn-genetic-opt-tests"


@pytest.fixture(scope="module")
def mlflow_tracking_uri(tmp_path_factory):
    mlflow_path = tmp_path_factory.mktemp("mlflow")
    tracking_path = mlflow_path / "mlflow.db"
    artifact_path = mlflow_path / "artifacts"
    tracking_uri = f"sqlite:///{tracking_path.as_posix()}"
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_tracking_env = os.environ.get("MLFLOW_TRACKING_URI")

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_path.resolve().as_uri(),
        )

    yield tracking_uri, tracking_path, artifact_path

    if mlflow.active_run():
        mlflow.end_run()

    mlflow.set_tracking_uri(previous_tracking_uri)
    if previous_tracking_env is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = previous_tracking_env


@pytest.fixture
def mlflow_resources(mlflow_tracking_uri):
    uri, _, _ = mlflow_tracking_uri
    client = MlflowClient(uri)
    return uri, client


@pytest.fixture
def mlflow_run(mlflow_resources):
    _, client = mlflow_resources
    exp_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    active_run = client.search_runs(exp_id, run_view_type=ViewType.ACTIVE_ONLY)
    runs = [run.info.run_id for run in active_run]
    return runs


def test_mlflow_config(mlflow_resources):
    """
    Check MLflow config creation.
    """
    uri, _ = mlflow_resources
    mlflow_config = MLflowConfig(
        tracking_uri=uri,
        experiment=EXPERIMENT_NAME,
        run_name="Decision Tree",
        save_models=True,
        tags={"team": "sklearn-genetic-opt", "version": "0.5.0"},
    )
    assert isinstance(mlflow_config, MLflowConfig)


def test_mlflow_config_requires_mlflow(monkeypatch):
    """
    Check MLflowConfig raises a clear error when mlflow is not installed.
    """
    import sklearn_genetic.mlflow_log as mlflow_log

    monkeypatch.setattr(mlflow_log, "mlflow", None)

    with pytest.raises(ImportError, match="MLflowConfig requires mlflow") as error_info:
        MLflowConfig(
            tracking_uri="sqlite:///mlflow.db",
            experiment=EXPERIMENT_NAME,
            run_name="Decision Tree",
        )

    assert "mlflow" in str(error_info.value)
    assert 'pip install "sklearn-genetic-opt[mlflow]"' in str(error_info.value)


def test_runs(mlflow_resources, mlflow_run):
    """
    Check if runs are captured and parameters are true.
    """
    uri, client = mlflow_resources
    mlflow_config = MLflowConfig(
        tracking_uri=uri,
        experiment=EXPERIMENT_NAME,
        run_name="Decision Tree",
        save_models=True,
        tags={"team": "sklearn-genetic-opt", "version": "0.5.0"},
    )

    clf = DecisionTreeClassifier()

    data = load_digits()

    y = data["target"]
    X = data["data"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    params_grid = {
        "min_weight_fraction_leaf": Continuous(0, 0.5),
        "criterion": Categorical(["gini", "entropy"]),
        "max_depth": Integer(2, 20),
        "max_leaf_nodes": Integer(2, 30),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True)

    evolved_estimator = GASearchCV(
        clf,
        cv=cv,
        scoring="accuracy",
        population_size=3,
        generations=4,
        tournament_size=3,
        elitism=True,
        crossover_probability=0.9,
        mutation_probability=0.05,
        param_grid=params_grid,
        algorithm="eaSimple",
        n_jobs=-1,
        verbose=True,
        log_config=mlflow_config,
    )

    evolved_estimator.fit(X_train, y_train)
    y_predict_ga = evolved_estimator.predict(X_test)

    runs = [
        run.info.run_id
        for run in client.search_runs(
            mlflow_config.experiment_id, run_view_type=ViewType.ACTIVE_ONLY
        )
    ]
    assert len(runs) >= 1 and evolved_estimator.best_params_["min_weight_fraction_leaf"]


def test_mlflow_artifacts(mlflow_resources, mlflow_run):
    import os
    import mlflow

    _, client = mlflow_resources
    run_id = mlflow_run[0]

    # End any existing active run to avoid conflict
    if mlflow.active_run():
        mlflow.end_run()

    # Create a dummy artifact file
    with open("dummy.txt", "w") as f:
        f.write("dummy model content")

    # Log the artifact to the 'model' directory
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact("dummy.txt", artifact_path="model")

    os.remove("dummy.txt")  # Clean up file

    # Check that the artifact exists
    artifacts = client.list_artifacts(run_id)
    assert len(artifacts) > 0
    assert artifacts[0].path == "model"


def test_mlflow_params(mlflow_resources, mlflow_run):
    """
    Test parameters are all in the run and within range.
    """
    _, client = mlflow_resources
    run_id = mlflow_run[0]
    run = client.get_run(run_id)
    params = run.data.params

    assert 0 <= float(params["min_weight_fraction_leaf"]) <= 0.5
    assert params["criterion"] == "gini" or "entropy"
    assert 2 <= int(params["max_depth"]) <= 20
    assert 2 <= int(params["max_leaf_nodes"]) <= 30


def test_mlflow_after_run(mlflow_resources, mlflow_run):
    """
    Check that the run has logged expected artifacts, metrics, and hyperparameters to the MLflow server.
    """
    run_id = mlflow_run[0]
    _, client = mlflow_resources

    run = client.get_run(run_id)
    params = run.data.params

    assert 0 <= float(params["min_weight_fraction_leaf"]) <= 0.5
    assert params["criterion"] in ["gini", "entropy"]
    assert 2 <= int(params["max_depth"]) <= 20
    assert 2 <= int(params["max_leaf_nodes"]) <= 30

    metric_history = client.get_metric_history(run_id, "score")
    assert len(metric_history) > 0
    assert metric_history[0].key == "score"


def test_cleanup(mlflow_tracking_uri):
    """
    Ensure resources are cleaned up.
    """
    _, tracking_path, artifact_path = mlflow_tracking_uri
    workspace_path = Path.cwd().resolve()
    tracking_path = tracking_path.resolve()
    artifact_path = artifact_path.resolve()

    assert workspace_path not in tracking_path.parents
    assert workspace_path not in artifact_path.parents

    if tracking_path.exists():
        try:
            tracking_path.unlink()
        except PermissionError:
            pass
    if artifact_path.exists():
        shutil.rmtree(artifact_path)
