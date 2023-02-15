import pytest
import shutil
import os

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


@pytest.fixture
def mlflow_resources():
    uri = mlflow.get_tracking_uri()
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

    runs = mlflow_run
    assert len(runs) >= 1 and evolved_estimator.best_params_["min_weight_fraction_leaf"]


def test_mlflow_artifacts(mlflow_resources, mlflow_run):
    _, client = mlflow_resources
    run_id = mlflow_run[0]
    run = client.get_run(run_id)
    assert client.list_artifacts(run_id)[0].path == "model"


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
    Check the end of the runs are logged artifacts/metric/hyperparameters exists in the mlflow server
    """
    run_id = mlflow_run[0]
    mlflow.end_run()
    _, client = mlflow_resources
    run = client.get_run(run_id)
    params = run.data.params

    assert 0 <= float(params["min_weight_fraction_leaf"]) <= 0.5
    assert params["criterion"] == "gini" or "entropy"
    assert 2 <= int(params["max_depth"]) <= 20
    assert 2 <= int(params["max_leaf_nodes"]) <= 30
    assert client.get_metric_history(run_id, "score")[0].key == "score"


def test_cleanup():
    """
    Ensure resources are cleaned up.
    """
    shutil.rmtree("mlruns")
    assert "mlruns" not in os.listdir(os.getcwd())
