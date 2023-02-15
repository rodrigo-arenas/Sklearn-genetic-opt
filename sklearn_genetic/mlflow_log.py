import logging

# Check if mlflow is installed as an extra requirement
try:
    import mlflow
except ModuleNotFoundError:  # noqa
    logger = logging.getLogger(__name__)  # noqa
    logger.error("MLflow not found, pip install mlflow to use MLflowConfig")  # noqa


class MLflowConfig:
    """
    Logs each fit of hyperparameters in a running instance of mlflow: https://mlflow.org/
    """

    def __init__(
        self,
        tracking_uri,
        experiment,
        run_name,
        save_models=False,
        registry_uri=None,
        tags=None,
    ):
        """

        Parameters
        ----------
        tracking_uri: str
            Address of local or remote tracking server.
        experiment: str
            Case sensitive name of an experiment to be activated.
        run_name: str
            Name of new run (stored as a mlflow.runName tag).
        save_models: bool, default=False
            If ``True``, it will log the estimator into mlflow artifacts
        registry_uri: str, default=None
            Address of local or remote model registry server.
        tags: dict, default=None
            Dictionary of tag_name: String -> value.

        """
        self.client = mlflow.tracking.MlflowClient()
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.save_models = save_models
        self.tags = tags
        self.registry_uri = registry_uri

        mlflow.set_registry_uri(self.registry_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)

        self.experiment_id = mlflow.get_experiment_by_name(self.experiment).experiment_id

        if self.tags is not None:
            mlflow.set_tags(self.tags)

    def create_run(self, parameters, score, estimator):
        """
        Parameters
        ----------
        parameters: dict
            A dictionary with the keys as the hyperparameter name and the value as the current value setting
        score:
            The cross-validation score achieved by the current parameters
        estimator: estimator object
            The current sklearn estimator that is being fitted

        """

        with mlflow.start_run(
            experiment_id=self.experiment_id, nested=True, run_name=self.run_name
        ):
            for parameter, value in parameters.items():
                mlflow.log_param(key=parameter, value=value)

            mlflow.log_metric(key="score", value=score)

            if self.save_models:
                mlflow.sklearn.log_model(estimator, "model")
