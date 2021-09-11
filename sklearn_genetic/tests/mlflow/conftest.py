import os
import pytest
import mlflow
import logging
from testcontainers.compose import DockerCompose


@pytest.fixture(scope="module", autouse=True)
def setup_mlflow():
    path = os.path.abspath(os.path.dirname(__file__))
    with DockerCompose(
        path, compose_file_name=["docker-compose.yml"], pull=True
    ) as compose:
        host = compose.get_service_host("mlflow", 5000)
        if host == "0.0.0.0":
            host = "localhost"
        port = compose.get_service_port("mlflow", 5000)
        logging.info(f"MLflow started @http://{host}:{port}")
        mlflow.set_tracking_uri(f"http://{host}:{port}")
        yield
