FROM python:3.9-slim-bullseye

RUN pip install --no-cache mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--backend-store-uri", "postgresql+psycopg2://postgres@postgres:5432/mlflow", "--default-artifact-root", "./mlruns"]