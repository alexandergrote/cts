import mlflow
import os

from mlflow import MlflowClient
from pathlib import Path
from urllib.parse import urlparse

from src.util.constants import Directory

def get_mlflow_client() -> MlflowClient:

    tracking_uri = Directory.ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    return client


def uri_to_path(uri: str) -> Path:

    parsed_uri_path = urlparse(uri).path

    if os.name == "nt":
        # convert / slashes to \ slashes
        # trim first two \ slashes
        parsed_uri_path = parsed_uri_path.replace('/', '\\')[1:]

    return Path(parsed_uri_path)