import mlflow
import os

import pandas as pd

from mlflow import MlflowClient
from pathlib import Path
from urllib.parse import urlparse

from src.util.constants import Directory

def get_mlflow_client() -> MlflowClient:

    tracking_uri = Directory.ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    return client


def get_tracking_uri():

    if os.name == 'nt':
        return Path(rf"file:\\{str(Directory.ROOT)}\mlruns")
    
    return Path(rf"file:{str(Directory.ROOT)}/mlruns")


def uri_to_path(uri: str) -> Path:

    parsed_uri_path = urlparse(uri).path

    if os.name == "nt":
        # convert / slashes to \ slashes
        # trim first two \ slashes
        parsed_uri_path = parsed_uri_path.replace('/', '\\')[1:]

    return Path(parsed_uri_path)


def get_last_n_runs(experiment_id: str, n: int, query: str):

    tracking_uri = Directory.ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    runs = MlflowClient().search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attribute.start_time DESC"],
        max_results=n,
    )

    return runs


def runs_to_df(runs):

    metrics = [pd.DataFrame(run.data.metrics, index=[0]) for run in runs]
    params = [pd.DataFrame(run.data.params, index=[0]) for run in runs]
    meta = [
        pd.DataFrame(
            {
                "end_time": run.info.end_time,
                "run_id": run.info.run_id,
                "artifact_uri": run.info.artifact_uri,
            },
            index=[0],
        )
        for run in runs
    ]

    metrics_df = pd.concat(metrics, ignore_index=True)
    params_df = pd.concat(params, ignore_index=True)
    meta_df = pd.concat(meta, ignore_index=True)

    return pd.concat([metrics_df, params_df, meta_df], axis=1)



if __name__ == "__main__":

    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--query', type=str)
    parser.add_argument('--n_runs', type=int, default=5)

    args = parser.parse_args()

    tracking_uri = Directory.ROOT / "mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiments = client.search_experiments(filter_string=f"name ILIKE '{args.query}'")

    config = {}

    for experiment in experiments:

        config[experiment.name] = {
            "id": experiment.experiment_id,
            "runs": args.n_runs,
            "query": "",
        }

    with open(Path(__file__).parent / f"{args.filename}.yaml", "w") as file:
        yaml.dump(config, file)