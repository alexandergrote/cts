import mlflow
from typing import Optional
from pathlib import Path

from src.util.constants import Directory


def experiment_exists(experiment_name: str, random_seed: int) -> bool:

    tracking_uri = Path(rf"file:\\{str(Directory.ROOT)}\mlruns")
    mlflow.set_tracking_uri(str(tracking_uri))

    # filter string
    # synthetic__feat_selection__cts__model__logistic_regression__features__null
    filter_string = f"name='{experiment_name}'"

    # get experiment data
    experiments = mlflow.MlflowClient().search_experiments(
        filter_string=filter_string
    )

    if len(experiments) == 0:
        return False
    
    if len(experiments) > 1:
        raise ValueError("More than one experiment found")

    experiment = experiments[0]

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    if runs.empty:
        return False

    random_seeds = runs['params.train_test_split.params.random_state'].to_list()

    if str(random_seed) not in random_seeds:
        return False

    return True
    

if __name__ == "__main__":

    experiment_exists(
        case='synthetic',
        feat_select='rf',
        model='logistic_regression',
        n_features=None,
        random_seed=42
    )