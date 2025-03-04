import mlflow
from typing import Optional, List
from pathlib import Path

from src.util.mlflow_util import get_tracking_uri, get_last_n_runs, runs_to_df

metrics = ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score']


def experiment_exists(experiment_name: str, random_seed: int) -> bool:

    tracking_uri = get_tracking_uri()

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
    
    show_results(experiment.experiment_id)

    return True


def show_results(experiment_id: str, metric_col_names: List[str] = metrics) -> Optional[Path]:

    runs = get_last_n_runs(
        experiment_id=experiment_id, n=1, query=''
    )

    data = runs_to_df(runs)

    print(data[metric_col_names])

if __name__ == "__main__":

    experiment_exists(
        case='synthetic',
        feat_select='rf',
        model='logistic_regression',
        n_features=None,
        random_seed=42
    )