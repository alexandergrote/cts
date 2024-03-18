import mlflow
from pathlib import Path

from src.util.constants import Directory


def delete_all_experiments():

    tracking_uri = Path(rf"file:\\{str(Directory.ROOT)}\mlruns")
    mlflow.set_tracking_uri(str(tracking_uri))

    # Get a list of all experiments
    experiments = mlflow.MlflowClient().search_experiments()
    
    # Iterate through all experiments
    for experiment in experiments:

        exp_id = experiment.experiment_id
        if exp_id == "0":
            continue

        # Delete the experiment by ID
        mlflow.delete_experiment(exp_id)
        print(f"Deleted experiment ID: {exp_id}")

if __name__ == "__main__":
    delete_all_experiments()