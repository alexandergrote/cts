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

        #bool_cases = any(experiment.name.startswith(exp) for exp in ['tmp', 'churn', 'synthetic', 'malware', 'spm', 'cts'])
        bool_cases = any((exp in experiment.name) * ('spm' in experiment.name) for exp in ['mrmr'])

        if not bool_cases:
            continue

        print(f"Deleting experiment ID: {experiment.name}")
        # Delete the experiment by ID
        mlflow.delete_experiment(exp_id)

if __name__ == "__main__":
    delete_all_experiments()