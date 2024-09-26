import mlflow
import typer
from pathlib import Path
import re

from src.util.constants import Directory

app = typer.Typer()

@app.command()
def delete_experiments(pattern: str):
    """
    Delete MLflow experiments matching the given pattern.

    :param pattern: Regular expression pattern to search for in experiment names
    """
    tracking_uri = Path(rf"file:\\{str(Directory.ROOT)}\mlruns")
    mlflow.set_tracking_uri(str(tracking_uri))

    # Get a list of all experiments
    experiments = mlflow.MlflowClient().search_experiments()

    # Filter experiments based on the given pattern
    filtered_experiments = [experiment for experiment in experiments if re.search(pattern, experiment.name)]

    if not filtered_experiments:
        print(f"No experiments found matching the pattern: {pattern}")
        return
    
    # Print the list of experiments to be deleted
    print(f"The following experiments will be deleted:")
    for experiment in filtered_experiments:
        print(f"ID: {experiment.experiment_id}, Name: {experiment.name}")

    # Ask for user confirmation before deletion
    confirmation = input(f"Are you sure you want to delete {len(filtered_experiments)} experiments matching the pattern: {pattern}? (y/n): ")
    if confirmation.lower() != 'y':
        print("Deletion cancelled.")
        return

    # Iterate through the filtered experiments and delete them
    for experiment in filtered_experiments:
        exp_id = experiment.experiment_id
        print(f"Deleting experiment ID: {experiment.name}")
        mlflow.delete_experiment(exp_id)

if __name__ == "__main__":
    app()
