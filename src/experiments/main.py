import typer

from typing import Optional, List

from src.util.custom_logging import console
from src.experiments.analysis.feat_selection import FeatureSelection
from src.fetch_data.mlflow_engine import QueryEngine
from src.experiments import Experiment
from src.util.constants import Directory


def get_experiment_configs() -> List[Experiment]:

    experiments = Experiment.create_feature_selection_experiments()

    return experiments
    

def execute(experiment_name: Optional[str] = None):

    experiments = get_experiment_configs()

    experiments_copy = experiments.copy()

    mlflow_engine = QueryEngine()

    analyser = FeatureSelection()   

    if experiment_name is not None:
        experiments_copy = [experiment for experiment in experiments_copy if experiment.name == experiment_name]

    for experiment in experiments_copy:
        experiment.run()

    console.rule("Get aggregated results of experiment runs")

    for experiment in experiments_copy:

        console.log(f"Analyse experiment: {experiment.name}")

        data = mlflow_engine.get_results_of_single_experiment(experiment_name=experiment.name, n=100)
        
        analyser.analyse(data=data)

if __name__ == "__main__":
    typer.run(execute)

    