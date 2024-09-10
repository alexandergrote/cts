import typer
import concurrent.futures
import subprocess
import pandas as pd

from typing import Optional, List

from src.util.custom_logging import console
from src.experiments.analysis.feat_selection import FeatureSelection
from src.fetch_data.mlflow_engine import QueryEngine
from src.experiments.main import Experiment


def get_experiment_configs() -> List[Experiment]:

    experiments = Experiment.create_feature_selection_experiments()

    return experiments


def run_process(command):
    """
    Run a process and wait for its execution.

    Args:
        command (str): The command to execute.

    Returns:
        subprocess.CompletedProcess: The result of the process execution.
    """
    return subprocess.run(command, shell=True, check=True)
    

def run_processes_in_parallel(commands, workers):
    """
    Run multiple processes in parallel and wait for their execution.

    Args:
        commands (list): A list of commands to execute.

    Returns:
        list: A list of subprocess.CompletedProcess objects.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_process, command) for command in commands]
        results = [future.result() for future in futures]

    return results
    

def execute(experiment_name: Optional[str] = None, filter_name: Optional[str] = None, run_in_parallel: bool = False, skip_execution: bool = False):

    experiments = get_experiment_configs()

    experiments_copy = experiments.copy()

    mlflow_engine = QueryEngine()

    analyser = FeatureSelection()   

    if experiment_name is not None:
        experiments_copy = [experiment for experiment in experiments_copy if experiment.name == experiment_name]

    if filter_name is not None:
        experiments_copy = [experiment for experiment in experiments_copy if filter_name in experiment.name]

    # ask for confirmation if you want to run these experiments
    if len(experiments_copy) == 0:
        return

        
    tmp_names = '\n'.join([el.name for el in experiments_copy])
    console.print(f"You are about to run these experiments:\n{tmp_names}")

    console.print("Press y to continue or n to abort.")
    answer = input("y/n: ")

    if answer!= "y":
        console.print("Aborting...")
        return
    
    if not skip_execution:

        if run_in_parallel:
            
            commands = [exp.command for exp in experiments_copy]
            run_processes_in_parallel(commands, workers=4)
        
        else:

            for experiment in experiments_copy:
                experiment.run()

    console.rule("Get aggregated results of experiment runs")

    datasets = pd.concat(
        [mlflow_engine.get_results_of_single_experiment(experiment_name=experiment.name, n=100) for experiment in experiments_copy]
        )

    analyser.analyse(data=datasets)

if __name__ == "__main__":
    typer.run(execute)

    