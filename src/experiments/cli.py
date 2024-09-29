import typer
import concurrent.futures
import subprocess
import re
import os
import pandas as pd

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

from src.util.custom_logging import console
from src.experiments.analysis.base import BaseAnalyser
from src.experiments.analysis.feat_selection import FeatureSelection
from src.experiments.analysis.cost_benefit import CostBenefit
from src.experiments.analysis.correlations import Correlations
from src.fetch_data.mlflow_engine import QueryEngine
from src.experiments.util.factory import ExperimentFactory
from src.experiments.util.types import Experiment
from src.util.caching import environ_pickle_cache

mlflow_engine = QueryEngine()


class ExperimentRunner(BaseModel):

    analyser: BaseAnalyser = Field(frozen=True)
    experiments: List[Experiment]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def experiments(self):
        return self._experiments
    
    @experiments.setter
    def experiments(self, experiments):

        if not isinstance(experiments, list):
            raise TypeError("The experiments parameter must be a list.")
        
        if len(experiments) == 0:
            raise ValueError("The experiments parameter must not be empty.")

        self._experiments = experiments


    @staticmethod
    def run_process(command):
        """
        Run a process and wait for its execution.

        Args:
            command (str): The command to execute.

        Returns:
            subprocess.CompletedProcess: The result of the process execution.
        """

        # get output from command
        try:
            output = subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e.stderr.decode('utf-8')}")

        
        return output


    @staticmethod
    def run_processes_in_parallel(commands, workers):
        """
        Run multiple processes in parallel and wait for their execution.

        Args:
            commands (list): A list of commands to execute.

        Returns:
            list: A list of subprocess.CompletedProcess objects.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(ExperimentRunner.run_process, command) for command in commands]
            results = [future.result() for future in futures]

        return results

    @staticmethod
    @environ_pickle_cache()
    def get_experiment_data_from_mlflow(experiments: List[str]):

        data = pd.concat(
            [mlflow_engine.get_results_of_single_experiment(experiment_name=el, n=100) for el in experiments]
        )

        return data


    def run(self, run_in_parallel: bool = False, skip_execution: bool = False, skip_visualization: bool = False):

        if not skip_execution:

            if run_in_parallel:
                
                commands = [exp.command for exp in self.experiments]
                ExperimentRunner.run_processes_in_parallel(commands, workers=4)
            
            else:

                for experiment in self.experiments:
                    ExperimentRunner.run_process(experiment.command)

        if skip_visualization:
            return

        console.rule("Get aggregated results of experiment runs")

        experiments = [el.name for el in self.experiments]

        datasets = ExperimentRunner.get_experiment_data_from_mlflow(experiments=experiments)

        self.analyser.analyse(data=datasets)

    @classmethod
    def get_all_runners(cls) -> List["ExperimentRunner"]:

        executeable_list = []

        combinations = [
            (ExperimentFactory.create_feature_selection_experiments(), FeatureSelection()),
            (ExperimentFactory.create_cost_benefit_experiments(), CostBenefit()),
            (ExperimentFactory.create_correlation_experiments(), Correlations())
        ]

        for experiments, analyser in combinations:

            if len(experiments) == 0:
                raise ValueError("No experiments found.")
            
            runner = cls(
                analyser=analyser, 
                experiments=experiments
            )

            executeable_list.append(runner)

        return executeable_list    
    

def execute(analyser: str, filter_name: Optional[str] = None, run_in_parallel: bool = False, skip_execution: bool = False, skip_visualization: bool = False, cache_mlflow: bool = False):

    if cache_mlflow:
        os.environ["src.experiments.cli.py.ExperimentRunner.get_experiment_data_from_mlflow"] = "cached"

    experiments = ExperimentRunner.get_all_runners()

    # filter experiments by analyser by checking if provided str, potentially a regex expression, matches only one of the analysers    el = experiments[0]
    
    experiments = [el if re.match(analyser, el.analyser.__class__.__name__, flags=re.IGNORECASE) else None for el in experiments]

    experiments = list(filter(None, experiments))

    if len(experiments) > 1:
        console.print(f"More than one analyser matches the provided string: {analyser}")
        return
    
    if len(experiments) == 0:
        console.print(f"No analyser matches the provided string: {analyser}")
        return
    
    experiment_analyser = experiments[0]
       
    # filter experiments by name by checking if provided str, potentially a regex expression, matches only one of the experiments
    if filter_name is not None:

        selected_exp = [el if re.match(filter_name, el.name, flags=re.IGNORECASE) else None for el in experiment_analyser.experiments]
        selected_exp = list(filter(None, selected_exp))

        if len(selected_exp) == 0:
            console.print(f"No experiment matches the provided string: {filter_name}")
            return

        experiment_analyser.experiments = selected_exp

    if len(experiment_analyser.experiments) > 1:
        
        tmp_names = '\n'.join([el.name for el in experiment_analyser.experiments])
        console.print(f"You are about to run these experiments:\n{tmp_names}")

        console.print("Press y to continue or n to abort.")
        answer = input("y/n: ")

        if answer!= "y":
            console.print("Aborting...")
            return
    
    experiment_analyser.run(run_in_parallel=run_in_parallel, skip_execution=skip_execution, skip_visualization=skip_visualization)
    

if __name__ == "__main__":
    typer.run(execute)

    