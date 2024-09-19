import typer
import concurrent.futures
import subprocess
import re
import pandas as pd

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List

from src.util.custom_logging import console
from src.experiments.analysis.base import BaseAnalyser
from src.experiments.analysis.feat_selection import FeatureSelection
from src.fetch_data.mlflow_engine import QueryEngine
from src.experiments.main import Experiment, ExperimentFactory

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
        return subprocess.run(command, shell=True, check=True)


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

    def run(self, run_in_parallel: bool = False, skip_execution: bool = False):

        if not skip_execution:

            if run_in_parallel:
                
                commands = [exp.command for exp in self.experiments]
                ExperimentRunner.run_processes_in_parallel(commands, workers=4)
            
            else:

                for experiment in self.experiments:
                    experiment.run()

        console.rule("Get aggregated results of experiment runs")

        datasets = pd.concat(
            [mlflow_engine.get_results_of_single_experiment(experiment_name=experiment.name, n=100) for experiment in self.experiments]
            )

        self.analyser.analyse(data=datasets)

    @classmethod
    def get_all_runners(cls) -> List["ExperimentRunner"]:

        executeable_list = []

        experiments = ExperimentFactory.create_feature_selection_experiments()

        if len(experiments) == 0:
            raise ValueError("No experiments found.")
        
        runner_feat_selection = cls(
            analyser=FeatureSelection(), 
            experiments=experiments
        )

        executeable_list.append(runner_feat_selection)

        return executeable_list    
    

def execute(analyser: str, filter_name: Optional[str] = None, run_in_parallel: bool = False, skip_execution: bool = False):

    experiments = ExperimentRunner.get_all_runners()

    # filter experiments by analyser by checking if provided str, potentially a regex expression, matches only one of the analysers    el = experiments[0]
    
    experiments = [el if re.match(analyser, el.analyser.__class__.__name__, flags=re.IGNORECASE) else None for el in experiments]

    if len(list(filter(None, experiments))) > 1:
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

        
    tmp_names = '\n'.join([el.name for el in experiment_analyser.experiments])
    console.print(f"You are about to run these experiments:\n{tmp_names}")

    console.print("Press y to continue or n to abort.")
    answer = input("y/n: ")

    if answer!= "y":
        console.print("Aborting...")
        return
    
    experiment_analyser.run(run_in_parallel=run_in_parallel, skip_execution=skip_execution)
    

if __name__ == "__main__":
    typer.run(execute)

    