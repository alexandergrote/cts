# Interpretable Process Analysis Through Sequential Pattern Mining

## Installation

```
pip install -r requirements.txt
pip install .
```

To obtain the data for the two real-world datasts, you need to go to the following links:
- malware data: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences
- churn data: https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information

## Structure

![Repository structure](/static/images/ml2mlflow2vis.png)

Broadly speaking, the repository contains two pipelines that serve different purposes:
1) Executing the Machine Learning experiments and storing the result of each run in a MlFlow database
2) Afterwards, we can query the results and visualize them.

You can run both pipelines together or indepently. This happens by means of `python src\experiments\cli.py <pipeline_name>`. You also have the possibility to skip either the execution or visualization of the pipelines:

- Only Machine Learning Experiment: `python src\experiments\cli.py <pipeline_name> --skip-visualization`  
- Only Visualization: `python src\experiments\cli.py <pipeline_name> --skip-execution` 

To reproduce all runs, you need to execute these function calls:
- python src/experiments/cli.py corr
- python src/experiments/cli.py feat --filter-name "spm.*" 
- python src/experiments/cli.py feat --filter-name "oh.*"
- python src/experiments/cli.py cost
- python src/experiments/cli.py sens 

## Additional Remarks for Source Code Usage

This repository contains some opionated snippets of code. For instance, it uses `hydra` for configuration management, ``mlflow`` to keep track of the machine learning runs and a custom cli interface for administering the different pipelines. With these remarks, we will hopefully make it easier for someone new to use this codebase.

### Hydra

First and foremost, the main entry script for each machine learning run is `src\main.py`. You can override the default configuration by passing in command line arguments. For example, to run the churn prediction pipeline with a different dataset, you can use the following command: ` python src\main.py fetch_data=churn`. For more information see the official [hydra documentation](https://hydra.cc/docs/intro/). 

### Mlflow

Start the mlflow gui with 

```
mlflow ui --port 5000
```
and inspect your results visually.