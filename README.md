# Finding the Needle in the Haystack - An Interpretable Sequential Pattern Mining Method for Classification Problems

This repository is associated with the paper "Finding the Needle in the Haystack - An Interpretable Sequential Pattern Mining Method for Classification Problems", which is currently under review.

Kindly be aware that the code is subject to change as the paper is under review.

## Explaining the Code Structure and High-Level Usage

![Repository structure](/static/images/ml2mlflow2vis.png)

Broadly speaking, the repository contains two pipelines that serve different purposes:
1) Executing the Machine Learning experiments and storing the result of each run in a MlFlow database
2) Afterwards, we can query the results from the MlFlow database, analyse and visualize them.

You can run both pipelines together or indepently. This happens by means of `python src/experiments/cli.py <pipeline_name>`. You also have the possibility to skip either the execution or anaylsis/visualization of the pipelines:

- Only Machine Learning Experiment: `python src/experiments/cli.py <pipeline_name> --skip-visualization`  
- Only Analysis / Visualization: `python src/experiments/cli.py <pipeline_name> --skip-execution` 

Overall, there are four different setups, as defined in `src/experiments/factory.py`:
1) Correlation Analysis
2) Feature Extraction 
3) Cost Sensitivity Analysis 
4) Sensitivity Analysis
5) Benchmarking

### Machine Learning Pipeline

This pipeline executes machine learning experiments using different models and configurations. The results are stored in an MlFlow database. Its steps are as follows:

1) Fetch data `src/fetch_data` with interface defined in `src/fetch_data/base.py`
2) Preprocess data `src/preprocess` with interface defined in `src/preprocess/base.py`
3) Train models `src/models` with interface defined in `src/models/base.py`
4) Evaluate models `src/evaluation` with interface defined in `src/evaluation/base.py`
5) Export results `src/export` with interface defined in `src/export/base.py`


### Analysis / Visualization Pipeline

The different analysing scsripts including the visualizations are located in `src/experiments/analysis` and are named accordingly. 

## Usage 

### Installation

To use this package, you need to install the required packages and have a version of Python 3 installed. The python package has been tested with Python 3.9.6.

First, clone the repository:

`git clone git@github.com:alexandergrote/cts.git`

Next, navigate to the repository directory and install the package using pip:

- `pip install -r requirements.txt`
- `pip install .`


### Datasets

The synthetic dataset is automatically generated. However, for the remaining datasets you need to go to the following links and download them manually. Afterwards, put the raw csv files in a folder called `data` in the root directory of this project.
- Malware data:
    - original data source: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-api-call-sequences
    - download link: https://1drv.ms/x/c/e6dfa373b2b71977/Ed1rLpy85jRFomJBcn0R_3ABnggQNxukhExnVf89mGhjOw?e=7dOKTr
- Churn data: 
    - original data:  https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information
    - download link: https://1drv.ms/x/c/e6dfa373b2b71977/Ee5LlWR0rwxFtTxGPzCt6loBFZkqO9rC3JqQFinHttZM-A?e=zGmu5a

### Commands

To reproduce all experiments, including the figures and tables, you need to execute these function calls, which correspond to the four different setups mentioned above. These function calls are:
- `python src/experiments/cli.py corr`
- `python src/experiments/cli.py feat --filter-name "spm.*"` 
- `python src/experiments/cli.py feat --filter-name "oh.*"`
- `python src/experiments/cli.py cost`
- `python src/experiments/cli.py sens`
- `python src/experiments/cli.py bench`

The resulting figures are then saved in a folder called `figures`. The output of the exploratory data analysis can be obtained by running
- `python src/fetch_data/util/eda.py --dataset synthetic`
- `python src/fetch_data/util/eda.py --dataset malware`
- `python src/fetch_data/util/eda.py --dataset churn`

### Special Files of Interest

- Feature Extraction and Selection with Delta Confidence: `src/preprocess/extraction/ts_features.py`
- Feature Extraction with PrefixSpan: `src/preprocess/extraction/spm.py`

## Tests

To run unit tests, simply execute: ``python -m unittest``

- If you wish to execute only the unit tests and not the integration tests, you can execute: ``python -m unittest discover tests/unit``
- If you wish to execute only the integration tests and not the unit tests, you can execute: ``python -m unittest discover tests/integration``

## Additional Remarks for Source Code Usage

This repository contains some opionated snippets of code. For instance, it uses `hydra` for configuration management, ``mlflow`` to keep track of the machine learning runs and a custom cli interface for administering the different pipelines. With these remarks, we will hopefully make it easier for someone new to use this codebase.

### Hydra

First and foremost, the main entry script for each machine learning run is `src/main.py`. You can override the default configuration by passing in command line arguments. For example, to run the churn prediction pipeline with a different dataset, you can use the following command: ` python src/main.py fetch_data=churn`. For more information see the official [hydra documentation](https://hydra.cc/docs/intro/). 

### Mlflow

Start the mlflow gui with 

```
mlflow ui --port 5000
```
and inspect your results visually.