python src\main.py ^
fetch_data=synthetic ^
preprocess=synthetic__boruta preprocess.params.steps.3.params.random_state=%2 preprocess.params.steps.2.params.n_features=%3 ^
model=%1 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="clf__synthetic__boruta"