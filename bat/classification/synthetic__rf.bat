python src\main.py ^
fetch_data=synthetic ^
preprocess=synthetic__rf preprocess.params.steps.2.params.n_features=%3 ^
train_test_split=stratified__synthetic train_test_split.params.random_state=%2 ^
model=%1 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="clf__synthetic__rf"