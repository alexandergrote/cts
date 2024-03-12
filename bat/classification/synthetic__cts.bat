echo --- model %2 and random_state: %1 ---

python src\main.py ^
fetch_data=synthetic ^
preprocess=synthetic preprocess.params.steps.2.params.random_state=%1 ^
model=%2 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="clf__synthetic__cts"