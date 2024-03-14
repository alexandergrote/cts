echo --- model %1 and random_state: %2 ---

python src\main.py ^
fetch_data=synthetic ^
preprocess=synthetic preprocess.params.steps.2.params.random_state=%2 ^
model=%1 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="clf__synthetic__cts"