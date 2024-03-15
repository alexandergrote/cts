python src\main.py ^
fetch_data=synthetic ^
preprocess=synthetic__cts ^
model=xgb ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="synthetic__feat_selection"