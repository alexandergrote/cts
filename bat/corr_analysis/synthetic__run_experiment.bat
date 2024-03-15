python src\main.py ^
fetch_data=synthetic ^
preprocess=synthetic__cts ^
train_test_split=stratified__synthetic ^
model=xgb ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="synthetic__feat_selection"