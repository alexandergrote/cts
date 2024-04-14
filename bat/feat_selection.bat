@echo off

for %%f in ("synthetic", "malware", "churn") do (

    python src\main.py ^
    constants=%%~f ^
    fetch_data=%%~f ^
    preprocess=cts ^
    train_test_split=stratified ^
    model=xgb ^
    evaluation=ml.yaml ^
    export=mlflow.yaml export.params.experiment_name="feat_selection__"%%~f

)

