@echo off

for %%f in ("synthetic", "malware", "churn") do (

    for %%s in ("baseline", "cts") do (

        python src\main.py ^
        constants=%%~f ^
        fetch_data=%%~f ^
        preprocess=%%~s ^
        train_test_split=stratified train_test_split.params.random_state=0 ^
        model=xgb_tuned ^
        evaluation=ml.yaml ^
        export=mlflow.yaml export.params.experiment_name="hyper_opt__"%%~f"__"%%~s

    )

)

