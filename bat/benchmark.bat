@echo off

for %%f in ("synthetic", "malware", "churn") do (

    for %%s in ("baseline", "cts") do (

        for %%n in (0, 1, 2, 3, 4, 5) do (

            python src\main.py ^
            constants=%%~f ^
            fetch_data=%%~f ^
            preprocess=%%~s ^
            train_test_split=stratified train_test_split.params.random_state=%%n ^
            model=xgb_tuned__%%~s__%%~f ^
            evaluation=ml.yaml ^
            export=mlflow.yaml export.params.experiment_name="benchmark__"%%~f"__"%%~s

        )

    )

)