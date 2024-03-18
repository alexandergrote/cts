@echo off

REM example call: my\path\single_file.bat malware baseline xgb 0 null

echo --- case %1, feature_selection %2, model %3, random_state %4 and features %5 ---

python src\main.py ^
constants=%1 ^
fetch_data=%1 ^
preprocess=%2 preprocess.params.selector.params.n_features=%5 ^
train_test_split=stratified train_test_split.params.random_state=%4 ^
model=%3 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name="clf__"%1"__"%2