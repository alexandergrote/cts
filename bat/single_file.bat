@echo off

REM example call: my\path\single_file.bat malware baseline xgb 0 null

echo --- case %1, feature_selection %2, model %3, random_state %4, features %5 and encoding %6 ---

set feat_algo=%2_%6
set identifier=%6__

python src\main.py ^
constants=%1 ^
fetch_data=%1 ^
preprocess=%feat_algo% preprocess.params.selector.params.n_features=%5 %feat_select% ^
train_test_split=stratified train_test_split.params.random_state=%4 ^
model=%3 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name=%identifier%%1"__preprocess__"%2"__model__"%3"__features__"%5