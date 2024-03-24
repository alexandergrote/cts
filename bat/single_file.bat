@echo off

REM example call: my\path\single_file.bat malware baseline xgb 0 null

echo --- case %1, feature_selection %2, model %3, random_state %4, features %5 and spm %6 ---

set feat_algo=%2

if "%6"=="spm" (
    set feat_algo=%2_spm
) 

python src\main.py ^
constants=%1 ^
fetch_data=%1 ^
preprocess=%feat_algo% preprocess.params.selector.params.perc_features=%5 ^
train_test_split=stratified train_test_split.params.random_state=%4 ^
model=%3 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name=%1"__feat_selection__"%feat_algo%"__model__"%3"__features__"%5