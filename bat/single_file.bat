@echo off

REM example call: my\path\single_file.bat malware baseline xgb 0 null

echo --- case %1, feature_selection %2, model %3, random_state %4 and features %5 ---

set theilu= 

if %1==churn if %2==cts (
    set theilu=preprocess.params.extractor.params.corr_threshold=0
) 

python src\main.py ^
constants=%1 ^
fetch_data=%1 ^
preprocess=%2 preprocess.params.selector.params.n_features=%5 %theilu% ^
train_test_split=stratified train_test_split.params.random_state=%4 ^
model=%3 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name=%1"__feat_selection__"%2"__model__"%3"__features__"%5