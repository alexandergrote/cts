@echo off

REM example call: my\path\feat_selection.bat malware

echo --- case %1 ---

if %1==malware (
    
    python src\main.py ^
    constants=%1 ^
    fetch_data=%1 +fetch_data.params.resampling_fraction=1 ^
    preprocess=cts preprocess.params.selector.params.n_features=null preprocess.params.extractor.params.corr_threshold=1 ^
    train_test_split=stratified ^
    model=xgb ^
    evaluation=ml.yaml ^
    export=mlflow.yaml export.params.experiment_name=%1"__correlation"
)

python src\main.py ^
constants=%1 ^
fetch_data=%1 ^
preprocess=cts preprocess.params.selector.params.n_features=null preprocess.params.extractor.params.corr_threshold=1 ^
train_test_split=stratified ^
model=xgb ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name=%1"__correlation"
