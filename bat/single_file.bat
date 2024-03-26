@echo off

REM example call: my\path\single_file.bat malware baseline xgb 0 null

echo --- case %1, feature_selection %2, model %3, random_state %4, features %5 and spm %6 ---

set identifier=

set feat_algo=%2

if "%6"=="spm" (
    set feat_algo=%2_spm
    set identifier=spm__
) 

set feat_select=

if "%6"=="feat_select" (

    set identifier=cts_comparison__
    set feat_algo=cts
    set feat_select_base=preprocess.params.selector.class_name=src.preprocess.selection.

    if "%2"=="rf" (
        set feat_select=%feat_select_base%random_forest.RFFeatSelection
    )

    if "%2"=="mutinfo" (
        set feat_select=%feat_select_base%mutual_info.MutInfoFeatSelection
    )

    if "%2"=="boruta" (
        set feat_select=%feat_select_base%boruta.BorutaFeatSelection
    )

    if "%2"=="cts" (
        set feat_select=%feat_select_base%mrmr.MRMRFeatSelection
    )

)


python src\main.py ^
constants=%1 ^
fetch_data=%1 ^
preprocess=%feat_algo% preprocess.params.selector.params.n_features=%5 %feat_select% ^
train_test_split=stratified train_test_split.params.random_state=%4 ^
model=%3 ^
evaluation=ml.yaml ^
export=mlflow.yaml export.params.experiment_name=%identifier%%1"__feat_selection__"%feat_algo%"__model__"%3"__features__"%5