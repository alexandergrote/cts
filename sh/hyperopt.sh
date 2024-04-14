#!/bin/bash

for case in "malware" "churn" "synthetic"
do
    vocab_size=2

    if [ $case = "synthetic" ]
    then 
        vocab_size=16
    elif [ $case = "malware" ]
    then
        vocab_size=260
    elif [ $case = "churn" ]
    then vocab_size=6
    fi
    
    python src/main.py \
    constants=$case \
    fetch_data=$case \
    preprocess=baseline_lstm  \
    train_test_split=stratified train_test_split.params.random_state=0 \
    model=lstm_tuned model.params.model.params.model.params.vocab_size=$vocab_size \
    evaluation=ml.yaml \
    export=mlflow.yaml export.params.experiment_name="hyper_opt__"$case"__"lstm

done

