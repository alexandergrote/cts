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

    for n in 0 1 2 3 4
    do

        python src/main.py \
        constants=$case \
        fetch_data=$case \
        preprocess=baseline_lstm  \
        train_test_split=stratified train_test_split.params.random_state=$n \
        model=lstm_tuned__$case model.params.model.params.vocab_size=$vocab_size \
        evaluation=ml.yaml \
        export=mlflow.yaml export.params.experiment_name="benchmark__"$case"__"lstm

    done

done

