#!/bin/bash

MODELS=( arms arms arms arms arms )
TRAINING_DATA_DIR="/Users/janmachacek/Muvr/muvr-open-training-data"
COMMON_PARAMS="--master local --train-path $TRAINING_DATA_DIR/train --test-path $TRAINING_DATA_DIR/test --output-path $TRAINING_DATA_DIR/test"

for M in "${MODELS[@]}"; do
    PARAMS="$COMMON_PARAMS --model $M"
    echo $PARAMS
    spark-submit --class io.muvr.em.ModelTrainerMain --driver-memory 4G target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar $PARAMS &
    spark-submit --class io.muvr.em.ModelTrainerMain --driver-memory 4G target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar $PARAMS --slacking=true &
done
