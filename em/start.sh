#!/bin/bash

sbt clean assembly

TRAINING_DATA_DIR="/Users/janmachacek/Muvr/muvr-open-training-data"
PARAMS="--master local --model arms --train-path $TRAINING_DATA_DIR/train --test-path $TRAINING_DATA_DIR/test --output-path $TRAINING_DATA_DIR/models $@"
spark-submit --class io.muvr.em.ModelTrainerMain --driver-memory 4G target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar $PARAMS
