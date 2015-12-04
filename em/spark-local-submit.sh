#!/bin/bash

TRAINING_DATA_DIR="/Users/janmachacek/Muvr/muvr-open-training-data"
PARAMS="--master local --model arms --train-path $TRAINING_DATA_DIR/train --test-path $TRAINING_DATA_DIR/test --output-path s3n://AKIAIWGORVA3QVK54PMA:emJ7hW7dup1Jg5aCcYcevFE5AGXLtH5zrV0Ko3W+@muvr-open-training-data/models $@"
spark-submit --class io.muvr.em.ModelTrainerMain --driver-memory 4G target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar $PARAMS
