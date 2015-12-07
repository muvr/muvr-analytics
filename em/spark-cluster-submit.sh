#!/bin/bash

# This script, which lives in muvr-analytics/em, must have the following files
# available (relative to muvr-analytics/em)
# ../../aws.pem
# ../../aws.profile

die () {
    echo >&2 "$@"
    exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DIR/../../aws.profile"

# AWS key pair PEM path"
AWS_KEY_PAIR=$DIR/../../aws.pem
[ -f $AWS_KEY_PAIR ] || die "Missing $AWS_KEY_PAIR"

# The models to train. The real Muvr system trains *different* models, this one only arms
MODELS=( arms arms arms arms arms )

# Where to locate the training data from
TRAINING_DATA_DIR="muvr-open-training-data"

# Absolute path of the spark JAR job
EM_JAR=$DIR/target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar

# Spark master public DNS
[ "$#" -eq 1 ] || die "Error: Spark Master public DNS required as the only arugment. $# argument(s) provided."
SPARK_MASTER=$1

# Copy JAR with Spark job to Spark master
scp -oStrictHostKeyChecking=no -i $AWS_KEY_PAIR $EM_JAR root@$SPARK_MASTER:~/em.jar

# Sync the Spark job accorss all slaves
ssh -oStrictHostKeyChecking=no -i $AWS_KEY_PAIR root@$SPARK_MASTER "/root/spark-ec2/copy-dir em.jar"

# Submit the spark job to the master
for M in "${MODELS[@]}"; do
  JOB="--class io.muvr.em.ModelTrainerMain --deploy-mode=cluster --driver-memory 6G --driver-cores 36 /root/em.jar"
  PARAMS="--model $M --train-path s3n://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$TRAINING_DATA_DIR/train --test-path s3n://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$TRAINING_DATA_DIR/test --output-path s3n://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$TRAINING_DATA_DIR/models"

  # submit the exercise $M job
  ssh -i $AWS_KEY_PAIR root@$SPARK_MASTER "/root/spark/bin/spark-submit $JOB $PARAMS"
  # submit the exercise vs. slacking for $M job
  ssh -i $AWS_KEY_PAIR root@$SPARK_MASTER "/root/spark/bin/spark-submit $JOB $PARAMS --slacking=true"
done
