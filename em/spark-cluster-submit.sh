#!/bin/bash

# Where to locate the training data from
TRAINING_DATA_DIR="muvr-open-training-data"

# Absolute path of the spark JAR job
EM_JAR=target/scala-2.10/em-assembly-1.0.0-SNAPSHOT.jar

# Spark master public DNS
SPARK_MASTER=ec2-54-154-183-184.eu-west-1.compute.amazonaws.com

# AWS key pair PEM path"
AWS_KEY_PAIR=~/Downloads/scalax-jan-test.pem

# AWS account access key id
AWS_ACCESS_KEY_ID=AKIAIWGORVA3QVK54PMA

# AWS secret access key
AWS_SECRET_ACCESS_KEY=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

# Copy JAR with Spark job to Spark master
scp -i $AWS_KEY_PAIR $EM_JAR root@ec2-54-154-183-184.eu-west-1.compute.amazonaws.com:~/em.jar

# Sync the Spark job accorss all slaves
ssh -i $AWS_KEY_PAIR root@$SPARK_MASTER "/root/spark-ec2/copy-dir em.jar"

# Submit the spark job to the master
ssh -i $AWS_KEY_PAIR root@$SPARK_MASTER "/root/spark/bin/spark-submit --class io.muvr.em.ModelTrainerMain --deploy-mode=cluster --driver-memory 10G --driver-cores 36 /root/em.jar --model arms --train-path s3n://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$TRAINING_DATA_DIR/train --test-path s3n://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY@$TRAINING_DATA_DIR/test --output-path /tmp/models"