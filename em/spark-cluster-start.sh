#!/bin/bash

# The directory where spark-ec2 script lives
SPARK_EC2_HOME=~/Minefield/spark/ec2

# Where to locate the training data from
TRAINING_DATA_DIR="muvr-open-training-data"

# AWS account access key id
AWS_ACCESS_KEY_ID=AKIAIWGORVA3QVK54PMA

# AWS secret access key
AWS_SECRET_ACCESS_KEY=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

# AWS key pair PEM path"
AWS_KEY_PAIR=~/Downloads/scalax-jan-test.pem

# AMI Id for the spark cluster machines
AWS_AMI_ID=ami-1942e46a

# AWS instance type for the spark cluster machines
AWS_INSTANCE_TYPE=c4.8xlarge

echo "Building the JAR file for the job..."
sbt clean assembly

echo "JAR for job is built!"

echo "Attempting to create Spark cluster"
$SPARK_EC2_HOME/spark-ec2 -k scalax-jan-test -i $AWS_KEY_PAIR --instance-type=$AWS_INSTANCE_TYPE --ami=$AWS_AMI_ID --region=eu-west-1 --copy-aws-credentials -s 1 launch scalax-spark-cluster
