#!/bin/bash

# 49c30c1
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

# The directory where spark-ec2 script lives
SPARK_EC2_HOME=$DIR/../../spark/ec2

[ -d $SPARK_EC2_HOME ] || die "Missing spark clone. Clone github.com:apache/spark.git in $DIR/../../."

# Where to locate the training data from
TRAINING_DATA_DIR="muvr-open-training-data"

# AMI Id for the spark cluster machines
AWS_AMI_ID=ami-7b14b208

# AWS instance type for the spark cluster slave machines
AWS_INSTANCE_TYPE=c4.2xlarge

# AWS instance type for the spark cluster master machine
AWS_MASTER_INSTANCE_TYPE=c4.2xlarge

export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

echo "Attempting to create Spark cluster"
SLAVE_COUNT=10
$SPARK_EC2_HOME/spark-ec2 -k scalax-jan-test -i $AWS_KEY_PAIR --master-instance-type=$AWS_MASTER_INSTANCE_TYPE --instance-type=$AWS_INSTANCE_TYPE --ami=$AWS_AMI_ID --region=eu-west-1 --copy-aws-credentials -s $SLAVE_COUNT launch scalax-spark-cluster
