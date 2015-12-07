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
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# AWS key pair PEM path"
AWS_KEY_PAIR=$DIR/../../aws.pem
[ -f $AWS_KEY_PAIR ] || die "Missing $AWS_KEY_PAIR"

# The directory where spark-ec2 script lives
SPARK_EC2_HOME=$DIR/../../spark/ec2

[ -d $SPARK_EC2_HOME ] || die "Missing spark clone. Clone github.com:apache/spark.git in $DIR/../../."

$SPARK_EC2_HOME/spark-ec2 -k scalax-jan-test -i $AWS_KEY_PAIR --region=eu-west-1 destroy scalax-spark-cluster