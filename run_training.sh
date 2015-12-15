#!/bin/bash

usage()
{
python python-analytics/start_training.py -h
}

DATASET="dataset/*.zip"
TEST_FOLDER=
OUTPUT="output"
MODEL_NAME="demo"
IS_ANALYSIS=
YAML_FOLDER=

while getopts "hd:o:t:m:y:a" OPTION
do
     case $OPTION in
         a)
             IS_ANALYSIS="true"
             ;;
         h)
             usage
             exit 1
             ;;
         d)
             DATASET=$OPTARG
             ;;
         o)
             OUTPUT=$OPTARG
             ;;
         t)
             TEST_FOLDER=$OPTARG
             ;;
         m)
             MODEL_NAME=$OPTARG
             ;;
         y)
            YAML_FOLDER=$OPTARG
            ;;
         ?)
             echo "Unsupported arguments"
             exit
             ;;
     esac
done

remove_output()
{
rm -f $OUTPUT/*
mkdir -p $OUTPUT
}

LOG_FILE="$OUTPUT/training_log"

printf "\n\nSTART TRAINING & EVALUATION with parameter:\n\tDataset: %s\n\tTest: %s\n\tOutput: %s\n\tModel: %s\n\n"  "$DATASET" "$TEST_FOLDER" "$OUTPUT" "$MODEL_NAME"

TRAINING_CMD="python python-analytics/start_training.py -d \"$DATASET\" -o $OUTPUT -m $MODEL_NAME"
if ! [ -z $IS_ANALYSIS ]
then
    TRAINING_CMD="$TRAINING_CMD -analysis"
fi
if ! [ -z $TEST_FOLDER ]
then
    TRAINING_CMD="$TRAINING_CMD -t $TEST_FOLDER"
fi
if ! [ -z $YAML_FOLDER ]
then
    TRAINING_CMD="$TRAINING_CMD -yaml $YAML_FOLDER"
fi
TRAINING_CMD="$TRAINING_CMD | tee $LOG_FILE"

remove_output
eval $TRAINING_CMD

EXIT_CODE=$?
if [[ $EXIT_CODE != 0 ]]
then
    exit $EXIT_CODE
else
    open $OUTPUT/dataset_stats.csv
fi