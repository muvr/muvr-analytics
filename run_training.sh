#!/bin/bash

usage()
{
python mlp/start_training.py -h
}

DATASET="dataset/*.zip"
TEST_FOLDER=
OUTPUT="output"
MODEL_NAME="demo"
IS_ANALYSIS=
EPOCH=10

while getopts "hd:o:t:m:l:a" OPTION
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
         l)
            EPOCH=$OPTARG
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

VISUAL="$OUTPUT/visualisation.png"
EVAL="$OUTPUT/evaluation.csv"
LOG_FILE="$OUTPUT/training_log"

printf "\n\nSTART TRAINING & EVALUATION with parameter:\n\tDataset: %s\n\tTest: %s\n\tOutput: %s\n\tModel: %s\n\n"  "$DATASET" "$TEST_FOLDER" "$OUTPUT" "$MODEL_NAME"

if ! [ -z $IS_ANALYSIS ]
then
    python mlp/start_training.py -d $DATASET -o $OUTPUT -e $EVAL -v $VISUAL -m $MODEL_NAME -loop $EPOCH -analysis | tee $LOG_FILE
else
    remove_output
    if [ -z $TEST_FOLDER ]
    then
        python mlp/start_training.py -d $DATASET -o $OUTPUT -e $EVAL -v $VISUAL -m $MODEL_NAME -loop $EPOCH | tee $LOG_FILE
    else
        python mlp/start_training.py -d $DATASET -o $OUTPUT -e $EVAL -v $VISUAL -t $TEST_FOLDER -m $MODEL_NAME -loop $EPOCH | tee $LOG_FILE
    fi
    EXIT_CODE=$?
    if [[ $EXIT_CODE != 0 ]]
    then
        exit $EXIT_CODE
    else
#        open $OUTPUT/visualisation.png
        column -s, -t < $OUTPUT/evaluation.csv
    fi
fi
