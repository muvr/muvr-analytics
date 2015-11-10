#!/bin/bash

usage()
{
python mlp/start_training.py -h
}

DATASET="dataset/*.zip"
TEST_FOLDER=
OUTPUT="output"
MODEL_NAME="demo"

while getopts "hd:o:t:m:" OPTION
do
     case $OPTION in
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
         ?)
             echo "Unsupported arguments"
             exit
             ;;
     esac
done

rm -f $OUTPUT/*
mkdir -p $OUTPUT

VISUAL="$OUTPUT/visualisation.png"
EVAL="$OUTPUT/evaluation.csv"

printf "\n\nSTART TRAINING & EVALUATION\n\n"

if [ -z $TEST_FOLDER ]
then
    python mlp/start_training.py -d $DATASET -o $OUTPUT -e $EVAL -v $VISUAL -m $MODEL_NAME
else
    python mlp/start_training.py -d $DATASET -o $OUTPUT -e $EVAL -v $VISUAL -t $TEST_FOLDER -m $MODEL_NAME
fi

EXIT_CODE=$?
if [[ $EXIT_CODE != 0 ]]
then
    exit $EXIT_CODE
else
    open $OUTPUT/visualisation.png
    open $OUTPUT/evaluation.csv
fi
