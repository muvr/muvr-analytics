#!/bin/bash

usage()
{
python mlp/preprocess_data.py -h
}

DATASET="dataset/*.zip"
OUTPUT="output"
TRAIN_RATIO=80
IS_SLACKING=

while getopts "hd:o:r:s" OPTION
do
     case $OPTION in
         s)
             IS_SLACKING="true"
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
         r)
             TRAIN_RATIO=$OPTARG
             ;;
         ?)
             echo "Unsupported arguments"
             exit
             ;;
     esac
done

LOG_FILE="$OUTPUT/log"

printf "\n\nPreprocess the dataset with parameter:\n\tDataset: %s\n\tOutput: %s\n\tTrain ratio: %s\n\n"  "$DATASET" "$OUTPUT" "$TRAIN_RATIO"


if [ -z $IS_SLACKING ]
then
    python mlp/preprocess_data.py -d $DATASET -o $OUTPUT -ratio $TRAIN_RATIO | tee $LOG_FILE
else
    python mlp/preprocess_data.py -d $DATASET -o $OUTPUT -ratio $TRAIN_RATIO -slacking | tee $LOG_FILE
fi
