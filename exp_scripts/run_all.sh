#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME=$1
if [ "$MODEL_NAME" != "gpt" ] && [ "$MODEL_NAME" != "mistral" ]; then
    echo "Error: Model name must be 'gpt' or 'mistral'"
    exit 2
fi

CURRENT_TIME=$(date +"%m%d%H%M")

LOG_FILE="${MODEL_NAME}_${CURRENT_TIME}"
echo "Logging to ${LOG_FILE}"

if [ "$MODEL_NAME" != "gpt" ] && [ "$MODEL_NAME" != "mistral" ]; then
    echo "Error: Model name must be 'gpt' or 'mistral'"
    exit 2
fi

./run_interaction_${MODEL_NAME}.sh > interaction_${LOG_FILE}.log
if [ $? -ne 0 ]; then
    echo "run_interaction_${MODEL_NAME}.sh failed"
    exit 3
fi

./run_interaction_${MODEL_NAME}_omni.sh > interaction_omni_${LOG_FILE}.log
if [ $? -ne 0 ]; then
    echo "run_interaction_${MODEL_NAME}_omni.sh failed"
    exit 4
fi

./run_script_full_${MODEL_NAME}.sh > script_full_${LOG_FILE}.log
if [ $? -ne 0 ]; then
    echo "run_script_full_${MODEL_NAME}.sh failed"
    exit 5
fi
