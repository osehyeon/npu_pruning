#!/bin/bash

MODEL_NAME=$1
INPUT_NAME=$2

python3 utils/onnxquant.py \
    --input-model model_float32/${MODEL_NAME}.onnx \
    --output-model model_uint8/${MODEL_NAME}.onnx \
    --calib-dataset utils/calib_dataset \
    --calib-count 20 \
    --input-name ${INPUT_NAME}

rm -rf augmented_model.onnx


