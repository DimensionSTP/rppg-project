#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
is_pretrained=False
strategy="ddp"
precision=32
batch_size=128
epoch=10
model_detail="PhysFormer-customized"

python $path/upload_to_hf_hub.py \
    is_tuned=$is_tuned \
    is_pretrained=$is_pretrained \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size \
    epoch=$epoch \
    model_detail=$model_detail
