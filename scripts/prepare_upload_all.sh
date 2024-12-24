#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
is_pretrained=False
strategy="ddp"
precision=32
batch_size=128
model_detail="PhysFormer-customized"

python $path/prepare_upload_all.py \
    is_tuned=$is_tuned \
    is_pretrained=$is_pretrained \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size \
    model_detail=$model_detail
