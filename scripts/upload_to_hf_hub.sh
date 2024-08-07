#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
backbone="regnetx_032"
strategy="ddp"
precision=32
batch_size=64
epoch=10
model_detail="RhythmNet-customized"

python $path/upload_to_hf_hub.py \
    is_tuned=$is_tuned \
    backbone=$backbone \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size \
    epoch=$epoch \
    model_detail=$model_detail
