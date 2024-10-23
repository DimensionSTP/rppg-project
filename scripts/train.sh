#!/bin/bash

is_tuned="untuned"
is_pretrained=False
strategy="ddp"
precision=32
batch_size=128

python main.py mode=train \
    is_tuned=$is_tuned \
    is_pretrained=$is_pretrained \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size
