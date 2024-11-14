#!/bin/bash

is_tuned="untuned"
is_pretrained=False
strategy="ddp"
precision=32
batch_size=128
workers_ratio=8
use_all_workers=False

python main.py mode=train \
    is_tuned=$is_tuned \
    is_pretrained=$is_pretrained \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers