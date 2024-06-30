#!/bin/bash

is_tuned="untuned"
backbone="regnetx_032"
strategy="ddp"
precision=32
batch_size=64

python main.py mode=train \
    is_tuned=$is_tuned \
    backbone=$backbone \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size
