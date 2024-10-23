#!/bin/bash

is_tuned="untuned"
is_pretrained=False
strategy="ddp"
precision=32
batch_size=128
epochs="9 10"

for epoch in $epochs
do
    python main.py mode=test \
        is_tuned=$is_tuned \
        is_pretrained=$is_pretrained \
        strategy=$strategy \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done
