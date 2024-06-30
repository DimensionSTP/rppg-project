#!/bin/bash

is_tuned="untuned"
backbone="regnetx_032"
strategy="ddp"
precision=32
batch_size=64
epochs="9 10"

for epoch in $epochs
do
    python main.py mode=test \
        is_tuned=$is_tuned \
        backbone=$backbone \
        strategy=$strategy \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done
