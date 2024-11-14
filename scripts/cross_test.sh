#!/bin/bash

is_tuned="untuned"
is_pretrained=False
strategy="ddp"
precision=32
batch_size=128
workers_ratio=8
use_all_workers=False
model_trained_dataset_name="vipl"
data_types="vipl ubfc pure mahnob"
epochs="9 10"

for data_type in $data_types
do
    for epoch in $epochs
    do
        python main.py mode=test \
            is_tuned=$is_tuned \
            is_pretrained=$is_pretrained \
            strategy=$strategy \
            precision=$precision \
            batch_size=$batch_size \
            workers_ratio=$workers_ratio \
            use_all_workers=$use_all_workers \
            dataset_name=$model_trained_dataset_name \
            data_type=$data_type \
            epoch=$epoch
    done
done
