#!/bin/bash

path="src/preprocessing/augmentation"
data_types="ubfc pure"
aug_intervals="30 10"
split_ratio=1e-1

for data_type in $data_types
do
    for aug_interval in $aug_intervals
    do
        python $path/augment_metadata.py \
            data_type=$data_type \
            aug_interval=$aug_interval \
            split_ratio=$split_ratio
    done
done
