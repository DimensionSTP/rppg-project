#!/bin/bash

path="src/preprocessing/augmentation"
data_type="vipl"
aug_interval=30
split_ratio=1e-1

python $path/augment_vipl_metadata.py \
    data_type=$data_type \
    aug_interval=$aug_interval \
    split_ratio=$split_ratio
