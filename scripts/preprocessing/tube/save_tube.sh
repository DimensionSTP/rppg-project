#!/bin/bash

path="src/preprocessing/tube"
data_type="vipl"
clip_frame_size=128
aug_interval=30

python $path/save_tube.py \
    data_type=$data_type \
    clip_frame_size=$clip_frame_size \
    aug_interval=$aug_interval
