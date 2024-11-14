#!/bin/bash

path="src/preprocessing/frame"
data_types="vipl ubfc pure mahnob"

for data_type in $data_types
do
    python $path/save_cropped_frame.py \
        data_type=$data_type
done
