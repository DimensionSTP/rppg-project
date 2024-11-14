#!/bin/bash

path="src/preprocessing/metadata"
data_type="pure"

python $path/get_pure_metadata.py \
    data_type=$data_type 