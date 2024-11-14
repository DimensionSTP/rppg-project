#!/bin/bash

path="src/preprocessing/metadata"
data_type="ubfc"

python $path/get_ubfc_metadata.py \
    data_type=$data_type 