#!/bin/bash

path="src/preprocessing/metadata"
data_type="mahnob"

python $path/get_mahnob_metadata.py \
    data_type=$data_type 
