#!/bin/bash

path="src/preprocessing"
aug_interval=30
split_ratio=1e-1

python $path/preprocess_vipl_metadata.py aug_interval=$aug_interval split_ratio=$split_ratio
