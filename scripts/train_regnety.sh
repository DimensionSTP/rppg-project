model_list="regnety_002 regnety_004 regnety_006 regnety_008 regnety_016 regnety_032 regnety_040 regnety_064 regnety_080 regnety_120 regnety_160"

for model in $model_list
do
    HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python train.py backbone=$model batch_size=64
done
