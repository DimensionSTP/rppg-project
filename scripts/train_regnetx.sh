model_list="regnetx_002 regnetx_004 regnetx_006 regnetx_008 regnetx_016 regnetx_032 regnetx_040 regnetx_064 regnetx_080 regnetx_120 regnetx_160"

for model in $model_list
do
    HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python train.py backbone=$model batch_size=64
done
