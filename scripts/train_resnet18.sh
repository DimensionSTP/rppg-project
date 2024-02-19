for rnn_num_layers in {1..5}
do
    HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python main.py mode=train is_tuned=untuned backbone="resnet18" batch_size=64  rnn_num_layers=$rnn_num_layers
done

for rnn_num_layers in {1..5}
do
    HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python main.py mode=train is_tuned=untuned backbone="resnet18" batch_size=64  rnn_num_layers=$rnn_num_layers direction="uni"
done
