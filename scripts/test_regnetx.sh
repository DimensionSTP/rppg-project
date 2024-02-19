model_list="regnetx_002 regnetx_004 regnetx_006 regnetx_008 regnetx_016 regnetx_032 regnetx_040 regnetx_064 regnetx_080 regnetx_120 regnetx_160"

for model in $model_list
do
    for ((epoch = 5; epoch <= 15; epoch++))
    do
        python main.py mode=test is_tuned=untuned backbone=$model batch_size=64 epoch=$epoch
    done
done

for epoch in "9 12 15"
do
    python main.py mode=test is_tuned=untuned backbone=regnetx_320 batch_size=32 epoch=$epoch
done
