model_list="regnety_002 regnety_004 regnety_006 regnety_008 regnety_016 regnety_032 regnety_040 regnety_064 regnety_080 regnety_120 regnety_160"

for model in $model_list
do
    for ((epoch = 5; epoch <= 16; epoch++))
    do
        python test.py backbone=$model batch_size=64 epoch=$epoch
    done
done

for epoch in "2 4 5"
do
    python test.py backbone=regnety_320 batch_size=32 epoch=$epoch
done
