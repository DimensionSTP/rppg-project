model_list="resnext50_32x4d resnext50d_32x4d resnext101_32x4d resnext101_32x8d resnext101_64x4d ecaresnext26t_32x4d ecaresnext50t_32x4d seresnext26d_32x4d seresnext26t_32x4d seresnext50_32x4d seresnext101_32x4d seresnext101_32x8d seresnext101d_32x8d"

for model in $model_list
do
    for ((epoch = 4; epoch <= 14; epoch++))
    do
        python main.py mode=test is_tuned=untuned backbone=$model epoch=$epoch
    done
done
