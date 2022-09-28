model_list0="resnext50_32x4d resnext50d_32x4d resnext101_32x4d resnext101_32x8d resnext101_64x4d"

for model in $model_list0
do
    python train.py backbone=$model batch_size=64
done

model_list1="ecaresnext26t_32x4d ecaresnext50t_32x4d seresnext26d_32x4d seresnext26t_32x4d seresnext50_32x4d seresnext101_32x4d seresnext101_32x8d seresnext101d_32x8d"

for model in $model_list1
do
    python train.py backbone=$model batch_size=64
done
