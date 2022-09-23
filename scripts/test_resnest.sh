model_list0="resnest14d resnest26d resnest50d resnest50d_1s4x24d"

for model in $model_list0
do
    for ((epoch = 9; epoch <= 16; epoch++))
    do
        python test.py backbone=$model epoch=$epoch
    done
done

model_list1="resnest50d_4s2x40d resnest101e"

for model in $model_list1
do
    for ((epoch = 5; epoch <= 11; epoch++))
    do
        python test.py backbone=$model batch_size=64 epoch=$epoch
    done
done

model_list2="resnest200e resnest269e"

for model in $model_list2
do
    for ((epoch = 3; epoch <= 9; epoch++))
    do
        python test.py backbone=$model batch_size=32 epoch=$epoch
    done
done
