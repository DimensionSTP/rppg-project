model_list0="resnest14d resnest26d resnest50d resnest50d_1s4x24d"

for model in $model_list0
do
    python main.py mode=train is_tuned=untuned backbone=$model
done

model_list1="resnest101e resnest50d_4s2x40d"

for model in $model_list1
do
    python main.py mode=train is_tuned=untuned backbone=$model batch_size=64
done

model_list2="resnest200e resnest269e"

for model in $model_list2
do
    python main.py mode=train is_tuned=untuned backbone=$model  batch_size=32
done
