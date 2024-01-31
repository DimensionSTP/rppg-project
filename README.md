# rPPG packaging project

## rPPG custom project

### Dataset
VIPL-HR-V1
Personal

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/rppg-project.git
cd rppg-project

# [OPTIONAL] create conda environment
conda create -n myenv python=3.7
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### Model Hyper-Parameters Tuning

* end-to-end
```shell
python main.py is_tuned=False
```

### Training

* end-to-end
```shell
python main.py mode=train is_tuned={bool}
```

### Testing

* end-to-end
```shell
python main.py mode=test is_tuned={bool} epoch={ckpt epoch}
```

### Prediction

* end-to-end
```shell
python main.py mode=predict is_tuned={bool} epoch={ckpt epoch}
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__