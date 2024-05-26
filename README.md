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
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting
```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}
HF_HOME={HF_HOME}
```

### Model Hyper-Parameters Tuning

* end-to-end
```shell
python main.py mode=tune is_tuned=untuned num_trials={num_trials}
```

### Training

* end-to-end
```shell
python main.py mode=train is_tuned={tuned or untuned} num_trials={num_trials}
```

### Testing

* end-to-end
```shell
python main.py mode=test is_tuned={tuned or untuned} num_trials={num_trials} epoch={ckpt epoch}
```

### Prediction

* end-to-end
```shell
python main.py mode=predict is_tuned={tuned or untuned} num_trials={num_trials} epoch={ckpt epoch}
```

### Examples of shell scipts

* train
```shell
bash scripts/train.sh
```

* test
```shell
bash scripts/test.sh
```


__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__