# rPPG packaging project

## rPPG custom project

### Dataset
VIPL-HR-V1
Pesonal

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

### Training

* end-to-end
```shell
python train.py
```

### Testing

* end-to-end
```shell
python test.py
```
