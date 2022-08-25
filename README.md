# rPPG packaging project

## Package structure

project tree

├─checkpoints  
├─configs  
│  ├─architecture_module  
│  ├─callbacks  
│  ├─dataset_module  
│  ├─hydra  
│  ├─logger  
│  └─trainer  
├─logs  
│  ├─runs  
│  └─wandb  
├─preprocessed_dataset  
│  ├─test  
│  ├─train  
│  └─val  
├─preprocessing  
│  ├─hrv  
│  │  ├─data  
│  │  ├─raw  
│  │  └─result  
│  └─raw  
│      ├─cam  
│      └─ecg  
└─src  
    ├─architecture_modules  
    │  └─models  
    ├─dataset_modules  
    │  └─datasets  
    ├─preprocess  
    │  ├─hrv  
    │  └─stmap  
    └─utils  

for train

    python train.py
