import glob
import pandas as pd
from tqdm import tqdm

STMAPS = glob.glob("stmaps/*300.npy")
SAVE_PATH = "preprocessed_dataset"
HRV_PATH = "./hrv/result"

for map in tqdm(STMAPS): 
    name = map[7:-8]
    df = pd.read_csv(f"{HRV_PATH}/{name}_10.csv")
    chunk_num = len(df) // 10
    for i in range(chunk_num):
        start = i * 10
        end = start + 10
        slice = df[start:end]
        if name[:3] == "hjh":
            slice.to_csv(f"../{SAVE_PATH}/val/{name}_300_{i}.csv", index=False)
        elif name[:3] == "yjh":
            slice.to_csv(f"../{SAVE_PATH}/test/{name}_300_{i}.csv", index=False)
        else:
            slice.to_csv(f"../{SAVE_PATH}/train/{name}_300_{i}.csv", index=False)