import glob
import numpy as np
from tqdm import tqdm

STMAPS = glob.glob("./preprocessing/mp_stmaps/*300.npy")
SAVE_PATH = "preprocessed_dataset"

for map in tqdm(STMAPS):
    map_name = map[-14:-4]
    stmap = np.load(map)
    chunk_num = stmap.shape[0] // 10
    for i in range(chunk_num):
        start = i * 10
        end = start + 10
        slice = stmap[start:end, :, :, :]
        if map_name[:3] == "hjh":
            np.save(f"./{SAVE_PATH}/val/{map_name}_{i}.npy", slice)
        elif map_name[:3] == "yjh":
            np.save(f"./{SAVE_PATH}/test/{map_name}_{i}.npy", slice)
        else:
            np.save(f"./{SAVE_PATH}/train/{map_name}_{i}.npy", slice)
