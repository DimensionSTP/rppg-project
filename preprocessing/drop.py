import os
import glob

from tqdm import tqdm

PATH = "preprocessed_dataset"
SPLIT = ["train", "val", "test"]

for split in tqdm(SPLIT):
    stmap_gather = glob.glob(f"{PATH}/{split}/*.npy")
    stmaps = [i[:-4] for i in stmap_gather]
    label_gather = glob.glob(f"{PATH}/{split}/*.csv")
    labels = [i[:-4] for i in label_gather]
    compare = list(set(stmaps) & set(labels))
    stmap_compare = [i + ".npy" for i in compare]
    label_compare = [i + ".csv" for i in compare]
    total = stmap_compare + label_compare
    file_list =glob.glob(f"{PATH}/{split}/*")
    remove_list = list(set(file_list) - set(total))
    [os.remove(f) for f in remove_list]