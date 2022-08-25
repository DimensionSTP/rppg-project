import os
import glob

path_list = ["./stmaps", "./hrv/result", "./preprocessed_dataset/train"]

for path in path_list:
    files = glob.glob(path + "/jj*")
    for f in files:
        new_f = os.path.join(path, 'j' + os.path.basename(f))  # 문자 추가
        os.rename(f, new_f)
        print('{} ==> {}'.format(f, new_f))