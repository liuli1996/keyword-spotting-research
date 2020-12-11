import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
import math
import pickle
from helper.utils import check_dir

def compute_stats(feature_dir, npy_list, start, end, output_file):
    start = max(0, start)
    end = min(len(npy_list), end)
    if start > end:
        return 0

    n = 0
    sum_x = None
    sum_x_power = None

    for i in range(start, end):
        feats = np.load(os.path.join(feature_dir, npy_list[i]))

        if sum_x is not None:
            sum_x += np.sum(feats, axis=0)
        else:
            sum_x = np.sum(feats, axis=0)

        if sum_x_power is not None:
            sum_x_power += np.sum(feats ** 2, axis=0)
        else:
            sum_x_power = np.sum(feats ** 2, axis=0)

        n += feats.shape[0]

    with open(output_file, "wb") as f:
        pickle.dump([sum_x, sum_x_power, n], f)

    return sum_x, sum_x_power, n

def compute_cmvn(stats_dir):
    sum_x = 0
    sum_x_power = 0
    nums = 0

    for file in os.listdir(stats_dir):
        if not file.startswith("cmvn"):
            continue
        f = open(os.path.join(stats_dir, file), "rb")
        x, x_power, n = pickle.load(f)
        sum_x += x
        sum_x_power += x_power
        nums += n

    mean = sum_x / nums
    var = sum_x_power / nums - mean ** 2
    return mean, var

def run_helper(args):
    return compute_stats(*args)

if __name__ == '__main__':
    feature_dir = r"D:\dataset\features\TIMIT\win30ms_hop10ms_mfcc_clean\train"
    output_dir = "./cmvn_stats"
    n_pieces = 8

    check_dir(output_dir)

    # parallel computing
    npy_list = os.listdir(feature_dir)
    interval = math.ceil(len(npy_list) / n_pieces)
    items = []
    for i in range(n_pieces):
        output_file = os.path.join(output_dir, "cmvn_stats_#{}.pkl".format(i + 1))
        items.append((feature_dir, npy_list, i * interval, (i + 1) * interval, output_file))
    p = mp.Pool(mp.cpu_count())
    res = list(tqdm(p.imap(run_helper, items), total=len(items), desc='Computing statistics: '))
    p.close()
    p.join()

    # compute CMVN
    mean, var = compute_cmvn(output_dir)
    print(mean, var)
    with open(os.path.join(output_dir, "TIMIT_train_global_stats.pkl"), "wb") as f:
        pickle.dump({"mean": mean, "var": var}, f)



