from glob import glob
import os
import csv
from pre_process import AudioPreprocessor
import librosa
import numpy as np
from helper.utils import check_dir
import multiprocessing as mp
from tqdm import tqdm
from sphfile import SPHFile
import pandas as pd

def extract_negative_utterance():
    # data_dir = "/users/liuli/database/TIMIT/TEST/"
    # output_file = "./csv/TIMIT.csv"
    txt_list = glob(os.path.join(r"D:\dataset\LibriSpeech\test-clean", "*\*\*.txt"), recursive=True)
    positive_keywords = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    negative_utt = []
    for file in txt_list:
        df = pd.read_csv(file, names=["contents"])
        df["id"] = df["contents"].apply(lambda x: x.split(" ", 1)[0])
        df["text"] = df["contents"].apply(lambda x: x.split(" ", 1)[-1].lower())

        for idx, row in df.iterrows():
            text = row["text"].split(" ")
            if not set(text) & set(positive_keywords):  # detecting the overlapped keyword
                negative_utt.append(row["id"])

    with open("./csv_win/LibriSpeech.csv", "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["filename", "label"])
        for id in negative_utt:
            csv_writer.writerow([os.path.join(r"D:\dataset\LibriSpeech\test-clean",
                                              id.split("-")[0],
                                              id.split("-")[1],
                                              id + ".flac"), "unknown"])
    f.close()

def write_LibriSpeech_test_csv(feature_dir, output_file):
    npy_list = glob(os.path.join(feature_dir, "*.npy"))
    with open(output_file, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["filename", "label"])
        for file in npy_list:
            csv_writer.writerow([file, "unknown"])
    f.close()

    df = pd.read_csv("../csv_win/user_defined_keywords_for_testing.csv")
    df = df[(True ^ df['label'].isin(['unknown']))]
    df_TIMIT = pd.read_csv(output_file)
    df_total = pd.concat([df, df_TIMIT], axis=0).reset_index(drop=True)
    df_total.to_csv(output_file, index=False)

def compute_TIMIT_mfcc(wav_file, output_file, piece_len=16000):
    processor = AudioPreprocessor()
    sig, sr = librosa.load(wav_file, sr=16000)
    n_pieces = len(sig) // piece_len  # lower bound
    # data padding
    if len(sig) <= n_pieces * 16000:
        sig = np.pad(sig, (0, n_pieces * 16000 - len(sig)), "constant")
    # slice into pieces
    for i in range(n_pieces):
        data = sig[i * piece_len: (i + 1) * piece_len]
        feat_spec = processor.compute_mfcc(data).reshape((1, 101, 40))
        # feat_spec = processor.compute_fbank(data).T
        np.save(output_file.replace(".npy", "_" + str(i) + ".npy"), feat_spec)

def run_helper(args):
    compute_TIMIT_mfcc(*args)

if __name__ == '__main__':
    # extract_negative_utterance(r"D:\dataset\TIMIT\TEST", "../csv_win/TIMIT.csv")
    csv_file = "../csv_win/LibriSpeech.csv"
    output_dir = "D:/dataset/features/LibriSpeech/win30ms_hop10ms_mfcc_clean/n_test"
    check_dir(output_dir)
    # open csv file
    with open(csv_file, "r") as f:
        wav_list = f.readlines()
        wav_list = list(map(lambda x: x.strip().split(",")[0], wav_list))
        wav_list.pop(0)  # remove the header
    f.close()
    # parallel computing
    items = [(file, os.path.join(output_dir, os.path.basename(file).replace(".flac", ".npy"))) for file in wav_list]
    pool = mp.Pool(mp.cpu_count())
    res = list(tqdm(pool.imap(run_helper, items), total=len(items), desc="Processing:"))  # to display the procedure
    pool.close()
    pool.join()

    write_LibriSpeech_test_csv("D:/dataset/features/LibriSpeech/win30ms_hop10ms_mfcc_clean/n_test",
                               "../csv_win/LibriSpeech_test.csv")
