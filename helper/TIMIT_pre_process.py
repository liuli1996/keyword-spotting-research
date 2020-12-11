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

def extract_negative_utterance(data_dir, output_file):
    # data_dir = "/users/liuli/database/TIMIT/TEST/"
    # output_file = "./csv/TIMIT.csv"
    wav_list = glob(os.path.join(data_dir, "*/*/*.WAV"), recursive=True)
    positive_keywords = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    positive_utt = []
    for file in wav_list:
        wrd_file = file.replace("_decoded.wav", ".WRD")
        with open(wrd_file, "r") as f:
            label = f.readlines()
            label_map = list(map(lambda x: x.strip().split(" ")[-1], label))
            if set(label_map) & set(positive_keywords):  # detecting the overlapped keyword
                positive_utt.append(file)
        f.close()
    # print(positive_utt)
    temp = set(wav_list) - set(positive_utt)  # remove utterances containing keywords

    with open(output_file, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["filename", "label"])
        for file in temp:
            csv_writer.writerow([file, "unknown"])
    f.close()

def sphere2wav(file):
    """
    convert .sphere format to .wav format
    """
    sph = SPHFile(file)
    sph.write_wav(filename=file.replace(".WAV", "_decoded.wav"))

def write_TIMIT_test_csv(feature_dir, output_file):
    npy_list = glob(os.path.join(feature_dir, "*.npy"))
    with open(output_file, "w", newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["filename", "label"])
        for file in npy_list:
            csv_writer.writerow([file, "unknown"])

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
        # feat_spec = processor.compute_mfcc(data).reshape((1, 101, 40))
        feat_spec = processor.compute_fbank(data).T
        np.save(output_file.replace(".npy", "_" + str(i) + ".npy"), feat_spec)

    # processor = AudioPreprocessor(n_dct_filters=13)
    # # data = load_audio(wav_file)
    # data, sr = librosa.load(wav_file, sr=None)
    # # feat_spec = processor.compute_mfcc(data).reshape((1, 101, 40))
    # feat_spec = processor.compute_mfcc(data).squeeze(2)  # for TIMIT dataset
    # # feat_spec = processor.compute_fbank(data).T
    # np.save(output_file, feat_spec)

def run_helper(args):
    compute_TIMIT_mfcc(*args)

if __name__ == '__main__':
    # extract_negative_utterance(r"D:\dataset\TIMIT\TEST", "../csv_win/TIMIT.csv")

    # csv_file = "../csv_win/TIMIT.csv"
    # output_dir = "D:/dataset/features/TIMIT/win30ms_hop10ms_40fbank_clean/n_test"
    # check_dir(output_dir)
    # # open csv file
    # with open(csv_file, "r") as f:
    #     wav_list = f.readlines()
    #     wav_list = list(map(lambda x: x.strip().split(",")[0], wav_list))
    #     wav_list.pop(0)  # remove the header
    #     wav_list = list(map(lambda x: x.replace("\\", "/"), wav_list))  # replace "\" in file path
    #
    # # parallel computing
    # items = [(file, os.path.join(output_dir, file.split("/")[-2] + "-" + os.path.basename(file).replace("_decoded.wav", ".npy"))) for file in wav_list]
    # pool = mp.Pool(mp.cpu_count())
    # res = list(tqdm(pool.imap(run_helper, items), total=len(items), desc="Processing:"))  # to display the procedure
    # pool.close()
    # pool.join()

    write_TIMIT_test_csv("D:/dataset/features/TIMIT/win30ms_hop10ms_13mfcc_clean/n_test", "../csv_win/TIMIT_test_e2e.csv")
