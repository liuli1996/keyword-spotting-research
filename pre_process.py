import numpy as np
import time
import pandas as pd
import os
import librosa
import random
import glob
import multiprocessing as mp
from tqdm import tqdm
from helper.utils import check_dir
import shutil
import audioread

def dct(n_filters, n_input):
    """Discrete cosine transform (DCT type-III) basis.

    .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    n_filters : int > 0 [scalar]
        number of output components (DCT filters)

    n_input : int > 0 [scalar]
        number of input components (frequency bins)

    Returns
    -------
    dct_basis: np.ndarray [shape=(n_filters, n_input)]
        DCT (type-III) basis vectors [1]_

    Notes
    -----
    This function caches at level 10.
    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis

class AudioPreprocessor():
    def __init__(self, sr=16000, n_dct_filters=40, n_mel_filters=40, f_max=4000, f_min=20, n_fft_ms=30, hop_ms=10):
        super().__init__()
        self.n_mel_filters = n_mel_filters
        self.dct_filters = dct(n_dct_filters, n_mel_filters)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = sr // 1000 * n_fft_ms
        self.hop_length = sr // 1000 * hop_ms

    def compute_mfcc(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mel_filters,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

    def compute_fbank(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mel_filters,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = np.array(data, order="F").astype(np.float32)
        return data

def load_audio(file_name, in_len=16000):
    try:
        data = librosa.load(file_name, sr=16000)[0]
    except audioread.exceptions.NoBackendError:
        print("There exists problem of the file {}". format(file_name), end="")
        raise

    if len(data) <= in_len:
        data = np.pad(data, (0, in_len - len(data)), "constant")  # data padding
    else:
        data = data[:in_len]  # data truncating
    return data

def compute_GSC_mfcc(wav_file, output_file):
    processor = AudioPreprocessor(n_dct_filters=13)
    # data = load_audio(wav_file)
    data, sr = librosa.load(wav_file, sr=None)
    # feat_spec = processor.compute_mfcc(data).reshape((1, 101, 40))
    feat_spec = processor.compute_mfcc(data).squeeze(2)  # for TIMIT dataset
    # feat_spec = processor.compute_fbank(data).T
    np.save(output_file, feat_spec)

def run_helper(args):
    compute_GSC_mfcc(*args)

if __name__ == '__main__':
    data_dir = r"D:\dataset\speech_commands_v0.02_augmented\silence"
    output_dir = r"D:\dataset\features\speech_commands_v0.02\win30ms_hop10ms_13mfcc_clean"
    wav_list = glob.glob(os.path.join(data_dir, "*.wav"), recursive=True)
    wav_list = list(map(lambda x: x.replace("\\", "/"), wav_list))
    check_dir(output_dir)

    # # copy the directory structure
    # files = os.listdir(data_dir)
    # for file in files:
    #     if os.path.isdir(os.path.join(data_dir, file)):
    #         check_dir(os.path.join(output_dir, file))
    #
    # # filter some audio
    # wav_list = list(filter(lambda x: x.split("/")[-2] != "_background_noise_", wav_list))

    # parallel computing
    items = [(file, os.path.join(output_dir, file.split("/")[-2], os.path.basename(file).replace(".wav", ".npy"))) for file in wav_list]
    pool = mp.Pool(mp.cpu_count())
    res = list(tqdm(pool.imap(run_helper, items), total=len(items), desc="Computing MFCC:"))  # to display the procedure
    pool.close()
    pool.join()

    # single test
    # run_helper(items[0])
