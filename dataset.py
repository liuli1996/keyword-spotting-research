import torch.utils.data as data
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from glob import glob
import os
import torch
import pickle

class SpeechDataset(data.Dataset):
    def __init__(self, csv_file, wanted_words):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.labels = {word: i for i, word in enumerate(wanted_words)}

    def __getitem__(self, index):
        fname = self.df.loc[index]['filename']
        audio_data = np.load(fname)
        label = self.df.loc[index]['label']
        label = self.labels[label]
        return audio_data, label

    def __len__(self):
        return len(self.df)

class TDNN_SpeechDataset(data.Dataset):
    def __init__(self, csv_file, wanted_words):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        if "train" in csv_file:
            self.mode = "train"
        else:
            self.mode = "test"
        self.df["filename"] = self.df["filename"].apply(lambda x: x.replace("D:/dataset/features/speech_commands_v0.02/win30ms_hop10ms_mfcc_clean",
                                                                            "D:/dataset/features/speech_commands_v0.02/win25ms_hop10ms_41fbank_cmvn/" + self.mode))
        self.labels = {word: i for i, word in enumerate(wanted_words)}

    def __getitem__(self, index):
        fname = self.df.loc[index]['filename']
        audio_data = np.load(fname)  # only fetch 80 frames
        if audio_data.shape[0] >= 80:
            audio_data = audio_data[:80, ]
        else:
            audio_data = np.concatenate((audio_data, [audio_data[-1, :] for i in range(80 - audio_data.shape[0])]), axis=0)
        label = self.df.loc[index]['label']
        label = self.labels[label]
        return audio_data, label

    def __len__(self):
        return len(self.df)

class E2E_SpeechDataset(data.Dataset):
    def __init__(self, csv_file, cmvn_file=None):
        super().__init__()
        self.df = pd.read_csv(csv_file)

        self.cmvn_file = cmvn_file
        if self.cmvn_file:
            f = open(cmvn_file, "rb")
            data_dict = pickle.load(f)
            self.mean = data_dict["mean"]
            self.var = data_dict["var"]

        # language model
        objetcs = torch.load("./models_pt/charLM/prep.pt")
        self.char_dict = objetcs["char_dict"]
        self.reverse_word_dict = objetcs["reverse_word_dict"]
        self.max_word_len = objetcs["max_word_len"]

    def __getitem__(self, index):
        file_name = self.df.loc[index]['filename']
        label_text = self.df.loc[index]['label']
        is_same = self.df.loc[index]["is_same"]
        audio_data = np.load(file_name)
        if self.cmvn_file:
            audio_data = (audio_data - self.mean) / np.sqrt(self.var)
        label = np.array(text2vec([label_text], self.char_dict, max_word_len=self.max_word_len))
        return audio_data, label, is_same

    def __len__(self):
        return len(self.df)

def text2vec(words, char_dict, max_word_len):
    """ Return list of list of int """
    word_vec = []
    for word in words:
        vec = [char_dict[ch] for ch in word]
        if len(vec) < max_word_len:
            vec += [char_dict["PAD"] for _ in range(max_word_len - len(vec))]
        vec = [char_dict["BOW"]] + vec + [char_dict["EOW"]]
        word_vec.append(vec)
    return word_vec

def _collate_fn(batch):
    # print(batch)  # batch is a tuple, the first is data, the second is label
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)  # descending order
    longest_sample = batch[0][0]
    freq_size = longest_sample.shape[1]
    batch_size = len(batch)
    max_seqlength = longest_sample.shape[0]
    inputs = torch.zeros(batch_size, max_seqlength, freq_size)
    input_lens = []
    labels = []
    is_same = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        label = sample[1]
        is_same_label = sample[2]
        seq_length = tensor.shape[0]
        inputs[x].narrow(dim=0, start=0, length=seq_length).copy_(torch.Tensor(tensor))
        input_lens.append(seq_length)
        labels.append(label)
        is_same.append(is_same_label)
    labels = torch.LongTensor(labels)
    is_same = torch.LongTensor(is_same)
    return inputs, labels, is_same, input_lens

class E2E_DataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

if __name__ == '__main__':
    train_dataset = E2E_SpeechDataset(csv_file="./csv_win/train_mfcc_e2e.csv")
    train_loader = E2E_DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=1)

    for data, labels, is_same, input_lens in train_loader:
        pass
