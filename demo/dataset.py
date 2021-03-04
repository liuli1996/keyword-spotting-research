import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import random

class TrainDataset(data.Dataset):
    def __init__(self, txt_file, feature_dir, num_utter, num_word, n_batches):
        super().__init__()
        self.feature_dir = feature_dir
        self.num_utter = num_utter
        self.num_word = num_word
        self.n_batches = n_batches
        self.words_dict = {}
        self.batch = []

        # structure: self.words_dict = {'word1': [file1, file2, ...], 'word2': [file1, file2, ...], ...}
        with open(txt_file, "r") as f:
            self.utterance_list = f.readlines()
            self.utterance_list = list(map(lambda x: x.strip(), self.utterance_list))
            self.utterance_list = list(map(lambda x: tuple(x.split(",")), self.utterance_list))

        for filename, keyword in self.utterance_list:
            if keyword not in self.words_dict:
                self.words_dict[keyword] = [os.path.join(self.feature_dir, keyword, filename + ".npy")]
            else:
                self.words_dict[keyword].append(os.path.join(self.feature_dir, keyword, filename + ".npy"))

        # structure: self.batch = [[file1, file2, ...], [file1, file2, ...], ...]
        self.create_batch(n_batches)

    def create_batch(self, batch_size):
        words_list = list(self.words_dict.keys())
        for i in range(batch_size):
            mini_batch = []
            selected_words = random.sample(words_list, self.num_word)  # 随机采样关键词
            for word in selected_words:
                audio_list = self.words_dict[word]
                selected_audio = random.sample(audio_list, self.num_utter)  # 随机采样语音
                for audio in selected_audio:
                    mini_batch.append(audio)
            self.batch.append(mini_batch)

    def __getitem__(self, index):
        mini_batch = self.batch[index]  # [file1, file2, ...]
        data_list = []
        for file in mini_batch:
            data = np.load(file)
            data_list.append(data)
        data = np.concatenate(data_list, axis=0)  # [-1, 101, 40]
        return data

    def __len__(self):
        return self.n_batches

if __name__ == "__main__":
    train_set = TrainDataset("fine_tune_append.txt", "features/train", 5, 5, 200)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
        num_workers=1)

    for data in train_loader:
        print(data.shape)
