import os
import random
import numpy as np
from manage_audio import AudioPreprocessor, load_audio

def prepare_audio(dataset_dir):
    f1 = open("fine_tune.txt", "w")
    f2 = open("dev.txt", "w")
    keyword_list = os.listdir(dataset_dir)
    keyword_list = list(filter(lambda x: os.path.isdir(os.path.join(dataset_dir, x)), keyword_list))
    keyword_list = list(filter(lambda x: x != "_background_noise_", keyword_list))
    for keyword in keyword_list:
        utterance_list = os.listdir(os.path.join(dataset_dir, keyword))
        utterance_selected_list = random.choices(utterance_list, k=30)
        for file in utterance_selected_list[0: 15]:
            f1.writelines([file.split(".")[0], ",", keyword, "\n"])
        for file in utterance_selected_list[15: 30]:
            f2.writelines([file.split(".")[0], ",", keyword, "\n"])
    f1.close()
    f2.close()

def extract_features_from_txt(txt_file, dataset_dir, output_dir):
    feature_extractor = AudioPreprocessor()
    f = open(txt_file, "r")
    utterance_list = f.readlines()
    utterance_list = list(map(lambda x: x.strip(), utterance_list))
    utterance_list = list(map(lambda x: tuple(x.split(",")), utterance_list))
    for file, keyword in utterance_list:
        wav_file = os.path.join(dataset_dir, keyword, file + ".wav")
        data = load_audio(wav_file)
        mfcc = feature_extractor.compute_mfccs(data).reshape(1, 101, 40)
        os.makedirs(os.path.join(output_dir, keyword), exist_ok=True)
        np.save(os.path.join(output_dir, keyword, file + ".npy"), mfcc)
    f.close()

def append_personalized_keyword_into_txt(txt_file, output_txt):
    with open(txt_file, "r") as f:
        contents = f.readlines()
    with open(output_txt, "w") as f:
        for content in contents:
            f.writelines(content)
        for file in os.listdir("features/train/personalized_keyword"):
            f.writelines([file.split(".")[0], ",", "personalized_keyword\n"])


if __name__ == '__main__':
    # prepare_audio(r"D:\dataset\speech_commands_v0.02")
    # extract_features_from_txt("fine_tune.txt", r"D:\dataset\speech_commands_v0.02", "features/train")
    # extract_features_from_txt("dev.txt", r"D:\dataset\speech_commands_v0.02", "features/dev")
    append_personalized_keyword_into_txt("fine_tune.txt", "fine_tune_append.txt")
