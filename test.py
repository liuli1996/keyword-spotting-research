import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
from helper.utils import create_logger
from pre_process import AudioPreprocessor, load_audio
import os
from model import Res15, LSTM, AMResNet, TradFPool3, DSCNN, TDNN
import librosa

logger = create_logger()

class ConfusionMatrix():
    def __init__(self, words_list):
        self.words_list = words_list
        self.word2label = {word: index for index, word in enumerate(self.words_list)}
        self.matrix = np.zeros(shape=(len(self.word2label), len(self.word2label)), dtype=np.int64)

    def update(self, prediction, label):
        self.matrix[self.word2label[label], self.word2label[prediction]] += 1
        return self.matrix

class TestProcessor():
    def __init__(self, model, model_type="cnn"):
        self.model = model
        self.model_type = model_type

    def compute_embedding(self, npy_file):
        data = np.load(npy_file)  # [1, 101, 40]
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "lstm":
                input_tensor = torch.from_numpy(data).unsqueeze(0)
                input_tensor = input_tensor.cuda()
                _, output = self.model(input_tensor)
                output = torch.mean(output, dim=1)
            elif self.model_type == "tdnn":
                if data.shape[0] >= 80:
                    data = data[:80, ]
                else:
                    data = np.concatenate((data, [data[-1, :] for i in range(80 - data.shape[0])]), axis=0)
                input_tensor = torch.from_numpy(data).unsqueeze(0)
                input_tensor = input_tensor.cuda()
                _, output = self.model(input_tensor)
            else:
                input_tensor = torch.from_numpy(data).unsqueeze(1)
                input_tensor = input_tensor.cuda()
                _, output = self.model(input_tensor)
        return output

    def compute_enroll_template(self, enroll_csv):
        # 创建关键词字典
        df = pd.read_csv(enroll_csv)
        keywords_dict = {}
        for index, (label, grouped) in enumerate(df.groupby(df['label'])):
            keywords_dict[label] = []
            for _, row in grouped.iterrows():
                embedding = self.compute_embedding(row['filename'])
                keywords_dict[label].append(embedding)
        logger.info('Model: {}, Num of words: {}, embedding shape: {}'.format(self.model_type, (index + 1), embedding.shape))

        # 计算每个关键词的均值向量
        for key, value in keywords_dict.items():
            embedding = torch.cat(value, dim=0)
            embedding = torch.mean(embedding, dim=0).reshape(1, -1)
            keywords_dict[key] = embedding
        return keywords_dict

    def compute_embedding_accuracy(self, enroll_csv, test_csv, threshold=0.7, plot_confusion_matrix=False):
        """计算模板匹配的准确率"""
        if plot_confusion_matrix:
            # if "customized" in enroll_csv:
            if True:
                words_list = ["zero","one","two","three","four","five","six","seven","eight","nine","silence","unknown"]
            # else:
            #     words_list = ["yes","no","up","down","left","right","on","off","stop","go","silence","unknown"]

            conf_matrix = ConfusionMatrix(words_list=words_list)
        else:
            conf_matrix = ConfusionMatrix(pd.read_csv(enroll_csv)['label'].unique())

        enroll_dict = self.compute_enroll_template(enroll_csv)

        # shuffle
        test_df = pd.read_csv(test_csv)
        test_df = test_df.sample(frac=1).reset_index(drop=True)

        similarity_scores = {}
        true_sample = 0

        for _, row in test_df.iterrows():
            test_tensor = self.compute_embedding(row["filename"])
            for key in enroll_dict.keys():
                reference_tensor = enroll_dict[key]
                cos_distance = F.cosine_similarity(reference_tensor, test_tensor, dim=1)
                similarity_scores.update({key: cos_distance})

            # 找相似度最大的key值
            pred_label = max(similarity_scores, key=similarity_scores.get)
            pred_score = similarity_scores[pred_label]
            if pred_score > threshold:
                conf_matrix.update(pred_label, row["label"])
            else:
                conf_matrix.update("unknown", row["label"])
                pred_label = "unknown"

            if pred_label == row["label"]:
                true_sample += 1

        acc = true_sample / len(test_df)
        logger.info('Accuracy: {}, Threshold: {}'.format(acc, threshold))
        logger.info('Words List: \n{}\nConfusion Matrix: \n{}'.format(conf_matrix.words_list, conf_matrix.matrix))
        if plot_confusion_matrix:
            return acc, conf_matrix.words_list, conf_matrix.matrix
        else:
            return acc

    def compute_roc_data(self, enroll_csv, test_csv):
        enroll_dict = self.compute_enroll_template(enroll_csv)
        test_df = pd.read_csv(test_csv)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        # logger.info("Removing \"unknown\" and \"silence\" from enroll_dict.")
        # enroll_dict.pop("unknown")
        # enroll_dict.pop("silence")

        scores = []
        labels = []

        for key in enroll_dict.keys():
            reference_tensor = enroll_dict[key]
            for _, row in test_df.iterrows():
                test_tensor = self.compute_embedding(row["filename"])
                cos_distance = F.cosine_similarity(reference_tensor, test_tensor, dim=1)
                scores.append(float(cos_distance))
                if row["label"] == key:
                    labels.append(1)
                else:
                    labels.append(0)

        scores = np.array(scores)
        labels = np.array(labels)
        return scores, labels

def revise_csv_file(csv_file, output_file):
    df = pd.read_csv(csv_file)
    df["filename"] = df["filename"].apply(lambda x: x.replace("/users/liuli/database/speech_commands/features/101_40_mfcc_clean",
                                                      "D:/dataset/features/speech_commands_v0.02/win30ms_hop10ms_mfcc_clean/"))
    # df["filename"] = df["filename"].apply(lambda x: x.replace("mfcc", "fbank"))
    df.to_csv(output_file, index=False, header=True)

if __name__ == '__main__':
    # revise_csv_file("./csv_linux/enroll.csv", "./csv_win/enroll_mfcc.csv")

    model_file = {
        # "res15": "./models_pt/res15/model_32000.pt",
        # "res15_fulllabel_finetune13layer_customized_only": "./models_pt/res15_fulllabel_finetune13layer_customized_only/model_10.pt",
        # "tradfpool3": "./models_pt/tradfpool3/model_32000.pt",
        # "tradfpool3_fulllabel_finetune4layer_customized_only": "./models_pt/tradfpool3_fulllabel_finetune4layer_customized_only/model_10.pt",
        # "lstm": "./models_pt/lstm/model_32000.pt",
        # "AmResNet": "./models_pt/AmResNet/model_24.pt",
        # "dscnn": "./models_pt/DSCNN/model_best.pt",
        "tdnn": "./models_pt/TDNN/model_best.pt",
    }

    enrollment_file = "./csv_win/user_defined_keywords_for_enrollment.csv"
    testing_file = "./csv_win/user_defined_keywords_for_testing.csv"
    # testing_file = "./csv_win/TIMIT_test.csv"

    # enrollment_file = "./csv_win/enroll_mfcc.csv"
    # testing_file = "./csv_win/test_mfcc.csv"

    for model_name, value in model_file.items():
        if model_name == "res15":
            model = Res15(n_labels=12).cuda()
            model.load(model_file[model_name])
        elif model_name == "res15_fulllabel_finetune13layer_customized_only":
            model = Res15(n_labels=26).cuda()
            model.load(model_file[model_name])
        elif model_name == "tradfpool3":
            model = TradFPool3(n_labels=12).cuda()
            model.load(model_file[model_name])
        elif model_name == "tradfpool3_fulllabel_finetune4layer_customized_only":
            model = TradFPool3(n_labels=26).cuda()
            model.load(model_file[model_name])
        elif model_name == "AmResNet":
            model = AMResNet(n_labels=26).cuda()
            model.load(model_file[model_name])
        elif model_name == "lstm":
            model = LSTM(n_labels=26).cuda()
            model.load(model_file[model_name])
            enrollment_file = "./csv_win/user_defined_keywords_for_enrollment_fbank.csv"
            # testing_file = "./csv_win/user_defined_keywords_for_testing_fbank.csv"
            testing_file = "./csv_win/TIMIT_test_fbank.csv"
        elif model_name == "dscnn":
            model = DSCNN(n_labels=12).cuda()
            model.load(model_file[model_name])
        elif model_name == "tdnn":
            model = TDNN(n_labels=12).cuda()
            model.load(model_file[model_name])
            # enrollment_file = "./csv_win/enroll_mfcc_tdnn.csv"
            # testing_file = "./csv_win/test_mfcc_tdnn.csv"
            enrollment_file = "./csv_win/user_defined_keywords_for_enrollment_tdnn.csv"
            testing_file = "./csv_win/user_defined_keywords_for_testing_tdnn.csv"
        else:
            raise ValueError("No suitable model file")

        test_processor = TestProcessor(model, model_type=model_name)
        test_processor.compute_embedding_accuracy(enrollment_file, testing_file)
        # test_processor.compute_embedding_accuracy(enrollment_file, testing_file, threshold=0.7)
        # scores, labels = test_processor.compute_roc_data(enrollment_file, testing_file)
        # # np.save("./roc_data/TIMIT_" + model_name + "_scores.npy", scores)
        # # np.save("./roc_data/TIMIT_" + model_name + "_labels.npy", labels)
        # np.save("./roc_data/" + model_name + "_scores.npy", scores)
        # np.save("./roc_data/" + model_name + "_labels.npy", labels)