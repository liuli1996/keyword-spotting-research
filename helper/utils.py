import torch
import logging
import os
import pandas as pd
import random

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_logger(log_path='./develop.log'):
    # 创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # 定义handler的输出格式
    fh_formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fh.setFormatter(fh_formatter)
    sh_formatter = logging.Formatter('%(message)s')
    sh.setFormatter(sh_formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)  # 没有偏置
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_eval(name, scores, labels, loss, end="\n"):
    """
    print训练/测试阶段的loss和准确率
    """
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, dim=1)[1].view(batch_size) == labels).float().sum() / batch_size
    loss = loss.item()  # gets the a scalar value held in the loss
    print("{} accuracy: {:>5}, loss: {}".format(name, accuracy, loss), end=end)
    return accuracy.item()

def revising_csv():
    origin_csv = "../csv_win/TIMIT_test_e2e.csv"
    target_csv = "../csv_win/TIMIT_test_e2e.csv"
    df = pd.read_csv(origin_csv)
    df["filename"] = df["filename"].apply(lambda x: x.replace("\\", "/"))
    df["filename"] = df["filename"].apply(lambda x: x.replace("//", "/"))
    df["filename"] = df["filename"].apply(lambda x: x.replace("1313mfcc", "13mfcc"))
    # df["label"] = df["filename"].apply(lambda x: x.split("/")[-2])
    df["is_same"] = 1

    keywords_list = ["zero","one","two","three","four","five","six","seven","eight","nine"]

    for idx, row in df.iterrows():
        if row["label"] == "unknown":
            df.loc[idx, "label"] = random.choice(keywords_list)
            df.loc[idx, "is_same"] = 0
        elif row["label"] == "silence" or random.random() > 0.5:
            df.loc[idx, "label"] = random.choice(list(set(keywords_list) - set(row["label"])))
            df.loc[idx, "is_same"] = 0

    # df_copy = df.copy()
    # df_copy["label"] = df_copy["label"].apply(lambda x: random.choice(list(set(keywords_list) - set(x))))
    # df_copy["is_same"] = 0
    # df = df[True ^ df["label"].isin(["silence"])]
    # df_total = pd.concat([df, df_copy])
    df.to_csv(target_csv, index=False)

def write_utt2spk():
    input_file = "../csv_linux/enroll_customized.csv"
    output_file = "/users/liuli/project/kaldi/egs/timit/s5/data/GSC_enroll_customized/utt2spk"
    check_dir("/users/liuli/project/kaldi/egs/timit/s5/data/GSC_enroll_customized")
    df = pd.read_csv(input_file)
    wf = open(output_file, "w")
    for index, row in df.iterrows():
        spk_id = row["filename"].split("/")[-2]
        utt_id = spk_id + "-" + row["filename"].split("/")[-1][:-4]
        wf.writelines([utt_id + " " + spk_id + "\n"])
    wf.close()
    os.system(r"sed -i 's/r$//' " + output_file)
    os.system("sort " + output_file + " -o " + output_file)
    os.system("/users/liuli/project/kaldi/egs/timit/s5/utils/utt2spk_to_spk2utt.pl " + output_file + " > " + output_file.replace("utt2spk", "spk2utt"))

def write_wavscp():
    input_file = "../csv_linux/enroll_customized.csv"
    output_file = "/users/liuli/project/kaldi/egs/timit/s5/data/GSC_enroll_customized/wav.scp"
    dataset_dir = "/users/liuli/database/speech_commands/audio"
    df = pd.read_csv(input_file)
    wf = open(output_file, "w")
    for index, row in df.iterrows():
        spk_id = row["filename"].split("/")[-2]
        utt_id = spk_id + "-" + row["filename"].split("/")[-1][:-4]
        wav_path = os.path.join(dataset_dir, spk_id, utt_id.split("-")[-1] + ".wav")
        wf.writelines([utt_id + " " + wav_path + "\n"])
    wf.close()
    os.system(r"sed -i 's/r$//' " + output_file)
    os.system("sort " + output_file + " -o " + output_file)

if __name__ == '__main__':
    revising_csv()
    # write_utt2spk()
    # write_wavscp()
