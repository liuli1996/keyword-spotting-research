import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import SpeechDataset, TDNN_SpeechDataset, E2E_DataLoader, E2E_SpeechDataset
from tensorboardX import SummaryWriter
import pandas as pd
from helper.utils import weights_init, print_eval
from model import LSTM, DSCNN, TDNN, GRU_AcousticModel, E2E_ASR_FREE
import numpy as np
import random
import torch.nn.functional as F

class Frame_MSE_Loss(nn.Module):
    def __init__(self):
        super(Frame_MSE_Loss, self).__init__()

    def forward(self, scores, label, input_lens):
        loss = 0
        for i in range(len(input_lens)):
            loss += F.mse_loss(scores[i, :input_lens[i], :], label[i, :input_lens[i], :])
        return loss / len(input_lens)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def configuration():
    config = {}
    model_name = 'e2e_asr_free'
    config['epoch'] = 60
    config['batch_size'] = 100

    config['train_csv'] = './csv_win/train_mfcc_e2e.csv'
    config['val_csv'] = './csv_win/test_mfcc_e2e.csv'

    df = pd.read_csv(config['train_csv'])
    config['wanted_words'] = df['label'].unique()
    print("The number of wanted words is {}".format(len(config["wanted_words"])))
    config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config['output_path'] = './models_pt/' + model_name
    os.makedirs(config['output_path'], exist_ok=True)
    config['logdir'] = './models_pt/' + model_name + '/log'
    return config

def train():
    setup_seed(20)
    config = configuration()

    writer1 = SummaryWriter(config['logdir'] + '/Train')

    # dataset
    train_dataset = E2E_SpeechDataset(csv_file=config['train_csv'],
                                      cmvn_file="./cmvn_stats/TIMIT_train_global_stats.pkl")
    test_dataset = E2E_SpeechDataset(csv_file=config['val_csv'],
                                     cmvn_file="./cmvn_stats/TIMIT_train_global_stats.pkl")

    # model
    model = E2E_ASR_FREE()
    model = model.to(config['device'])
    model.apply(weights_init)

    # optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # data_loader
    best_acc = 0
    step = 0
    train_loader = E2E_DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4)
    test_loader = E2E_DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4)

    for epoch in range(config['epoch']):
        if epoch % 2 == 0:
            model.eval()
            loss_sum = 0
            with torch.no_grad():
                accs = []
                for _, (data, labels, is_same, input_lens) in enumerate(test_loader):
                    data = data.to(config['device'])
                    labels = labels.to(config['device'])
                    is_same = is_same.to(config['device'])
                    scores, _ = model(data, labels)
                    loss = criterion(scores, is_same)
                    accs.append(print_eval("dev", scores, is_same, loss))
                avg_acc = np.mean(accs)
                print("final dev accuracy: {}, best accuracy: {}".format(avg_acc, best_acc))

            if avg_acc > best_acc:
                model.save(config['output_path'] + '/model_best.pt')
                best_acc = avg_acc

            writer1.add_scalar('Test/Loss', loss_sum, epoch)
            model.save(config['output_path'] + '/model_ep' + str(epoch) + '.pt')
            print('model saved !')

        model.train()
        for data, labels, is_same, input_lens in train_loader:
            step += 1
            optimizer.zero_grad()
            data = data.to(config['device'])
            labels = labels.to(config['device'])
            is_same = is_same.to(config['device'])

            scores, _ = model(data, labels)
            loss = criterion(scores, is_same)

            loss.backward()
            optimizer.step()
            writer1.add_scalar('Train/Loss', loss, step)
            print_eval("train step #{}".format(step), scores, is_same, loss)

def evaluate(model_file):
    config = configuration()
    model = E2E_ASR_FREE().cuda()
    model.load(model_file)
    test_dataset = E2E_SpeechDataset(csv_file="./csv_win/user_defined_keywords_for_testing_e2e.csv",
                                     cmvn_file="./cmvn_stats/TIMIT_train_global_stats.pkl")
    test_loader = E2E_DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8)

    scores = []
    roc_labels = []
    model.eval()
    with torch.no_grad():
        positive_examples = 0
        for data, labels, is_same, input_lens in test_loader:
            data = data.to(config['device'])
            labels = labels.to(config['device'])
            is_same = is_same.to(config['device'])
            score, _ = model(data, labels)
            score = F.softmax(score)
            if torch.argmax(score) == is_same:
                positive_examples += 1
            scores.append(torch.max(score, dim=1)[0].item())
            roc_labels.append(is_same.item())
    avg_acc = positive_examples / len(test_dataset)
    print("final dev accuracy: {}".format(avg_acc))

    scores = np.array(scores)
    roc_labels = np.array(roc_labels)
    model_name = "e2e_asr_free"
    np.save("./roc_data/" + model_name + "_scores.npy", scores)
    np.save("./roc_data/" + model_name + "_labels.npy", roc_labels)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # train()
    evaluate("./models_pt/e2e_asr_free/model_best.pt")
