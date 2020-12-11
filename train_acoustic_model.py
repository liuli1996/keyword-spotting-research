import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import SpeechDataset, TDNN_SpeechDataset, E2E_DataLoader, E2E_SpeechDataset
from tensorboardX import SummaryWriter
import pandas as pd
from helper.utils import weights_init_, print_eval
from model import LSTM, DSCNN, TDNN, GRU_AcousticModel
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
    model_name = 'e2e_asr_free_am'
    config['epoch'] = 60
    config['batch_size'] = 40

    config['train_csv'] = './csv_win/train_mfcc.csv'
    config['val_csv'] = './csv_win/test_mfcc.csv'

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
    train_dataset = E2E_SpeechDataset(csv_file=r"./csv_win/train_mfcc.csv",
                                      cmvn_file=r"./cmvn_stats/TIMIT_train_global_stats.pkl")
    test_dataset = E2E_SpeechDataset(csv_file=r"./csv_win/user_defined_keywords_for_testing.csv.csv",
                                     cmvn_file=r"./cmvn_stats/TIMIT_train_global_stats.pkl")

    # model
    model = GRU_AcousticModel()
    model = model.to(config['device'])
    model.apply(weights_init_)

    # optimizer
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = Frame_MSE_Loss()

    # data_loader
    min_loss = 10000
    step = 0
    train_loader = E2E_DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4)
    test_loader = E2E_DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4)

    for epoch in range(config['epoch']):
        if epoch % 2 == 0:
            model.eval()
            loss_sum = 0
            with torch.no_grad():
                for idx, (data, input_lens) in enumerate(test_loader):
                    data = data.to(config['device'])
                    scores, _ = model(data)
                    loss = criterion(scores, data, input_lens)
                    print("{}/{} dev loss: {}".format(idx + 1, len(test_dataset), loss))
                    loss_sum += loss
                avg_loss = loss_sum / len(test_dataset)
            print("avg dev loss: {}, min loss: {}".format(avg_loss, min_loss))

            if avg_loss < min_loss:
                model.save(config['output_path'] + '/model_best.pt')
                min_loss = avg_loss

            writer1.add_scalar('Test/Loss', loss_sum, epoch)
            model.save(config['output_path'] + '/model_ep' + str(epoch) + '.pt')
            print('model saved !')

        model.train()
        for data, input_lens in train_loader:
            step += 1
            optimizer.zero_grad()
            data = data.to(config['device'])

            scores, _ = model(data)
            loss = criterion(scores, data, input_lens)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            writer1.add_scalar('Train/Loss', loss, step)
            print("step #{}, training loss: {}".format(step, loss))

def evaluate(model_file):
    config = configuration()
    model = TDNN(n_labels=12).cuda()
    model.load(model_file)
    test_dataset = TDNN_SpeechDataset(config["val_csv"], config["wanted_words"])
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8)
    model.eval()
    with torch.no_grad():
        positive_examples = 0
        for data, labels in test_loader:
            data = data.to(config['device'])
            labels = labels.to(config['device'])
            scores, _ = model(data)
            if torch.argmax(scores) == labels:
                positive_examples += 1
        avg_acc = positive_examples / len(test_dataset)
        print("final dev accuracy: {}".format(avg_acc))

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
    # evaluate("./models_pt/TDNN/model_best.pt")
