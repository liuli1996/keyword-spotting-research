import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import SpeechDataset, TDNN_SpeechDataset
from tensorboardX import SummaryWriter
import pandas as pd
from helper.utils import weights_init, print_eval
from model import LSTM, DSCNN, TDNN
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def configuration():
    config = {}
    model_name = 'TDNN'
    config['epoch'] = 90
    config['batch_size'] = 100

    config['train_csv'] = './csv_linux/train_fulllabel.csv'
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
    train_dataset = TDNN_SpeechDataset(config['train_csv'], config['wanted_words'])
    test_dataset = TDNN_SpeechDataset(config['val_csv'], config['wanted_words'])

    # model
    model = TDNN(n_labels=12)
    model = model.to(config['device'])
    model.apply(weights_init)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # data_loader
    step = 0
    best_acc = 0
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4)

    for epoch in range(config['epoch']):
        for data, labels in train_loader:
            step += 1
            optimizer.zero_grad()
            data = data.to(config['device'])
            labels = labels.to(config['device'])

            scores, _ = model(data)
            loss = criterion(scores, labels)

            loss.backward()
            optimizer.step()
            writer1.add_scalar('Train/Loss', loss, step)
            print_eval("train step #{}".format(step), scores, labels, loss)

            if step % 800 == 0:
                model.save(config['output_path'] + '/model_' + str(step) + '.pt')
                print('model saved !')
                model.eval()
                with torch.no_grad():
                    accs = []
                    for data, labels in test_loader:
                        data = data.to(config['device'])
                        labels = labels.to(config['device'])
                        scores, _ = model(data)
                        loss = criterion(scores, labels)
                        accs.append(print_eval("dev", scores, labels, loss))
                    avg_acc = np.mean(accs)
                    print("final dev accuracy: {}, best accuracy: {}".format(avg_acc, best_acc))
                    if avg_acc > best_acc:
                        model.save(config['output_path'] + '/model_best.pt')
                        best_acc = avg_acc
                    writer1.add_scalar('Test/Acc', avg_acc, step)
                model.train()

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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train()
    # evaluate("./models_pt/TDNN/model_best.pt")
