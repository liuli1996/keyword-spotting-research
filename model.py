import torch
import torch.nn as nn
import torch.nn.functional as F
from model_modules import AMSoftmax, DS_Convolution, MLP, Highway, SincConv, LogCompression
from torchsummary import summary
import torch
import random

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))  # 将模型数据加载到cpu

class TPool2(SerializableModule):
    def __init__(self, n_labels, n_maps=94, embedding_size=128):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=n_maps, kernel_size=(21, 8), stride=1, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(n_maps))
        self.max_pool2d_1 = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3), padding=0))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=n_maps, out_channels=n_maps, kernel_size=(6, 4), stride=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(n_maps))
        self.max_pool2d_2 = nn.Sequential(nn.MaxPool2d(kernel_size=1, stride=1, padding=0))
        self.lin = nn.Linear(in_features=35 * 8 * n_maps, out_features=32, bias=False)
        self.dnn = nn.Sequential(nn.Linear(in_features=32, out_features=embedding_size, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(embedding_size))
        self.softmax = nn.Linear(embedding_size, n_labels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool2d_1(x)
        x = self.conv_2(x)
        x = self.max_pool2d_2(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        embedding = self.dnn(x)
        x = self.softmax(embedding)
        return x, embedding

class TradFPool3(SerializableModule):
    def __init__(self, n_labels, n_maps=64, embedding_size=128):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=n_maps, kernel_size=(20, 8), stride=1, bias=False),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(n_maps))
        self.max_pool2d_1 = nn.Sequential(nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=n_maps, out_channels=n_maps, kernel_size=(10, 4), stride=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.BatchNorm2d(n_maps))
        self.max_pool2d_2 = nn.Sequential(nn.MaxPool2d(kernel_size=1, stride=1, padding=0))
        self.lin = nn.Linear(in_features=73 * 8 * n_maps, out_features=32, bias=False)
        self.dnn = nn.Sequential(nn.Linear(in_features=32, out_features=embedding_size, bias=False),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(embedding_size))
        self.softmax = nn.Linear(embedding_size, n_labels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool2d_1(x)
        x = self.conv_2(x)
        x = self.max_pool2d_2(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        embedding = self.dnn(x)
        x = self.softmax(embedding)
        return x, embedding

class Res15(SerializableModule):
    def __init__(self, n_labels=12, n_maps=45, n_layers=13):
        # 继承SerializableModule的属性
        super().__init__()
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
            bias=False) for i in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))

            if i == 0:
                old_x = y

            if i > 0:
                y = getattr(self, "bn{}".format(i))(y)

            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y

        # view()相当于reshape
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        # 在时间上取平均
        embedding = torch.mean(x, 2)
        x = self.output(embedding)
        return x, embedding

class LSTM(SerializableModule):
    def __init__(self, n_labels):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.linear = nn.Linear(128, n_labels, bias=False)

    def forward(self, x):
        embedding, (h_n, h_c) = self.lstm(x, None)
        y = self.linear(embedding[:, -1, :])
        return y, embedding

class AMResNet(SerializableModule):
    def __init__(self, n_labels, m=0.2, s=30, n_maps=45, n_layers=13):
        # 继承SerializableModule的属性
        super().__init__()
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers
        self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
            bias=False) for i in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = AMSoftmax(n_maps, n_labels, m=m, s=s)

    def forward(self, x, labels=None):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))

            if i == 0:
                old_x = y

            if i > 0:
                y = getattr(self, "bn{}".format(i))(y)

            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y

        # view()相当于reshape
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        # 在时间上取平均
        embedding = torch.mean(x, 2)
        embedding = F.normalize(embedding, dim=1)
        x = self.output(embedding, labels)
        return x, embedding

class DSCNN(SerializableModule):
    def __init__(self, n_labels=12):
        super(DSCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(25, 5), padding=(12, 2)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.ds_block1 = DS_Convolution(in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
        self.ds_block2 = DS_Convolution(in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
        self.ds_block3 = DS_Convolution(in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
        self.ds_block4 = DS_Convolution(in_channels=64, out_channels=64, kernel_size=(25, 5), padding=(12, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc1 = nn.Linear(in_features=64, out_features=n_labels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.ds_block1(y)
        y = self.ds_block2(y)
        y = self.ds_block3(y)
        y = self.ds_block4(y)
        embedding = self.avg_pool(y)
        embedding = embedding.squeeze(-1).squeeze(-1)
        y = self.fc1(embedding)
        return y, embedding

class TDNN(SerializableModule):
    def __init__(self, n_labels, win_len=11, n_feats=41, n_phones=48, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(TDNN, self).__init__()
        self.feature_window = nn.Unfold(kernel_size=((win_len), n_feats))
        self.PHONE_NN = MLP(dnn_lay=[128, 128, 128],
                            dnn_drop=[0.15, 0.15, 0.15],
                            dnn_use_batchnorm=[True, True, True],
                            dnn_use_laynorm=[False, False, False],
                            dnn_use_laynorm_inp=False,
                            dnn_use_batchnorm_inp=False,
                            dnn_act=["relu", "relu", "relu"],
                            inp_dim=451).to(device)
        self.PHONE_NN.load_state_dict(torch.load(r"C:\Users\Administrator\PycharmProjects\kws_end2end\pytorch-kaldi\final_architecture1.pkl")["model_par"])
        self.PHONE_FC = MLP(dnn_lay=[n_phones],
                            dnn_drop=[0.0],
                            dnn_use_batchnorm=[False],
                            dnn_use_laynorm=[False],
                            dnn_use_laynorm_inp=False,
                            dnn_use_batchnorm_inp=False,
                            dnn_act=["linear"],
                            inp_dim=128).to(device)
        self.PHONE_FC.load_state_dict(torch.load(r"C:\Users\Administrator\PycharmProjects\kws_end2end\pytorch-kaldi\final_architecture3.pkl")["model_par"])
        self.WORD_NN = nn.Sequential(nn.Linear(in_features=17 * n_phones, out_features=64),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.Linear(in_features=64, out_features=n_labels))
    def forward(self, x):
        y = x.unsqueeze(1)
        y = self.feature_window(y)
        y = y.permute(0, 2, 1)
        batch_size, n_frames, n_feats = y.shape[0], y.shape[1], y.shape[2]
        y = y.reshape(-1, n_feats)
        y = self.PHONE_NN(y)
        y = self.PHONE_FC(y)
        y = y.reshape(batch_size, n_frames, -1)
        y = F.max_pool2d(y, kernel_size=(5, 1), stride=(4, 1))  # [batch, 17, n_feats]
        y = y.reshape(batch_size, -1)
        y = self.WORD_NN(y)
        return y, y

class GRU_AcousticModel(SerializableModule):
    def __init__(self, feat_dims=13, embedding_size=300, n_layers=1, device="cuda"):
        super(GRU_AcousticModel, self).__init__()
        self.device = device
        self.feat_dims = feat_dims
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.encoder = nn.GRU(input_size=feat_dims,
                              hidden_size=embedding_size,
                              num_layers=n_layers,
                              bias=True,
                              batch_first=True)
        self.fc = nn.Sequential(nn.Linear(in_features=embedding_size, out_features=embedding_size),
                                nn.ReLU())
        self.decoder = nn.GRU(input_size=embedding_size,
                              hidden_size=feat_dims,
                              num_layers=1,
                              bias=True,
                              batch_first=True)

    def forward(self, x, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)
        max_len = x.size(1)
        outputs = torch.zeros(batch_size, max_len, self.feat_dims).to(self.device)
        encoder_output, hidden = self.encoder(x)
        embedding = self.fc(encoder_output[:, -1, :]).unsqueeze(1)
        last_hidden = torch.zeros(1, batch_size, self.feat_dims).to(self.device)
        for t in range(max_len):
            output, hidden = self.decoder(embedding, last_hidden)
            outputs[:, t: t+1, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            last_hidden = x[:, t, :].unsqueeze(0).contiguous() if teacher_force else hidden
        return outputs, embedding

class charLM(nn.Module):
    """CNN + highway network + LSTM
    # Input:
        4D tensor with shape [batch_size, in_channel, height, width]
    # Output:
        2D Tensor with shape [batch_size, vocab_size]
    # Arguments:
        char_emb_dim: the size of each character's embedding
        word_emb_dim: the size of each word's embedding
        vocab_size: num of unique words
        num_char: num of characters
        use_gpu: True or False
    """

    def __init__(self, char_emb_dim, word_emb_dim, vocab_size, num_char, use_gpu):
        super(charLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size

        # char embedding layer
        self.char_embed = nn.Embedding(num_char, char_emb_dim)

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(1, 3) for i in range(300)]

        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,  # in_channel
                    out_channel,  # out_channel
                    kernel_size=(char_emb_dim, filter_width),  # (height, width)
                    bias=True
                )
            )

        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        # LSTM
        self.lstm_num_layers = 1

        self.lstm = nn.GRU(input_size=self.highway_input_dim,
                           hidden_size=self.word_emb_dim,
                           num_layers=self.lstm_num_layers,
                           bias=True,
                           dropout=0.5,
                           batch_first=True)

        # output layer
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

        if use_gpu is True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.highway1 = self.highway1.cuda()
            self.highway2 = self.highway2.cuda()
            self.lstm = self.lstm.cuda()
            self.dropout = self.dropout.cuda()
            self.char_embed = self.char_embed.cuda()
            self.linear = self.linear.cuda()
            self.batch_norm = self.batch_norm.cuda()

    def forward(self, x, hidden=None):
        # Input: Variable of Tensor with shape [num_seq, seq_len, max_word_len+2]
        # Return: Variable of Tensor with shape [num_words, len(word_dict)]
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len+2]

        x = self.char_embed(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]

        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        # [num_seq*seq_len, 1, max_word_len+2, char_emb_dim]

        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway1(x)
        embedding = self.highway2(x)
        # [num_seq*seq_len, total_num_filters]

        x = embedding.contiguous().view(lstm_batch_size, lstm_seq_len, -1)
        # [num_seq, seq_len, total_num_filters]

        x, hidden = self.lstm(x)
        # [seq_len, num_seq, hidden_size]

        x = self.dropout(x)
        # [seq_len, num_seq, hidden_size]

        x = x.contiguous().view(lstm_batch_size * lstm_seq_len, -1)
        # [num_seq*seq_len, hidden_size]

        x = self.linear(x)
        # [num_seq*seq_len, vocab_size]
        return x, embedding

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = torch.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)
            chosen = chosen.squeeze(2)
            # (batch_size, out_channel)
            chosen_list.append(chosen)

        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, dim=1)

class E2E_ASR_FREE(SerializableModule):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(E2E_ASR_FREE, self).__init__()
        self.acoustic_model = GRU_AcousticModel().to(device)
        self.acoustic_model.load("./models_pt/e2e_asr_free_am/model_best.pt")
        self.language_model = charLM(char_emb_dim=50,
                                     word_emb_dim=256,
                                     vocab_size=49,
                                     num_char=49,
                                     use_gpu=True if device=="cuda" else False)
        self.language_model.load_state_dict(torch.load("./models_pt/charLM/model.pt"))
        for paras in self.acoustic_model.parameters():
            paras.requires_grad = False
        for paras in self.language_model.parameters():
            paras.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(in_features=600, out_features=256),
                                nn.ReLU())
        self.output = nn.Linear(in_features=256, out_features=2)

    def forward(self, x, label):
        _, am_embedding = self.acoustic_model(x)
        _, lm_embedding = self.language_model(label)
        x = self.fc(torch.cat([am_embedding.squeeze(1), lm_embedding], dim=1))
        y = self.output(x)
        return y, x

class SincNet(SerializableModule):
    def __init__(self):
        super(SincNet, self).__init__()
        self.sinc_block = nn.Sequential(
            SincConv(N_filt=40, Filt_dim=101, fs=16000),
            LogCompression(),
            nn.BatchNorm2d(40),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        y = self.sinc_conv(x)
        return y

if __name__ == '__main__':
    model = E2E_ASR_FREE().cuda()
    x = torch.ones((64, 1, 23), dtype=torch.int64).cuda()
    x = x * 10
    y = torch.Tensor(64, 100, 13).cuda()
    z = model(y, x)
    pass
