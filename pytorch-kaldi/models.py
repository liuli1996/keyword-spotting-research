import torch
import torch.nn as nn
import numpy as np
import torchsummary
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initialized like this, but not used in forward!

class MLP(nn.Module):
    def __init__(self, dnn_lay, dnn_drop, dnn_use_batchnorm, dnn_use_laynorm, dnn_use_laynorm_inp, dnn_use_batchnorm_inp, dnn_act, inp_dim):
        super(MLP, self).__init__()
        self.input_dim = inp_dim
        self.dnn_lay = dnn_lay
        self.dnn_drop = dnn_drop
        self.dnn_use_batchnorm = dnn_use_batchnorm
        self.dnn_use_laynorm = dnn_use_laynorm
        self.dnn_use_laynorm_inp = dnn_use_laynorm_inp
        self.dnn_use_batchnorm_inp = dnn_use_batchnorm_inp
        self.dnn_act = dnn_act

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                    np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                )
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.dnn_use_batchnorm_inp):

            x = self.bn0((x))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))

        return x

class TDNN(nn.Module):
    def __init__(self, win_len=11, n_feats=41, n_phones=48, device="cuda" if torch.cuda.is_available() else "cpu"):
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
                                     nn.Linear(in_features=64, out_features=12),
                                     nn.Softmax())
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
        return y

if __name__ == '__main__':
    model = TDNN()
    x = torch.Tensor(64, 1, 80, 41).cuda()
    y = model(x)
    pass