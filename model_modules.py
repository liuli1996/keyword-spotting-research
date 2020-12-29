import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AMSoftmax(nn.Module):
    def __init__(self, n_maps, n_labels, s=30, m=0.2):
        super().__init__()
        self.embedding_size = n_maps
        self.num_classes = n_labels
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.Tensor(n_labels, n_maps))
        nn.init.normal_(self.weights, mean=0, std=0.01)  # initial

    def forward(self, x, labels):
        """

        :param x:
        :param labels: torch.LongTensor, shape=()
        :return:
        """
        assert x.size(1) == self.embedding_size
        x = F.linear(x, F.normalize(self.weights))
        margin = torch.zeros_like(x)
        if labels != None:
            margin.scatter_(1, labels.view(-1, 1), self.m)  # (dim, index, src) scatter "src" into "margin" following index along dim
        x = self.s * (x - margin)
        return x

class DS_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DS_Convolution, self).__init__()
        self.dw_block = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.pw_block = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.dw_block(x)
        y = self.pw_block(y)
        return y

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

class Highway(nn.Module):
    """Highway network"""

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = torch.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1 - t, x)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right,0)
    y = torch.cat([y_left, torch.autograd.Variable(torch.ones(1)).cuda(),y_right])
    return y

class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs):
        super(SincConv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        """
        :param x: [batch_size, 1, wav_length]
        :return:  [batch_size, self.N_filt, wav_length - self.Filt_dim]
        """
        filters = torch.autograd.Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = torch.autograd.Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()

        min_freq = 50.0
        min_band = 50.0

        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N)
        window = torch.autograd.Variable(window.float().cuda())

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)

            band_pass = band_pass / torch.max(band_pass)

            filters[i, :] = band_pass.cuda() * window

        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim))

        return out

class LogCompression(nn.Module):
    def __init__(self):
        super(LogCompression, self).__init__()

    def forward(self, x):
        return torch.log(torch.abs(x) + 1)

if __name__ == '__main__':
    model = LogCompression()
    x = torch.rand((10))
    y = model(x)
    pass
