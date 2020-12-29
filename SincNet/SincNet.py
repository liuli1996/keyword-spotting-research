import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right,0)
    y = torch.cat([y_left, torch.autograd.Variable(torch.ones(1)).cuda(),y_right])
    return y

class sinc_conv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs):
        super(sinc_conv, self).__init__()

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

if __name__ == '__main__':
    model = sinc_conv(80, 251, 16000).cuda()
    x = torch.Tensor(64, 1, 3200).cuda()
    y = model(x)
    # x = get_mel_points(16000, 80)
    # y = get_bands(x, 16000)
    pass
