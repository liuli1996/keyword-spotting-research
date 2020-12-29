import torch
import torch.nn as nn
import torch.nn.functional as F

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class SpeechResModel(SerializableModule):
    def __init__(self, config):
        # 继承SerializableModule的属性
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.dense = nn.Sequential(nn.Linear(n_maps, n_labels),
                                   nn.Softmax(dim=-1))

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y

            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        # view()相当于reshape
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        # 在时间上取平均
        embedding = torch.mean(x, 2)
        x = self.dense(embedding)
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
        x = x.unsqueeze(0)
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