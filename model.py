import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from resnet1d import BaseNet

# Residual Block
class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, downsample=False, pooling=False
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
        out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


class attention(nn.Module):
    def __init__(self, config):
        super(attention, self).__init__()
        self.att_dim = 256
        self.W = nn.Parameter(torch.randn(256, self.att_dim))
        self.V = nn.Parameter(torch.randn(self.att_dim, 1))
        self.scale = self.att_dim ** -0.5

    def forward(self, x):
        x = x.permute(0, 2, 1)
        e = torch.matmul(x, self.W)
        e = torch.matmul(torch.tanh(e), self.V)
        e = e * self.scale
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        return x


## Model code
class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        self.time_model = BaseNet()
        self.attention = attention(config)

    def forward(self, x):
        time = self.time_model(x)
        time_feats = self.attention(time)

        return time_feats


class ft_loss(nn.Module):
    def __init__(self, chkpoint_pth, config, device):
        super(ft_loss, self).__init__()

        self.eeg_encoder = encoder(config)
        self.classes = 5

        chkpoint = torch.load(chkpoint_pth, map_location=device)
        eeg_dict = chkpoint["eeg_model_state_dict"]

        self.eeg_encoder.load_state_dict(eeg_dict)
        self.lin = nn.Linear(256, self.classes)

    def forward(self, time_dat):

        time_feats = self.eeg_encoder(time_dat)
        x = self.lin(time_feats)
        return x


class predict_model(nn.Module):
    def __init__(self, config):
        super(predict_model, self).__init__()

        self.classes = 5
        self.eeg_encoder = encoder(config)      
        self.lin = nn.Linear(256, self.classes)

        ckt = torch.load(config.save_path, map_location=config.device)
        
        keys = ["spect_model.conv1.0.weight", "spect_model.conv1.0.bias", "spect_model.conv1.1.weight", "spect_model.conv1.1.bias", "spect_model.conv1.1.running_mean",\
            "spect_model.conv1.1.running_var", "spect_model.conv1.1.num_batches_tracked", "spect_model.conv2.conv1.weight", "spect_model.conv2.conv1.bias", "spect_model.conv2.bn1.weight", "spect_model.conv2.bn1.bias", "spect_model.conv2.bn1.running_mean", "spect_model.conv2.bn1.running_var", "spect_model.conv2.bn1.num_batches_tracked", "spect_model.conv2.conv2.weight", "spect_model.conv2.conv2.bias", "spect_model.conv2.bn2.weight", "spect_model.conv2.bn2.bias", "spect_model.conv2.bn2.running_mean", "spect_model.conv2.bn2.running_var", "spect_model.conv2.bn2.num_batches_tracked", "spect_model.conv2.downsample.0.weight", "spect_model.conv2.downsample.0.bias", "spect_model.conv2.downsample.1.weight", "spect_model.conv2.downsample.1.bias", "spect_model.conv2.downsample.1.running_mean", "spect_model.conv2.downsample.1.running_var", "spect_model.conv2.downsample.1.num_batches_tracked", "spect_model.conv3.conv1.weight", "spect_model.conv3.conv1.bias", "spect_model.conv3.bn1.weight", "spect_model.conv3.bn1.bias", "spect_model.conv3.bn1.running_mean", "spect_model.conv3.bn1.running_var", "spect_model.conv3.bn1.num_batches_tracked", "spect_model.conv3.conv2.weight", "spect_model.conv3.conv2.bias", "spect_model.conv3.bn2.weight", "spect_model.conv3.bn2.bias", "spect_model.conv3.bn2.running_mean", "spect_model.conv3.bn2.running_var", "spect_model.conv3.bn2.num_batches_tracked", "spect_model.conv3.downsample.0.weight", "spect_model.conv3.downsample.0.bias", "spect_model.conv3.downsample.1.weight", "spect_model.conv3.downsample.1.bias", "spect_model.conv3.downsample.1.running_mean", "spect_model.conv3.downsample.1.running_var", "spect_model.conv3.downsample.1.num_batches_tracked", "spect_model.conv4.conv1.weight", "spect_model.conv4.conv1.bias", "spect_model.conv4.bn1.weight", "spect_model.conv4.bn1.bias", "spect_model.conv4.bn1.running_mean", "spect_model.conv4.bn1.running_var", "spect_model.conv4.bn1.num_batches_tracked", "spect_model.conv4.conv2.weight", "spect_model.conv4.conv2.bias", "spect_model.conv4.bn2.weight", "spect_model.conv4.bn2.bias", "spect_model.conv4.bn2.running_mean", "spect_model.conv4.bn2.running_var", "spect_model.conv4.bn2.num_batches_tracked", "spect_model.conv4.downsample.0.weight", "spect_model.conv4.downsample.0.bias", "spect_model.conv4.downsample.1.weight", "spect_model.conv4.downsample.1.bias", "spect_model.conv4.downsample.1.running_mean", "spect_model.conv4.downsample.1.running_var", "spect_model.conv4.downsample.1.num_batches_tracked", "spect_model.fc.0.weight", "spect_model.fc.0.bias", "spect_model.fc.2.weight", "spect_model.fc.2.bias", "spect_model.sup.0.weight", "spect_model.sup.0.bias", "spect_model.sup.2.weight", "spect_model.sup.2.bias", "spect_model.byol_mapping.0.weight", "spect_model.byol_mapping.0.bias", "spect_model.byol_mapping.2.weight", "spect_model.byol_mapping.2.bias"]
        
        for i in keys:
            del ckt['eeg_model_state_dict'][i]
        
        self.eeg_encoder.load_state_dict(ckt["eeg_model_state_dict"])
        
        self.lin.load_state_dict(ckt["lin_layer_state_dict"])
              
    def forward(self, x):

        x = self.eeg_encoder(x)
        x = self.lin(x)
        return x