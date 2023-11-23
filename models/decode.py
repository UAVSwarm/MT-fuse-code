import torch
from torch import nn
from models.common import reflect_conv
from models.transformer import BaseFeatureExtraction as lite_tf
from models.transformer import gradient_all_shape


class SSE(nn.Module):
    def __init__(self, inchannel, kernel_size=3):
        super(SSE, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.lite_tf = lite_tf(dim=2, num_heads=2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.g = gradient_all_shape(in_channel=inchannel)

    def forward(self, x):  # x.size() 30,40,50,30
        x_in = x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x_cnn = self.conv1(x)  # 30,1,50,30 cnn model
        x_tf = self.conv2(self.lite_tf(x)) #tf model
        x = x_tf + x_cnn
        return self.sigmoid(x)*x + self.g(x_in)  # 30,1,50,30



class decode(nn.Module):
    def __init__(self):
        super(decode, self).__init__()
        self.conv1 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=16, kernel_size=1, out_channels=1, stride=1, pad=0)

        self.sse1 = SSE(inchannel=16)
        self.see2 = SSE(inchannel=32)
        self.see3 = SSE(inchannel=64)

    def forward(self, x1, x2, x3, x4):
        act = nn.LeakyReLU()
        d3 = act(self.conv1(x4))
        d2 = act(self.conv2(d3+self.see3(x3)))
        d1 = act(self.conv3(d2+self.see2(x2)))
        d = act(self.conv4(d1+self.sse1(x1)))
        return d

