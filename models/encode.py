import torch
from torch import nn
from models.common import reflect_conv

class BEAD(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(BEAD, self).__init__()
        self.conv = reflect_conv(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, pad=1)
        self.resconv = reflect_conv(in_channels=inchannel, out_channels=outchannel, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        x_in = x
        act = nn.LeakyReLU()
        x = self.conv(x)
        x = act(x)
        res_x = self.resconv(x_in)
        res_x = act(res_x)

        x = x * res_x
        x = x + res_x

        return x




class encode(nn.Module):
    def __init__(self):
        super(encode, self).__init__()
        self.conv1 = BEAD(inchannel=2, outchannel=16)
        self.conv2 = BEAD(inchannel=16, outchannel=32)
        self.conv3 = BEAD(inchannel=32, outchannel=64)
        self.conv4 = BEAD(inchannel=64, outchannel=128)

    def forward(self, x):
        act = nn.LeakyReLU()
        x1 = act(self.conv1(x))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        x4 = act(self.conv4(x3))
        return x1, x2, x3, x4



if __name__ == '__main__':
   img = torch.randn(1,2,8,8).cuda()
   model = encode().cuda()
   print(model(img))
