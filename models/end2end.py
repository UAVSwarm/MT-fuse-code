import torch
from torch import nn
from models.encode import encode
from models.decode import decode


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.encode = encode()
        self.decode = decode()

    def forward(self, vis, inf):
        f = torch.cat((vis, inf), dim=1)
        x1, x2, x3, x4 = self.encode(f)
        f_ = self.decode(x1, x2, x3, x4)
        return f_


if __name__ == '__main__':
   img = torch.randn(1,1,8,8).cuda()
   model = model().cuda()
   print(model(img, img))
