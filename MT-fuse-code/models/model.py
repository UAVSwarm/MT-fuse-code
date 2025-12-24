
import torch
from torch import nn
from models.DBB import DiverseBranchBlock

class Fusion(nn.Module):
    def __init__(self, deploy):
        super(Fusion, self).__init__()
        self.deploy = deploy
        self.conv1 = DiverseBranchBlock(in_channels=2, out_channels=1, stride=1, kernel_size=3, padding=1)
        self.conv2 = DiverseBranchBlock(in_channels=3, out_channels=1, stride=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vi, ir):
        if self.deploy:
            self.conv1.switch_to_deploy()
            self.conv1.deploy = True
            self.conv2.switch_to_deploy()
            self.conv2.deploy = True
        same = (vi+ir)
        diff1 = (vi-ir)/2
        diff2 = (ir-vi)/2
        avg_out_1 = torch.mean(diff1, dim=1, keepdim=True)
        max_out_1, _ = torch.max(diff2, dim=1, keepdim=True)
        out_1 = torch.cat([avg_out_1, max_out_1], dim=1)
        out_1 = self.conv1(out_1)
        avg_out_2 = torch.mean(diff2, dim=1, keepdim=True)
        max_out_2, _ = torch.max(diff2, dim=1, keepdim=True)
        out_2 = torch.cat([avg_out_2, max_out_2], dim=1)
        out = torch.cat([out_1, out_2], dim=1)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return same+diff1*out+diff2*(1-out)




class FusionNet(nn.Module):
    def __init__(self, deploy=False):
        super(FusionNet, self).__init__()
        self.deploy = deploy
        self.conv1 = DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv11 = DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv22 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv33 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv44 = DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)


        self.conv5 = DiverseBranchBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = DiverseBranchBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = DiverseBranchBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = DiverseBranchBlock(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, vi, ir):
        if self.deploy:
            self.conv1.switch_to_deploy()
            self.conv1.deploy = True
            self.conv2.switch_to_deploy()
            self.conv2.deploy = True
            self.conv3.switch_to_deploy()
            self.conv3.deploy = True
            self.conv4.switch_to_deploy()
            self.conv4.deploy = True
            self.conv11.switch_to_deploy()
            self.conv11.deploy = True
            self.conv22.switch_to_deploy()
            self.conv22.deploy = True
            self.conv33.switch_to_deploy()
            self.conv33.deploy = True
            self.conv44.switch_to_deploy()
            self.conv44.deploy = True
            self.conv5.switch_to_deploy()
            self.conv5.deploy = True
            self.conv6.switch_to_deploy()
            self.conv6.deploy = True
            self.conv7.switch_to_deploy()
            self.conv7.deploy = True
            self.conv8.switch_to_deploy()
            self.conv8.deploy = True
        f = torch.cat([vi, ir], dim=1)
        f = self.conv1(f)*self.conv11(f)
        f = self.conv2(f) * self.conv22(f)
        f = self.conv3(f) * self.conv33(f)
        f = self.conv4(f) * self.conv44(f)

        f = self.conv5(f)
        f = self.conv6(f)
        f = self.conv7(f)
        f= self.conv8(f)
        return f

class FusionNet_dual(nn.Module):
    def __init__(self, deploy=False):
        super(FusionNet_dual, self).__init__()
        self.deploy = deploy
        self.conv1 = DiverseBranchBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv11 = DiverseBranchBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv22 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv33 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv44 = DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv5 = DiverseBranchBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = DiverseBranchBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = DiverseBranchBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = DiverseBranchBlock(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)




    def forward(self, vi, ir):

        vi, ir = self.conv1(vi) * self.conv11(vi), self.conv1(ir) * self.conv11(ir)
        vi, ir = self.conv2(vi) * self.conv22(vi), self.conv2(ir) * self.conv22(ir)
        vi, ir = self.conv3(vi) * self.conv33(vi), self.conv3(ir) * self.conv33(ir)
        vi, ir = self.conv4(vi) * self.conv44(vi), self.conv4(ir) * self.conv44(ir)
        f = vi+ir

        f =self.conv5(f)
        f=self.conv6(f)
        f=self.conv7(f)
        f=self.conv8(f)
        return f




class FusionNet_5(nn.Module):
    def __init__(self, deploy=False):
        super(FusionNet_5, self).__init__()
        self.deploy = deploy
        self.conv1 = DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = DiverseBranchBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv11 = DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv22 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv33 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv44 = DiverseBranchBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv55 = DiverseBranchBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv6 = DiverseBranchBlock(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = DiverseBranchBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8 = DiverseBranchBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv9 = DiverseBranchBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv10 = DiverseBranchBlock(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)




    def forward(self, vi, ir):
        if self.deploy:
            self.conv1.switch_to_deploy()
            self.conv1.deploy = True
            self.conv2.switch_to_deploy()
            self.conv2.deploy = True
            self.conv3.switch_to_deploy()
            self.conv3.deploy = True
            self.conv4.switch_to_deploy()
            self.conv4.deploy = True
            self.conv5.switch_to_deploy()
            self.conv5.deploy = True
            self.conv6.switch_to_deploy()
            self.conv6.deploy = True
            self.conv7.switch_to_deploy()
            self.conv7.deploy = True
            self.conv8.switch_to_deploy()
            self.conv8.deploy = True
            self.conv9.switch_to_deploy()
            self.conv9.deploy = True
            self.conv10.switch_to_deploy()
            self.conv10.deploy = True
            self.conv11.switch_to_deploy()
            self.conv11.deploy = True
            self.conv22.switch_to_deploy()
            self.conv22.deploy = True
            self.conv33.switch_to_deploy()
            self.conv33.deploy = True
            self.conv44.switch_to_deploy()
            self.conv44.deploy = True
            self.conv55.switch_to_deploy()
            self.conv55.deploy = True
        f = torch.cat([vi, ir], dim=1)
        f1 = self.conv1(f) * self.conv11(f)
        f2 = self.conv2(f1) * self.conv22(f1)
        f3 = self.conv3(f2) * self.conv33(f2)
        f4 = self.conv4(f3) * self.conv44(f3)
        f = self.conv5(f4) * self.conv5(f4)

        f=self.conv6(f)
        f=self.conv7(f)
        f=self.conv8(f)
        f = self.conv9(f)
        f = self.conv10(f)
        return f

class FusionNet_3(nn.Module):
    def __init__(self, deploy=False):
        super(FusionNet_3, self).__init__()
        self.deploy = deploy
        self.conv1 = DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv11 = DiverseBranchBlock(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv22 = DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv33 = DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = DiverseBranchBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = DiverseBranchBlock(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = DiverseBranchBlock(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)




    def forward(self, vi, ir):
        if self.deploy:
            self.conv1.switch_to_deploy()
            self.conv1.deploy = True
            self.conv2.switch_to_deploy()
            self.conv2.deploy = True
            self.conv3.switch_to_deploy()
            self.conv3.deploy = True
            self.conv4.switch_to_deploy()
            self.conv4.deploy = True
            self.conv5.switch_to_deploy()
            self.conv5.deploy = True
            self.conv6.switch_to_deploy()
            self.conv6.deploy = True
            self.conv11.switch_to_deploy()
            self.conv11.deploy = True
            self.conv22.switch_to_deploy()
            self.conv22.deploy = True
            self.conv33.switch_to_deploy()
            self.conv33.deploy = True
        f = torch.cat([vi, ir], dim=1)
        f1 = self.conv1(f) * self.conv11(f)
        f2 = self.conv2(f1) * self.conv22(f1)
        f3 = self.conv3(f2) * self.conv33(f2)

        f=self.conv4(f3)
        f=self.conv5(f)
        f=self.conv6(f)
        return f


if __name__ == '__main__':
    img = torch.randn(2, 1, 40, 40).cuda()
    model = FusionNet(deploy=True).cuda()
    import time
    start = time.time()
    f = model(img, img)
    print(f.shape)
    ##0.19 0.24 0.21
