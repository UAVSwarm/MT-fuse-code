import torch
import ssl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pytorch_msssim import ssim
from models.common import gradient
from loss import final_l1_loss

# Data preparation
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    #.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对所有通道进行归一化，使其分布在[-1, 1]范围内
])

# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#
# #task1_data = [data for data in train_dataset if data[1] < 5]
# #task2_data = [data for data in train_dataset if data[1] >= 5]
# # Split data into two groups
# train_dataset_size = len(train_dataset)
# train_split_sizes = [train_dataset_size // 2, train_dataset_size - train_dataset_size // 2]
# task1_data, task2_data = random_split(train_dataset, train_split_sizes)
#
#
#
# task1_loader = DataLoader(task1_data, batch_size=64, shuffle=True)
# task2_loader = DataLoader(task2_data, batch_size=64, shuffle=True)
#
# test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
#
# #task1_test_data = [data for data in test_dataset if data[1] < 5]
# #task2_test_data = [data for data in test_dataset if data[1] >= 5]
# test_dataset_size = len(test_dataset)
# test_split_sizes = [test_dataset_size // 2, test_dataset_size - test_dataset_size // 2]
# task1_test_data, task2_test_data = random_split(test_dataset, test_split_sizes)
#
# task1_test_loader = DataLoader(task1_test_data, batch_size=64, shuffle=False)
# task2_test_loader = DataLoader(task2_test_data, batch_size=64, shuffle=False)


# Model definition
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# EWC implementation
class EWC_keep_en:
    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.importance = importance
        self.device = device
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)
#计算fisher信息矩阵
    def _compute_fisher(self, dataloader):
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        self.model.train()
        train_tqdm = tqdm(dataloader, total=len(dataloader), ascii=True, desc='ewc')
        for vis_image, vis_y_image, vis_cr_image, vis_cb_image, inf_image, enhanced_y, name in train_tqdm:
            vis_y_image = vis_y_image.cuda()
            inf_image = inf_image.cuda()
            enhanced_y = enhanced_y.cuda()
            self.model.zero_grad()
            fused = self.model(vis_y_image, inf_image)
            loss_aux = final_l1_loss(enhanced_y, inf_image, fused)
            loss_gard = F.l1_loss(gradient(fused), torch.max(gradient(enhanced_y), gradient(inf_image)))
            loss = 50 * loss_gard + 10 * loss_aux
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    try:
                        fisher[n] += (p.grad ** 2) / len(dataloader)
                    except:
                        pass


        return fisher


    def penalty(self, new_model):
        loss = 0
        for n, p in new_model.named_parameters():
            if p.requires_grad:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss * (self.importance / 2)

