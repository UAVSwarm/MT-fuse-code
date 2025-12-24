"""测试融合网络"""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.data_loder_test import llvip
from models.common import YCrCb2RGB, RGB2YCrCb, clamp

from models.model import FusionNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



if __name__ == '__main__':
    datasets_path = 'Test/LLVIP'
    num_works = 1
    save_path = datasets_path + '/ours'
    print(save_path)
    pretrain = './runs/VIF.pth'
    print(pretrain)

    test_dataset = llvip(datasets_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_works, pin_memory=True)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #######加载模型
    model =(FusionNet(deploy=True)).cuda()
    ###############加载
    model.load_state_dict(torch.load(pretrain))
    model.eval()
    ##########加载数据
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image.cuda()
            cb = vis_cb_image.cuda()
            cr = vis_cr_image.cuda()
            inf_image = inf_image.cuda()


            #########编码
            fused = model(vis_y_image, inf_image)
            ###########转为rgb
            fused = clamp(fused)

            rgb_fused_image = YCrCb2RGB(fused[0], cr[0], cb[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{save_path}/{name[0]}')



