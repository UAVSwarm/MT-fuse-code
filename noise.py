import glob

import skimage
from PIL import Image
import numpy as np
import cv2



def main_loc(path):
    # 读取图片
    img = cv2.imread(path)
    # 设置高斯分布的均值和方差
    mean = 100
    # 设置高斯分布的标准差
    sigma = 10
    # 根据均值和标准差生成符合高斯分布的噪声
    #gauss = np.random.normal(mean, sigma, (128, 128, 3))
    gauss = np.random.normal(mean, sigma, (1024, 1280, 3))
    # 给图片添加高斯噪声
    noisy_img = img + gauss
    # 设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    # 保存图片
    save_path = path.replace('Vis', 'Vis_noised')
    print(save_path)
    cv2.imwrite(save_path, noisy_img)

import tqdm
if __name__ == '__main__':
    name_list = glob.glob('test_data/Vis/*.jpg')
    #name_list = glob.glob('/home/groupyun/桌面/sdd/MT-fuse-code/datasets/Vis/*.jpg')
    idx=0
    for name in name_list:
        main_loc(name)
        print(idx)
        idx=idx+1