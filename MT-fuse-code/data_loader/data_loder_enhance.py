import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
from models.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])


class llvip(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得TNO数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'Vis':
                self.vis_path = temp_path  # 获得可见光路径
            elif sub_dir == 'enhanced':
                self.enhanced = temp_path  # 获取增强路径

        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name))
        enhanced = Image.open(os.path.join(self.enhanced, name))

        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        enhanced = self.transform(enhanced)

        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        enhanced_y, _, _ = RGB2YCrCb(enhanced)
        return vis_image, vis_y_image, vis_cr_image, vis_cb_image, inf_image, enhanced_y, name
    def __len__(self):
        return len(self.name_list)


