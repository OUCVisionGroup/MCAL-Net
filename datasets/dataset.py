import torch
import random
import numpy as np
import os
import cv2
import kornia
from glob import glob
from torch.utils.data import Dataset
from utils.util import trans_np


def kornia2lab(tensor, rgb2lab=True):
    if rgb2lab:
        # tensor should be im_tensor [n 3 h w] 0~1
        # l 0~100     a,b  -127~127
        im_lab = kornia.color.rgb_to_lab(tensor)
        return im_lab
    else:
        # tensor should be lab_tensor [n 3 h w] l 0~100  a,b  -127~127
        # im_tensor  [n 3 h w]  0~1
        im_tensor = kornia.color.lab_to_rgb(tensor)
        return im_tensor


# l通道归一化自注意力
class TrainDataset(Dataset):
    def __init__(self, dataroot='/data/liuwj/Datasets/UIEB/train', output_size=(256, 256)):
        # 1.get raw image path, ref txt path and ref iamges path
        self.raw_path = os.path.join(dataroot, 'raw_pic')
        self.ref_path = os.path.join(dataroot, 'ref_pic')
        self.txt_path = os.path.join(dataroot, 'ref_txt')
        self.output_size = output_size

        # 2.set box for raw images, ref images and its name ,ect on
        self.raw_images = []
        self.ref_images = []
        self.tr_num = {}
        self.pic_name = []

        pics = os.listdir(self.raw_path)

        # 3.add pic
        for pic in pics:
            self.raw_images.append(os.path.join(self.raw_path, pic))
            self.ref_images.append(os.path.join(self.ref_path, pic))
            self.pic_name.append(pic)
            name, type = os.path.splitext(pic)
            txt = name + '.txt'
            true_num_path = os.path.join(self.txt_path, txt)

            with open(true_num_path) as f:
                for line in f:
                    words = line.split(',')
                    # to num list
                    word = map(float, words)
                    num = list(word)
                    self.tr_num[pic] = [num[0], num[3]]

    def __getitem__(self, index):
        # get L channel map(0~1)
        raw_img = cv2.imread(self.raw_images[index])
        raw_img = cv2.resize(raw_img, self.output_size)
        rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        ref_img = cv2.imread(self.ref_images[index])
        ref_img = cv2.resize(ref_img, self.output_size)
        rgb_label = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1

        gray = 1 - ((l - torch.min(l)) / (torch.max(l) - torch.min(l)))

        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        lab_label = kornia.color.rgb_to_lab(rgb_label)
        ab_label = lab_label[1:, :, :] / 127.  # -1~1

        #hsv space
        hsv = kornia.color.rgb_to_hsv(rgb) # h:0~2*np.pi  s,v: 0~1
        h = hsv[:1, :, :] / (2*np.pi) # 0~1
        sv = hsv[1:, :, :] # 0~1
        hsv = torch.cat([h, sv], dim=0)

        # get ref images L channel mean and std
        true_num = self.tr_num[self.pic_name[index]]

        return {'l': l, 'lab': lab, 'rgb': rgb, 'hsv': hsv, 'gray': gray, 'label_ab': ab_label, 'label_lab': lab_label, 'label_rgb': rgb_label, 'params_origin': params_origin, 'true_num': true_num}

    def __len__(self):
        return len(self.raw_images)


# l通道归一化自注意力
class TestDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + 'jpg')) + glob(os.path.join(self.data_path, '*.' + 'JPEG')) + glob(os.path.join(self.data_path, '*.' + 'png')) + glob(os.path.join(self.data_path, '*.' + 'bmp'))
        self.transform = self.test_transform

    def test_transform(self, img):
        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1
        gray = 1 - ((l - torch.min(l)) / (torch.max(l) - torch.min(l)))

        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        # hsv space
        hsv = kornia.color.rgb_to_hsv(rgb)  # h:0~2*np.pi  s,v: 0~1
        h = hsv[:1, :, :] / (2 * np.pi)  # 0~1
        sv = hsv[1:, :, :]  # 0~1
        hsv = torch.cat([h, sv], dim=0)

        return lab, rgb, hsv, gray, params_origin

    def __getitem__(self, index):
        lab, rgb, hsv, gray, params_origin = self.transform(self.uw_images[index])
        name = os.path.basename(self.uw_images[index])
        return lab, rgb, hsv, gray, params_origin, name

    def __len__(self):
        return len(self.uw_images)


# abalation：without multi color space（消融：去掉多颜色空间，只有LAB）
class TrainDataset1(Dataset):
    def __init__(self, dataroot = '/data/liuwj/Datasets/UIEB/train', output_size=(256, 256)):
        # 1.get raw image path, ref txt path and ref iamges path
        self.raw_path = os.path.join(dataroot, 'raw_pic')
        self.ref_path = os.path.join(dataroot, 'ref_pic')
        self.txt_path = os.path.join(dataroot, 'ref_txt')
        self.output_size = output_size

        # 2.set box for raw images, ref images and its name ,ect on
        self.raw_images = []
        self.ref_images = []
        self.tr_num = {}
        self.pic_name = []

        pics = os.listdir(self.raw_path)

        # 3.add pic
        for pic in pics:
            self.raw_images.append(os.path.join(self.raw_path, pic))
            self.ref_images.append(os.path.join(self.ref_path, pic))
            self.pic_name.append(pic)
            name, type = os.path.splitext(pic)
            txt = name + '.txt'
            true_num_path = os.path.join(self.txt_path, txt)

            with open(true_num_path) as f:
                for line in f:
                    words = line.split(',')
                    # to num list
                    word = map(float, words)
                    num = list(word)
                    self.tr_num[pic] = [num[0], num[3]]

    def __getitem__(self, index):
        # get L channel map(0~1)
        raw_img = cv2.imread(self.raw_images[index])
        raw_img = cv2.resize(raw_img, self.output_size)
        rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        ref_img = cv2.imread(self.ref_images[index])
        ref_img = cv2.resize(ref_img, self.output_size)
        rgb_label = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1

        gray = 1 - ((l - torch.min(l)) / (torch.max(l) - torch.min(l)))

        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        lab_label = kornia.color.rgb_to_lab(rgb_label)
        ab_label = lab_label[1:, :, :] / 127.  # -1~1

        # get ref images L channel mean and std
        true_num = self.tr_num[self.pic_name[index]]

        return {'l': l, 'lab': lab, 'gray': gray, 'label_ab': ab_label, 'label_lab': lab_label, 'label_rgb': rgb_label, 'params_origin': params_origin, 'true_num': true_num}

    def __len__(self):
        return len(self.raw_images)


# ablation: multi color
class TestDataset1(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + 'jpg')) + glob(os.path.join(self.data_path, '*.' + 'JPEG')) + glob(os.path.join(self.data_path, '*.' + 'png')) + glob(os.path.join(self.data_path, '*.' + 'bmp'))
        self.transform = self.test_transform

    def test_transform(self, img):
        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1
        gray = 1 - ((l - torch.min(l)) / (torch.max(l) - torch.min(l)))

        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        return lab, gray, params_origin

    def __getitem__(self, index):
        lab, gray, params_origin = self.transform(self.uw_images[index])
        name = os.path.basename(self.uw_images[index])
        return lab, gray, params_origin, name

    def __len__(self):
        return len(self.uw_images)


# ablation: L Self-attention（消融：去掉L通道自注意力）
class TrainDataset2(Dataset):
    def __init__(self, dataroot = '/data/liuwj/Datasets/UIEB/train', output_size=(256, 256)):
        # 1.get raw image path, ref txt path and ref iamges path
        self.raw_path = os.path.join(dataroot, 'raw_pic')
        self.ref_path = os.path.join(dataroot, 'ref_pic')
        self.txt_path = os.path.join(dataroot, 'ref_txt')
        self.output_size = output_size

        # 2.set box for raw images, ref images and its name ,ect on
        self.raw_images = []
        self.ref_images = []
        self.tr_num = {}
        self.pic_name = []

        pics = os.listdir(self.raw_path)

        # 3.add pic
        for pic in pics:
            self.raw_images.append(os.path.join(self.raw_path, pic))
            self.ref_images.append(os.path.join(self.ref_path, pic))
            self.pic_name.append(pic)
            name, type = os.path.splitext(pic)
            txt = name + '.txt'
            true_num_path = os.path.join(self.txt_path, txt)

            with open(true_num_path) as f:
                for line in f:
                    words = line.split(',')
                    # to num list
                    word = map(float, words)
                    num = list(word)
                    self.tr_num[pic] = [num[0], num[3]]

    def __getitem__(self, index):
        # get L channel map(0~1)
        raw_img = cv2.imread(self.raw_images[index])
        raw_img = cv2.resize(raw_img, self.output_size)
        rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

        ref_img = cv2.imread(self.ref_images[index])
        ref_img = cv2.resize(ref_img, self.output_size)
        rgb_label = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w
        rgb_label = torch.tensor(rgb_label.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1
        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        lab_label = kornia.color.rgb_to_lab(rgb_label)
        ab_label = lab_label[1:, :, :] / 127.  # -1~1

        #hsv space
        hsv = kornia.color.rgb_to_hsv(rgb) # h:0~2*np.pi  s,v: 0~1
        h = hsv[:1, :, :] / (2*np.pi) # 0~1
        sv = hsv[1:, :, :] # 0~1
        hsv = torch.cat([h, sv], dim=0)

        # get ref images L channel mean and std
        true_num = self.tr_num[self.pic_name[index]]

        return {'l': l, 'lab': lab, 'rgb': rgb, 'hsv': hsv, 'label_ab': ab_label, 'label_lab': lab_label, 'label_rgb': rgb_label, 'params_origin': params_origin, 'true_num': true_num}

    def __len__(self):
        return len(self.raw_images)


# ablation: L self attention（消融：去掉L通道自注意力）
class TestDataset2(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + 'jpg')) + glob(os.path.join(self.data_path, '*.' + 'JPEG')) + glob(os.path.join(self.data_path, '*.' + 'png')) + glob(os.path.join(self.data_path, '*.' + 'bmp'))
        self.transform = self.test_transform

    def test_transform(self, img):
        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1
        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        # hsv space
        hsv = kornia.color.rgb_to_hsv(rgb)  # h:0~2*np.pi  s,v: 0~1
        h = hsv[:1, :, :] / (2 * np.pi)  # 0~1
        sv = hsv[1:, :, :]  # 0~1
        hsv = torch.cat([h, sv], dim=0)

        return lab, rgb, hsv, params_origin

    def __getitem__(self, index):
        lab, rgb, hsv, params_origin = self.transform(self.uw_images[index])
        name = os.path.basename(self.uw_images[index])
        return lab, rgb, hsv, params_origin, name

    def __len__(self):
        return len(self.uw_images)


# big size（大尺寸图像测试）
class TestDataset3(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + 'jpg')) + glob(os.path.join(self.data_path, '*.' + 'JPEG')) + glob(os.path.join(self.data_path, '*.' + 'png')) + glob(os.path.join(self.data_path, '*.' + 'bmp'))
        self.transform = self.test_transform

    def test_transform(self, img):
        img = cv2.imread(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = torch.tensor(rgb.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        img1 = cv2.resize(img, (256, 256))
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb1 = torch.tensor(rgb1.astype(float).transpose(2, 0, 1), dtype=torch.float) / 255.  # 3 h w

        #lab space
        lab = kornia.color.rgb_to_lab(rgb)  # l: 0~100; a,b: -127~127
        params_origin = trans_np(np.array(lab[0, :, :]))

        l = lab[:1, :, :] / 100.  # 0~1
        gray = 1 - ((l - torch.min(l)) / (torch.max(l) - torch.min(l)))

        ab = lab[1:, :, :] / 127.  # -1~1
        lab = torch.cat([l, ab], dim=0)

        lab1 = kornia.color.rgb_to_lab(rgb1)  # l: 0~100; a,b: -127~127
        l1 = lab1[:1, :, :] / 100.  # 0~1
        gray1 = 1 - ((l1 - torch.min(l1)) / (torch.max(l1) - torch.min(l1)))

        ab1 = lab1[1:, :, :] / 127.  # -1~1
        lab1 = torch.cat([l1, ab1], dim=0)

        # hsv space
        hsv = kornia.color.rgb_to_hsv(rgb)  # h:0~2*np.pi  s,v: 0~1
        h = hsv[:1, :, :] / (2 * np.pi)  # 0~1
        sv = hsv[1:, :, :]  # 0~1
        hsv = torch.cat([h, sv], dim=0)

        return lab, lab1, rgb, hsv, gray, gray1, params_origin

    def __getitem__(self, index):
        lab, lab1, rgb, hsv, gray, gray1, params_origin = self.transform(self.uw_images[index])
        name = os.path.basename(self.uw_images[index])
        return lab, lab1, rgb, hsv, gray, gray1, params_origin, name

    def __len__(self):
        return len(self.uw_images)

