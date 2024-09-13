import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import numpy as np
from models.LAB_Net import WaterNet
from datasets.dataset import TestDataset
from torch.utils.data import DataLoader
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', default='./ckpt/weights_without_spa/epoch_200.pth', help="path to the saved checkpoint of model")
parser.add_argument('--test_path', default='/data/liuwj/Datasets/NUID/NF', type=str, help='path to the test set')
parser.add_argument('--bs_test', default=1, type=int, help='[test] batch size (default: 1)')
parser.add_argument('--out_path', default='./results/NF', type=str, help='path to the result')

args = parser.parse_args()
print(args)


model = WaterNet()
model.load_state_dict(torch.load(args.ckpt_path))
model.eval()
model.cuda()


with torch.no_grad():
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    test_set = TestDataset(args.test_path)
    test_loader = DataLoader(test_set, batch_size=args.bs_test, shuffle=False, num_workers=8, pin_memory=True)
    for i, (lab, rgb, hsv, gray, params_origin, name) in enumerate(test_loader):

        lab, rgb, hsv, gray = lab.cuda(), rgb.cuda(), hsv.cuda(), gray.cuda()
        params_origin = params_origin.cuda()
        out = model(lab, rgb, hsv, gray, params_origin)['lab_rgb']
        out = out.to(device="cpu").numpy().squeeze()
        out = np.clip(out * 255.0, 0, 255)
        save_img = Image.fromarray(np.uint8(out).transpose(1, 2, 0))
        save_img.save(os.path.join(args.out_path, str(name[0])))
        print('%d|%d' % (i + 1, len(test_set)))

