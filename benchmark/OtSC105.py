import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
from imageio import imwrite
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'

'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()

path = args.path
f = open(path + '/tri_testlist.txt', 'r')
Dpsnr_list = []
j=0
print(f'=========================Starting testing=========================')
print(f'Dataset: Pictures   Model: {model.name}   TTA: {TTA}')
print(f'InputPath : {path} ')

for i in f:
    int(j)
    j = j+1
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    I0 = cv2.imread(path + name + '\im1.jpg')
    I2 = cv2.imread(path + name + '\im2.jpg')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    padder = InputPadder(I0.shape, divisor=32)
    I0, I2 = padder.pad(I0, I2)
    mid1 = (padder.unpad(model.inference(I0, I2, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    mid = model.inference(I0, I2, TTA=TTA, fast_TTA=TTA)[0]
    imwrite(path + name +'\mid.jpg', mid1)
    print(str(j) + '、' + 'Output : '+ path + name + '\mid' +'.jpg')
    psnr0 = -10 * math.log10(((I0 - I2) * (I0 - I2)).mean())
    psnr1 = -10 * math.log10(((I0 - mid) * (I0 - mid)).mean())
    psnr2 = -10 * math.log10(((mid - I2) * (mid - I2)).mean())
    dpsnr = psnr0 - (psnr1 + psnr2)/2
    Dpsnr_list.append(dpsnr)
    print("Dpsnr = {} psnr0 = {} psnr1 = {} psnr2 ={}".format(dpsnr,psnr0,psnr1,psnr2))

print(f'Apsnr = {(np.sum(Dpsnr_list))/j}')
filename = os.path.join(path + 'Result.txt')
with open(filename, 'w') as f:
    for item in Dpsnr_list:
        f.write("%s\n" % item)
    f.write(f'the result apsnr = {(np.sum(Dpsnr_list))/j}')
print(f'=========================Done=========================')