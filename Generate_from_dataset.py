import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave
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
parser.add_argument('--model', default='ours_t', type=str)
parser.add_argument('--InputPath', type=str, required=True)
parser.add_argument('--n', default=8, type=int)
parser.add_argument('--OutputPath', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours_t', 'ours_small_t'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small_t':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()

path = args.InputPath
OutputPath = args.OutputPath
f = open(path + '/tri_testlist.txt', 'r')
j=0
print(f'=========================Starting Generation=========================')
print(f'Dataset: Pictures   Model: {model.name}   TTA: {TTA}')
print(f'InputPath : {path}  Output Path : {OutputPath}')

for i in f:
    int(j)
    j = j+1
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    I0 = cv2.imread(path + '\\' + name + '\im1.jpg')
    I2 = cv2.imread(path + '\\' + name + '\im2.jpg')
    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)
    images = [I0[:, :, ::-1]]
    preds = model.multi_inference(I0_, I2_, TTA=TTA, time_list=[(i + 1) * (1. / args.n) for i in range(args.n - 1)], fast_TTA=TTA)
    for pred in preds:
        images.append((padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
    images.append(I2[:, :, ::-1])
    mimsave(OutputPath + '\img' +str(j) +'.gif', images, fps=args.n)
    print(str(j) + '、' + 'Output : ' + OutputPath + '\img' +str(j) +'.gif')

print(f'=========================Done=========================')
