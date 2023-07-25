import argparse
parser = argparse.ArgumentParser()
parser.add_argument('architecture', choices=['fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fcos_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large'])
parser.add_argument('dataset', choices=['BDD100K', 'DAWN', 'SODA'])
parser.add_argument('condition', choices=['weather', 'scene', 'timeofday', 'period', 'location'])
parser.add_argument('output')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch', default=1, type=int)
parser.add_argument('--lr', default=0.00005, type=float)
parser.add_argument('--condition_tr', type=int)
parser.add_argument('--condition_val', type=int)
parser.add_argument('--condition_ts', type=int)
args = parser.parse_args()

assert args.condition_tr != args.condition_ts and args.condition_val != args.condition_ts and args.condition_tr != args.condition_val, 'If --condition_ts, --condition_val and --condition_tr must be different'

# srun python3 train-bilevel.py fcos_resnet50_fpn DAWN weather try-bilevel.pth --condition_tr 0 --condition_val 1 --condition_ts 2

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Sampler
from torchvision import models
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import data

from time import time
import numpy as np
import pandas as pd
import sys

import random
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

############# CALCULATE SIGN ####################
def calculate_w(train_grads, valid_grads, mlamb=1, mu=0.01):
    
    flat_train_grads = []
    
    for batch in range(len(train_grads)):
        grad_list = []
        for param_grad in train_grads[batch]:
            grad_list.append(param_grad.view(-1))

        flat_train_grads.append(torch.cat(grad_list, dim=0))
    
    flat_valid_grads = []
    
    for batch in range(len(valid_grads)):
        grad_list = []
        for param_grad in valid_grads[batch]:
            grad_list.append(param_grad.view(-1))

        flat_valid_grads.append(torch.cat(grad_list, dim=0))
    
    w = []

    for batch in range(len(flat_train_grads)):
        train_grad = flat_train_grads[batch]
        w_acc = 0
        for valid_grad in flat_valid_grads:
            w_acc += torch.dot(valid_grad, train_grad)/((torch.dot(train_grad, train_grad) / mlamb) + mu)

        w.append(w_acc)
        
    w = torch.tensor(w, device=device)
    
    w = w / torch.linalg.norm(w, ord=1)

    return w

########## TEST MODEL ###############
def test_model(model, metric, loader, device):
    with torch.no_grad():
        model.to(device)
        model.eval()

        for i, (images, targets) in enumerate(loader):
                images = images.to(device)
                with torch.no_grad():
                    preds = model.model(images)
                
                metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], targets)
                # loss_value = loss(preds, targets)
        
        m = metric.compute()
    
    return  m



######################## TRAIN DATA ########################

# Albumentation examples for object detection
# https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

H, W = 256, 512
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(0.1),
    A.Resize(int(H*1.1), int(W*1.1)),
    A.RandomCrop(H, W),
    A.Normalize((0, 0, 0), (1, 1, 1)),
    ToTensorV2()
], bbox_params=A.BboxParams('pascal_voc', ['labels']))

if args.dataset == 'BDD100K':
    dataset = data.BDD100K
if args.dataset == 'DAWN':
    dataset = data.DAWN
if args.dataset == 'SODA':
    dataset = data.SODA

tr = dataset('/data/auto', 'train', args.condition, train_transform) 

H, W = 128, 256
val_transform = A.Compose([
    A.Resize(H, W),
    A.Normalize((0, 0, 0), (1, 1, 1)),
    ToTensorV2()
], bbox_params=A.BboxParams('pascal_voc', ['labels']))

ts = dataset('/data/auto', 'test', args.condition, val_transform)

val = dataset('/data/auto', 'val', args.condition, val_transform)

if args.condition is not None:
    tr = data.RestrictCondition(tr, [args.condition_tr])
    val = data.RestrictCondition(val, [args.condition_val])
    ts = data.RestrictCondition(ts, [args.condition_ts])

tr = DataLoader(tr, args.batch, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)
val = DataLoader(val, args.batch, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)
ts = DataLoader(ts, args.batch, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)

######################## MODEL ########################

class Model(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture == 'fcos_resnet50_fpn':
            self.model = models.detection.fcos_resnet50_fpn(weights='DEFAULT')
            self.model.head = models.detection.fcos.FCOSHead(256, 1, dataset.K+1)
            self.backbone = torch.nn.Sequential(*list(self.model.backbone.body.children()))
            backbone_out = 2048
        if architecture == 'ssd300_vgg16':
            # this only works if we increase images resolution to > 256
            self.model = models.detection.ssd300_vgg16(weights='DEFAULT')
            self.model.head = models.detection.ssd.SSDHead([512, 1024, 512, 256, 256, 256], [4, 6, 6, 6, 4, 4], dataset.K+1)
            self.backbone = self.model.backbone.features
            backbone_out = 512
    def forward(self, x, y):
        return self.model(x, y)

model = Model(args.architecture)
model.to(device)
model.train()

model.to(device)
loss = torch.nn.CrossEntropyLoss()
metric = MeanAveragePrecision()

#################### OPTIMIZER ####################

optimizer = torch.optim.Adam(model.parameters(), args.lr)

######################## TRAIN LOOP ########################

best_loss = -np.inf
plateau = 0

for epoch in range(args.epochs):
    print(f'Epoch: {epoch}')
    tic = time()
    model.train()
    
    avg_loss = 0
    avg_losses = {}
    

    for image_tr, target_tr in tr:
        train_grads = []
        valid_grads = []
        image_val, target_val = next(iter(val))

        ##### TRAIN SET #####
        image_tr = image_tr.to(device)
        target_tr = [{k: v.to(device) for k, v in t.items()} for t in target_tr]

        losses_tr = model(image_tr, target_tr)
        
        optimizer.zero_grad()
        sum_losses = sum(l for l in losses_tr.values())
        sum_losses.backward(retain_graph=True)

        for k, v in losses_tr.items():
            avg_losses[k] = avg_losses.get(k, 0) + float(v)/len(tr)
        
        avg_loss += float(sum_losses) / len(tr)

        tr_grad_list = []

        for param in model.parameters():
            if param.requires_grad:
                param_grad = param.grad.detach().clone()
                tr_grad_list.append(param_grad)
                
        train_grads.append(tr_grad_list)
             
        #### VAL SET #####

        image_val = image_val.to(device)
        target_val = [{k: v.to(device) for k, v in t.items()} for t in target_val]
  
        losses_val = model(image_val, target_val)

        optimizer.zero_grad()
        sum_losses = sum(l for l in losses_val.values())
        sum_losses.backward(retain_graph=True)

        val_grad_list = []

        for param in model.parameters():
            if param.requires_grad:
                param_grad = param.grad.detach().clone()
                val_grad_list.append(param_grad)
                
        valid_grads.append(val_grad_list)
        
        w = calculate_w(train_grads, valid_grads)
        # w = torch.clamp(w, min=0)
             
        optimizer.zero_grad()
        for batch in range(len(train_grads)):
            idx = 0
            
            for param in model.parameters():
                if param.requires_grad:
                    param_grad = train_grads[batch][idx]
                    param.grad += w[batch]*param_grad
                    
                    idx += 1
                    
        optimizer.step()


    print(f'Train - Loss: {avg_loss}' + ' - '.join(f'{k}: {v}' for k, v in avg_losses.items()))
    print()
    
    
   
    if (epoch+1) % 5 == 0:
        metric_train = test_model(model, metric, tr, device)
        print(f'mAP TRAIN DATALOADER -', metric_train)
        
        metric_val = test_model(model, metric, val, device)
        print(f'mAP VAL DATALOADER -', metric_val)
        
        metric_ts = test_model(model, metric, ts, device)
        print(f'mAP TEST DATALOADER -', metric_ts)

    toc = time()
    print(f' Train Time - {toc-tic:.1f}s')

torch.save(model.model.cpu(), args.output)


