import argparse
parser = argparse.ArgumentParser()
parser.add_argument('architecture', choices=['fasterrcnn_resnet50_fpn_v2', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fcos_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large'])
parser.add_argument('dataset', choices=['BDD100K', 'DAWN', 'SODA'])
parser.add_argument('condition', choices=['weather', 'scene', 'timeofday', 'period', 'location'])
parser.add_argument('output')
parser.add_argument('--condition_head', action='store_true')
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--condition_transfer', action='store_true')
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--exclude-condition', type=int) 
parser.add_argument('--lr', default=0.00005, type=float)
args = parser.parse_args()

assert args.condition_head or not args.adversarial, 'If --adversarial, then --condition_head is mandatory'

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from time import time
import data
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################## TRAIN DATA ########################

# Albumentation examples for object detection
# https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

H, W = 256, 512
transform = A.Compose([
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

tr = dataset('/data/auto', 'train', args.condition, transform) 

n_condition_classes = len(dataset.conditions[args.condition])

# Exclude a condition
if args.exclude_condition is not None:
    tr = data.RestrictCondition(tr, [i for i in range(n_condition_classes) if i != args.exclude_condition])
    n_condition_classes -= 1


# tr = torch.utils.data.Subset(tr, range(1))  # TEMP DEBUG 10%
tr = DataLoader(tr, 8, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)

######################## TEST DATA ########################

H, W = 128, 256
transform = A.Compose([
    A.Resize(H, W),
    A.Normalize((0, 0, 0), (1, 1, 1)),
    ToTensorV2()
], bbox_params=A.BboxParams('pascal_voc', ['labels']))

if args.dataset == 'BDD100K':
    dataset = data.BDD100K
if args.dataset == 'DAWN':
    dataset = data.DAWN
if args.dataset == 'SODA':
    dataset = data.SODA

objects = dataset.categories
K = dataset.K


######################## MODEL ########################

class ConditionHead(torch.nn.Module):
    '''Adjusting the model to the final objective.'''
    def __init__(self, backbone_out):
        super().__init__()
        self.head = torch.nn.Linear(backbone_out, n_condition_classes)

    def forward(self, x):
        x = x.mean([2, 3])  # global avgpool (dimensions 2 and 3 only)
        return self.head(x)

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
        self.head = ConditionHead(backbone_out)

    def forward(self, x, y):
        x_ = x
        encoding = []
        for layer in self.backbone:
            x_ = layer(x_)
            encoding.append(x_)
        return self.model(x, y), encoding, self.head(self.backbone(x))

model = Model(args.architecture)
model.to(device)
model.train()

model.to(device)
ce_loss = torch.nn.CrossEntropyLoss()

#################### OPTIMIZER ####################

optimizer_for_objdetect = torch.optim.Adam(model.model.parameters(), args.lr)
optimizer_for_adversarial = torch.optim.Adam(model.model.parameters(), args.lr)
optimizer_for_condition = torch.optim.Adam(model.head.parameters(), args.lr)
optimizer_for_transfer = torch.optim.Adam(model.model.parameters(), args.lr)

# https://www.mdpi.com/2072-4292/13/1/89 - Waymo Open Dataset ~ 70 000 training images
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) #momentum=0.9, weight_decay=0.0005
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 11],
#     gamma=0.9)


######################## TRAIN LOOP ########################

# Losses for csv save
if args.adversarial:
    total_results = {'epochs': [], 'loss':[], 'classification':[], 'bbox_regression':[], 'bbox_ctrness':[], 'condition':[], 'condition_adv':[], 'condition_transfer':[], 'map':[], 'map_small':[], 'map_medium':[], 'map_large':[], 'map_50':[], 'map_75':[]}
else:
    total_results = {'epochs': [], 'loss':[], 'classification':[], 'bbox_regression':[], 'bbox_ctrness':[], 'map':[], 'map_small':[], 'map_medium':[], 'map_large':[], 'map_50':[], 'map_75':[]}

for epoch in range(args.epochs):
    model.train()
    tic = time()
    avg_loss = 0
    avg_losses = {}
    avg_transfer_loss = 0

    for images, targets in tr:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses, _, _ = model(images, targets)
        
        # object detection loss
        optimizer_for_objdetect.zero_grad()
        sum_losses = sum(l for l in losses.values())
        sum_losses.backward(retain_graph=True)
        grads_objdetect = [(param, param.grad.clone()) for param in model.parameters() if param.grad is not None]

        for k, v in losses.items():
            avg_losses[k] = avg_losses.get(k, 0) + float(v)/len(tr)
        
        avg_loss += float(sum_losses) / len(tr)

        if args.condition_head:
            _, _, condition_preds = model(images, targets)  # fpass
            targets_head = torch.stack([t['condition'] for t in targets])
            #torch.autograd.set_detect_anomaly(True)
            optimizer_for_condition.zero_grad()
            condition_loss = ce_loss(condition_preds, targets_head)
            condition_loss.backward(retain_graph=True)
            grads_condition = [(param, param.grad.clone()) for param in model.parameters() if param.grad is not None]
            avg_losses['condition'] = avg_losses.get('condition', 0) + float(condition_loss)/len(tr)

        if args.adversarial:
            _, _, condition_preds = model(images, targets)
            targets_adv = torch.ones_like(condition_preds)/n_condition_classes
            optimizer_for_adversarial.zero_grad()
            logprobs = torch.nn.functional.log_softmax(condition_preds, 1)
            adv_loss = torch.nn.functional.kl_div(logprobs, targets_adv, reduction='batchmean')
            adv_loss.backward(retain_graph=True)
            grads_adversarial = [(param, param.grad.clone()) for param in model.parameters() if param.grad is not None]
            avg_losses['condition_adv'] = avg_losses.get('condition_adv', 0) + float(adv_loss)/len(tr)

        if args.condition_transfer:
            # Condition-Transfer Loss: to enforce the latent distributions of different conditions to be as similar as possible.
            _, encoding_outputs, _ = model(images, targets)
            conditions = torch.stack([t['condition'] for t in targets])
            unique_conditions = torch.unique(conditions)
            # print('all-conditions: ', conditions)

            optimizer_for_transfer.zero_grad()
            transfer_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for i, condition1 in enumerate(unique_conditions):
                equal1 = conditions == condition1
                
                # for each encoder layer get the average for this condition
                # h_conditions1 = avera\ge per layer activation map
                
                h_condition1 = [torch.flatten(torch.mean(torch.stack(([a for a in p[equal1]]), dim=0), dim=0)) for p in encoding_outputs]

                for condition2 in unique_conditions[i+1:]:
                    # print('condition2: ', condition2)
                    equal2 = conditions == condition2
                    h_condition2 = [torch.flatten(torch.mean(torch.stack(([a for a in p[equal2]]), dim=0), dim=0)) for p in encoding_outputs]

                    # we could multiply by betas to give more weight to some layers
                    # h_condition1 = (N, K, H, W)

                    loss = torch.mean(torch.sum((torch.cat(h_condition1)-torch.cat(h_condition2))**2)).clone().detach().requires_grad_(True) # Torch.mean vem do facto de conditional adversarial ter reduction=batchmean
                    transfer_loss = loss + transfer_loss

            transfer_loss.backward(retain_graph=True)
            grads_transfer = [(param, param.grad.clone()) for param in model.parameters() if param.grad is not None]
            avg_losses['condition_transfer'] = avg_losses.get('condition_transfer', 0) + float(transfer_loss)/len(tr)
    
    # Optimizer STEP
    for param, grad in grads_objdetect:
         param.grad = grad
    optimizer_for_objdetect.step()
    if args.adversarial:
        for param, grad in grads_condition:
            param.grad = grad
        optimizer_for_condition.step()
        for param, grad in grads_adversarial:
            param.grad = grad
        optimizer_for_adversarial.step()
        for param, grad in grads_transfer:
            param.grad = grad
        optimizer_for_transfer.step()

    toc = time()

    # SAVE
    print(f'Epoch {epoch+1} - {toc-tic:.1f}s - Loss: {avg_loss} - ' + ' - '.join(f'{k}: {v}' for k, v in avg_losses.items()))

    # for each epoch, evaluate performance   
    model.eval()
    metric = MeanAveragePrecision()
    tic = time()
    if args.exclude_condition is None:
        for cd in range(n_condition_classes):
            ts = dataset('/data/auto', 'val', args.condition, transform)
            ts = data.RestrictCondition(ts, [cd])
            # ts = torch.utils.data.Subset(ts, range(1))  # TEMP DEBUG
            ts = DataLoader(ts, 8, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)

            for i, (images, targets) in enumerate(ts):
                images = images.to(device)
                with torch.no_grad():
                    preds = model.model(images)
                metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], targets)
            m = metric.compute()
            print(f'test mAP condition {cd} - {toc-tic:.1f}s -', m)

            total_results['epochs'].append(epoch)
            total_results['loss'].append(avg_loss)
            total_results['classification'].append(avg_losses['classification'])
            total_results['bbox_regression'].append(avg_losses['bbox_regression'])
            total_results['bbox_ctrness'].append(avg_losses['bbox_ctrness'])
            if args.adversarial:
                total_results['condition'].append(avg_losses['condition'])
                total_results['condition_adv'].append(avg_losses['condition_adv'])
                total_results['condition_transfer'].append(avg_losses['condition_transfer'])
            total_results['map'].append(m['map'].numpy().item())
            total_results['map_small'].append(m['map_small'].numpy().item())
            total_results['map_medium'].append(m['map_medium'].numpy().item())
            total_results['map_large'].append(m['map_large'].numpy().item())
            total_results['map_50'].append(m['map_50'].numpy().item())
            total_results['map_75'].append(m['map_75'].numpy().item())
    else:
        ts = dataset('/data/auto', 'val', args.condition, transform)
        ts = data.RestrictCondition(ts, [args.exclude_condition])
        # ts = torch.utils.data.Subset(ts, range(1))  # TEMP DEBUG
        ts = DataLoader(ts, 8, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)
        
        for i, (images, targets) in enumerate(ts):
            images = images.to(device)
            with torch.no_grad():
                preds = model.model(images)
            metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], targets)
        m = metric.compute()
        print(f'test mAP condition - {toc-tic:.1f}s -', m)

        total_results['epochs'].append(epoch)
        total_results['loss'].append(avg_loss)
        total_results['classification'].append(avg_losses['classification'])
        total_results['bbox_regression'].append(avg_losses['bbox_regression'])
        total_results['bbox_ctrness'].append(avg_losses['bbox_ctrness'])
        if args.adversarial:
            total_results['condition'].append(avg_losses['condition'])
            total_results['condition_adv'].append(avg_losses['condition_adv'])
            total_results['condition_transfer'].append(avg_losses['condition_transfer'])
        total_results['map'].append(m['map'].numpy().item())
        total_results['map_small'].append(m['map_small'].numpy().item())
        total_results['map_medium'].append(m['map_medium'].numpy().item())
        total_results['map_large'].append(m['map_large'].numpy().item())
        total_results['map_50'].append(m['map_50'].numpy().item())
        total_results['map_75'].append(m['map_75'].numpy().item())
    
    toc = time()

# CSV file
info = pd.DataFrame(total_results)
info.to_csv(f'{args.output}.csv', index=False)

torch.save(model.model.cpu(), args.output)
