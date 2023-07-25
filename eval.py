import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset', choices=['BDD100K', 'DAWN', 'SODA'])
parser.add_argument('condition', choices=['weather', 'scene', 'timeofday'])
parser.add_argument('--exclude-condition', type=int)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import data
import objdetect as od
import aux_func

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################## DATA ########################

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
ts = dataset('/data/auto', 'val', args.condition, transform)

if args.exclude_condition is not None:
    ts = data.RestrictCondition(ts, [args.exclude_condition])

# ts = torch.utils.data.Subset(ts, range(1))  # TEMP DEBUG
ts = DataLoader(ts, 1, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)

######################## MODEL ########################

model = torch.load(args.model, map_location=device)
model.to(device)

######################## EVAL LOOP ########################

import os
directory = f"{args.dataset}_eval"
parent_dir = os.getcwd() 
path = os.path.join(parent_dir, directory)
if not os.path.exists(path):
    os.makedirs(path)


metric = MeanAveragePrecision()
model.eval()

# Color Map 
colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'black']
for b in range(K):
    print(objects[b], '-', colors[b])

for i, (images, targets) in enumerate(ts):
    images = images.to(device)
    with torch.no_grad():
        preds = model(images)
    metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], targets)
    # res_i = metric.compute()
    # print(f'image {i} \n', '-'.join(f'{k} - {v}\n' for k, v in sorted(res_i.items())))
    print(i, 'out of', len(ts))
    # SAVING IMAGES
    if i<100:
        aux_func.bboxes_plot_GT(images, targets, colors, objects, args.dataset, i)
        aux_func.bboxes_plot_PRED(images, preds, colors, objects, args.dataset, i)
        aux_func.bboxes_plot_GT_PRED(images, targets, preds, colors, objects, args.dataset, i)

res = metric.compute()
print('Final results \n', '-'.join(f'{k} - {v}\n' for k, v in sorted(res.items())))

