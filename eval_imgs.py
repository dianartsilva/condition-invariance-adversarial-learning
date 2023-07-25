import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset', choices=['BDD100K', 'DAWN', 'SODA'])
parser.add_argument('condition', choices=['weather', 'scene', 'timeofday'])
parser.add_argument('--exclude-condition', type=int)
parser.add_argument('--fold', default='test')
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import objdetect as od
import torch
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################## DATA ########################

H, W = 128, 256
transform = A.Compose([
    A.Resize(H, W),
    A.Normalize((0, 0, 0), (1, 1, 1)),
    ToTensorV2()
], bbox_params=A.BboxParams('albumentations', ['labels']))

if args.dataset == 'BDD100K':
    dataset = data.BDD100K
if args.dataset == 'DAWN':
    dataset = data.DAWN
if args.dataset == 'SODA':
    dataset = data.SODA


ts = dataset('/data', args.fold, args.condition, transform)
if args.exclude_condition is not None:
    ts = data.RestrictCondition(ts, [args.exclude_condition])

# FIXME: train only 1 image
#ts = torch.utils.data.Subset(ts, range(1))  # TEMP DEBUG

ts = DataLoader(ts, 1, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)

######################## MODEL ########################

model = torch.load(args.model, map_location=device)
model.to(device)

######################## EVAL LOOP ########################

metric = MeanAveragePrecision()
model.eval()

for i, (images, targets) in enumerate(ts):
    images = images.to(device)
    with torch.no_grad():
        preds = model(images)
    metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], targets)
    res = metric.compute()
    print(args.model, ' '.join(f'{k} {v}' for k, v in sorted(res.items())))

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,5))
    plt.imshow(images[0].permute(1, 2, 0))
    od.utils.plot(targets[0]['boxes'], labels=[l.item() for l in targets[0]['labels']], color='cyan')
    print('preds:', preds[0]['boxes'])
    od.utils.plot(preds[0]['boxes'], labels=['%d (%d%%)' % (l, s*100) for l, s in zip(preds[0]['labels'], preds[0]['scores'])], color='red')
    plt.title('Targets (cyan), Preds (red)')
    plt.suptitle(f'Image {i}')
    plt.show()
    plt.close(fig)
    fig.savefig(f'images_DAWN/{name}.png',bbox_inches='tight', dpi=150)
