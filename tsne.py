import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('dataset', choices=['BDD100K', 'DAWN', 'SODA'])
parser.add_argument('condition', choices=['weather', 'scene', 'timeofday'])
parser.add_argument('--exclude-condition', type=int)
parser.add_argument('--fold', default='test')
args = parser.parse_args()

# srun python3 tse.py model-fcos_resnet50_fpn-DAWN-baseline-fp-weather-exclude-0.pth DAWN weather --exclude-condition 0



import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import objdetect as od
import torch
import data

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


ts = dataset('/data/auto', 'val', args.condition, transform)
n_condition_classes = len(dataset.conditions[args.condition])

conditions = ['fog', 'rain', 'sand', 'snow']
# conditions= conditions[args.exclude_condition]

# Exclude a condition
if args.exclude_condition is not None:
    ts = data.RestrictCondition(ts, [args.exclude_condition])
    n_condition_classes -= 1

# FIXME: train only 1 image
#ts = torch.utils.data.Subset(ts, range(1))  # TEMP DEBUG

ts = DataLoader(ts, 1, True, collate_fn=data.custom_collate, num_workers=4, pin_memory=True)

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

    def forward(self, x):
        x_ = x
        encoding = []
        for layer in self.backbone:
            x_ = layer(x_)
            encoding.append(x_)
        return encoding, self.head(self.backbone(x))

model = Model('fcos_resnet50_fpn')
model.model = torch.load(args.model, map_location=device)
model.to(device)

######################## EVAL LOOP ########################

metric = MeanAveragePrecision()
model.eval()

test_predictions = []
test_targets = []
test_embeddings = []
num = 0

for i, (images, targets) in enumerate(ts):
    images = images.to(device)
    with torch.no_grad():
        encoding_outputs, _ = model(images)
    test_targets.append(targets[0]['condition'].detach().cpu().numpy())
    encodings = [torch.flatten(p) for p in encoding_outputs]
    outputs = torch.cat(encodings).detach().cpu().numpy()
    outputs = [a for a in outputs]
    test_embeddings.append(outputs)
    num +=1

# print(test_embeddings)
test_embeddings = np.array(test_embeddings).reshape(num, len(test_embeddings[0]))
test_targets = np.array(test_targets)
print(test_targets)

# similarity_matrix01 = cosine_similarity(test_embeddings[test_targets==0], test_embeddings[test_targets==1])
# similarity_matrix12 = cosine_similarity(test_embeddings[test_targets==1], test_embeddings[test_targets==2])
# similarity_matrix02 = cosine_similarity(test_embeddings[test_targets==0], test_embeddings[test_targets==2])

# print(similarity_matrix01)
# print(similarity_matrix12)
# print(similarity_matrix02)

# average_similarity12 = similarity_matrix12.mean()
# average_similarity01 = similarity_matrix01.mean()
# average_similarity02 = similarity_matrix02.mean()
# print("Average Similarity 12:", average_similarity12)
# print("Average Similarity 02:", average_similarity02)
# print("Average Similarity 01:", average_similarity01)


####################### T-SNE ######################
# Create a two dimensional t-SNE projection of the embeddings
tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(test_embeddings)

# Plot those points as a scatter plot and label them based on the pred labels
cmap = cm.get_cmap('tab20')

fig = plt.figure(figsize=(10,5))
fig, ax = plt.subplots(figsize=(8,8))

for lab in range(n_condition_classes):
    indices = test_targets==lab
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=cm.get_cmap('Set1')(lab), label=f'Category: {conditions[lab]}', alpha=0.5)
    # ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=cm.get_cmap('Set1')(lab), alpha=0.5)

ax.legend(fontsize='large', markerscale=2)
plt.show()
plt.close(fig)
fig.savefig(f'TSNE/{args.model}-{args.exclude_condition}.png',bbox_inches='tight', dpi=150)


