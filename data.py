from torch.utils.data import Dataset
from skimage.io import imread
from PIL import Image
import numpy as np
import torch
import json, os
import aux_func

def custom_collate(batch):
    images = torch.stack([b['image'] for b in batch])
    #rel2abs_size = torch.tensor([[images.shape[3], images.shape[2]]*2], dtype=torch.float32)
    targets = [{
        # bboxes must be in format 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
        'boxes': torch.tensor(b['bboxes'], dtype=torch.float32), # * rel2abs_size,
        # we add 1 to the labels so they start at 1
        'labels': torch.tensor(b['labels'], dtype=torch.int64) + 1,
        'condition': torch.tensor(b['condition'], dtype=torch.int64),
    } if len(b['bboxes']) > 0 else {
        'boxes': torch.empty(0, 4),
        'labels': torch.empty(0, dtype=torch.int64),
        'condition': torch.tensor(b['condition'], dtype=torch.int64),
    } for b in batch]
    return images, targets

######################### BDD100K ###########################

class BDD100K(Dataset):
    K = 10
    categories = ['person', 'traffic sign', 'traffic light', 'car', 'bus', 'truck', 'bike', 'rider', 'motor', 'train']
    weathers = ['rainy', 'snowy', 'clear', 'overcast', 'partly cloudy', 'foggy']
    scenes = ['tunnel', 'residential', 'parking lot', 'city street', 'gas stations', 'highway']
    timeofdays = ['daytime', 'night', 'dawn/dusk']
    conditions = {'weather': weathers, 'scene': scenes, 'timeofday': timeofdays}

    def __init__(self, root, fold, condition, transform=None):
        fold = 'val' if fold == 'test' else fold
        assert fold in ('train', 'val')
        assert condition in ('weather', 'scene', 'timeofday')
        self.root = root
        self.fold = fold
        self.dict_transform = transform
        # h, w = 720, 1280
        props = json.load(open(os.path.join(root, 'bdd100k', 'labels', f'bdd100k_labels_images_{fold}.json')))
        conditions = self.conditions[condition]
        self.props = [{
            'image': f"{root}/bdd100k/images/100k/{fold}/{p['name']}",
            'bboxes': [(l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2']) for l in p['labels'] if 'box2d' in l and l['category'] in self.categories],
            'labels': [self.categories.index(l['category']) for l in p['labels'] if 'box2d' in l and l['category'] in self.categories],
            'condition': conditions.index(p['attributes'][condition]),
        } for p in props if p['attributes'][condition] != 'undefined']
        # filter properties without objects
        self.props = [p for p in self.props if len(p['bboxes']) > 0]

    def __len__(self):
        return len(self.props)

    def __getitem__(self, i):
        d = self.props[i]
        d = {**d, 'image': imread(d['image'])} # adds the image to the dictionary which already contains keys
        if self.dict_transform:
            d = self.dict_transform(**d)
        return d

    def get_condition(self, i):  # faster than getitem if we only want the condition
        return self.props[i]['condition']


class RestrictCondition(Dataset):
    ''' Choose only certain specific conditions within a general one.
    For instance, choose all the 'weather' conditions except 'partly cloudy'.
    '''
    def __init__(self, ds, allowed_conditions):
        self.ds = ds
        self.allowed_conditions = allowed_conditions
        self.ix = [i for i in range(len(ds)) if ds.get_condition(i) in allowed_conditions]

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        d = self.ds[self.ix[i]]
        d['condition'] = self.allowed_conditions.index(d['condition'])
        return d

######################### DAWN ###########################

class DAWN(Dataset):
    K = 6
    categories = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'person']
    weathers = ['fog', 'rain', 'sand', 'snow']
    conditions = {'weather': weathers}

    def __init__(self, root, fold, condition, transform=None):
        
        assert fold in ('train', 'val', 'test')
        assert condition == 'weather'
        
        self.root = root
        self.fold = fold
        self.dict_transform = transform
        
        #### XML Reader #######

        import xml.etree.ElementTree as ET 
        import os
        
        categories = self.categories
        weathers = self.weathers

        rand = np.random.RandomState(123)
        
        self.props = []

        for weather in weathers:
            files = sorted(os.listdir(os.path.join(self.root, 'DAWN', weather.capitalize(), f'{weather.capitalize()}_PASCAL_VOC')))
            ix = rand.choice(len(files), len(files), False)

            if fold == 'train':
                ix = ix[:int(0.70*len(files))]
            if fold == 'val':
                ix = ix[int(0.70*len(files)):int(0.85*len(files))]
            if fold == 'test':
                ix = ix[int(0.85*len(files)):]
            
            files = [files[i] for i in ix]
            
            for filename in files:
                tree = ET.parse(f'{self.root}/DAWN/{weather.capitalize()}/{weather.capitalize()}_PASCAL_VOC/{filename}') 
                root = tree.getroot()
                
                fn = root[1].text  ### POR CAUSA DE UMA IMAGEM QUE NAO TEM O FILENAME CORRETO!!!
                if fn[-4:] != '.jpg':
                    fn = fn +'.jpg'

                self.props.append({
                    'image': f"{self.root}/DAWN/{weather.capitalize()}/{fn}",
                    'bboxes': [(float(obj[4][0].text), float(obj[4][1].text), float(obj[4][2].text), float(obj[4][3].text)) for obj in root.iter('object') if obj[0].text in categories],
                    'labels': [categories.index(obj[0].text) for obj in root.iter('object') if obj[0].text in categories],
                    'condition': weathers.index(weather),
                })

        self.props = [p for p in self.props if len(p['bboxes']) > 0]


    def __len__(self):
        return len(self.props)

    def __getitem__(self, i):
        d = self.props[i]
        d = {**d, 'image': np.array(Image.open(d['image']).convert('RGB'))} # DIFFERENT FROM IMREAD BECAUSE THERE WAS 1 IMAGE IN GRAYSCALE!!!
            
        if self.dict_transform:
            d = self.dict_transform(**d)
        return d

    def get_condition(self, i):  # faster than getitem if we only want the condition
        return self.props[i]['condition']


######################## SODA #############################

class SODA(Dataset):
    K=6
    categories = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']
    categories_id = [1,2,3,4,5,6] # id starts in 1
    weathers = ['Clear', 'Overcast', 'Rainy']
    locations = ['Citystreet', 'Countryroad', 'Highway']
    periods = ['Daytime', 'Night']
    conditions = {'weather': weathers, 'location': locations, 'period': periods}

    def __init__(self, root, fold, condition, transform=None):
        fold = 'val' if fold == 'test' else fold
        assert fold in ('train', 'val')
        assert condition in ('weather', 'location', 'period')
        
        self.root = root
        self.fold = fold
        self.dict_transform = transform
        
        conditions = self.conditions[condition]
        rand = np.random.RandomState(123)

        self.props = []
        for fold_soda in ['train', 'val']:
            data = json.load(open(os.path.join('/data/auto', 'SODA10M', 'SSLAD-2D', 'labeled', 'annotations', f'instance_{fold_soda}.json')))
            props = [{
                    'image': f"{root}/SODA10M/SSLAD-2D/labeled/{fold_soda}/{p['file_name']}",
                    # 'bboxes': [(l['bbox'][0], l['bbox'][1], l['bbox'][0]+l['bbox'][2], l['bbox'][1]+l['bbox'][3])for l in data['annotations'] if  l['image_id']==p['id'] and 'bbox' in l and l['category_id'] in self.categories_id],
                    'bboxes': [(max(l['bbox'][0], 0), max(l['bbox'][1], 0), min(l['bbox'][0]+l['bbox'][2], p['width']), min(l['bbox'][1]+l['bbox'][3], p['height']) )for l in data['annotations'] if  l['image_id']==p['id'] and 'bbox' in l and l['category_id'] in self.categories_id],
                    'labels': [l['category_id']-1 for l in data['annotations'] if l['image_id']==p['id'] and l['category_id'] in self.categories_id], # Starts in 1, instead 0
                    'condition': conditions.index(p[condition])
                } for p in data['images']]
            
            self.props += props
        self.props = [p for p in self.props if len(p['bboxes']) > 0]

        ix = rand.choice(len(self.props), len(self.props), False)
        if fold == 'train':
            ix = ix[:int(0.70*len(self.props))] 
        else:
            ix = ix[int(0.70*len(self.props)):]

        self.props = [self.props[i] for i in ix]


    def __len__(self):
        return len(self.props)

    def __getitem__(self, i):
        d = self.props[i]
        d = {**d, 'image': imread(d['image'])} # adds the image to the dictionary which already contains keys
        if self.dict_transform:
            d = self.dict_transform(**d)
        return d

    def get_condition(self, i):  # faster than getitem if we only want the condition
        return self.props[i]['condition']

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('condition', choices=['weather', 'scene', 'timeofday'])
    # args = parser.parse_args()
    # import matplotlib.pyplot as plt
    
    print('Weather')
    ds = DAWN('/data/auto', 'train', 'weather')
    
    selected_classes = [1,2]
    idx_list = []
    
    import random

    for cl in selected_classes:
        idx_list.append(random.choice([index for (index, item) in enumerate(ds) if item['condition']==cl]))
                

    print(idx_list)

    # print('DAWN Train- ', len(ds))
    # ds = DAWN('/data/auto', 'val', 'weather')
    # print('DAWN Test - ', len(ds))
    # ds = BDD100K('/data/auto', 'train', 'weather')
    # print('BDD100k Train - ', len(ds))
    # ds = BDD100K('/data/auto', 'val', 'weather')
    # print('BDD100k Test - ', len(ds))
    # ds = SODA('/data/auto', 'train', 'weather')
    # print('SODA Train - ', len(ds))
    # ds = SODA('/data/auto', 'val', 'weather')
    # print('SODA Test - ', len(ds))

    # print('Scene')
    # ds = BDD100K('/data/auto', 'train', 'scene')
    # print('BDD100k Train - ', len(ds))
    # ds = BDD100K('/data/auto', 'val', 'scene')
    # print('BDD100k Test - ', len(ds))
    # ds = SODA('/data/auto', 'train', 'location')
    # print('SODA Train - ', len(ds))
    # ds = SODA('/data/auto', 'val', 'location')
    # print('SODA Test - ', len(ds))

    # print('Time of Day')
    # ds = BDD100K('/data/auto', 'train', 'timeofday')
    # print('BDD100k Train - ', len(ds))
    # ds = BDD100K('/data/auto', 'val', 'timeofday')
    # print('BDD100k Test - ', len(ds))
    # ds = SODA('/data/auto', 'train', 'period')
    # print('SODA Train - ', len(ds))
    # ds = SODA('/data/auto', 'val', 'period')
    # print('SODA Test - ', len(ds))

    # i = 0
    # for d in ds:
    #     print(i, '-', d['image'])
    #     aux_func.bb_boxes_plot(d['img'], [{'bboxes': d['bboxes'],'labels': d['labels']}], DAWN.categories, i)
    #     i += 1


    
    # import albumentations as A
    # from albumentations.pytorch import ToTensorV2

    # H, W = 256, 512
    # transform = A.Compose([
    #     A.HorizontalFlip(),
    #     A.RandomBrightnessContrast(0.1),
    #     # A.Resize(int(H*1.1), int(W*1.1)),
    #     # A.RandomCrop(H, W),
    #     A.Normalize((0, 0, 0), (1, 1, 1)),
    #     ToTensorV2()
    # ])
    # #  bbox_params=A.BboxParams('pascal_voc', ['labels']))

    # xmin = 1000000
    # d_xmins = (0,0)
    # xmax = -1000000
    # d_xmaxs = (0,0)
    # ymin = 1000000
    # d_ymins = (0,0)
    # ymax = -1000000
    # d_ymaxs = (0,0)

    # ds = SODA('/data/auto', 'val', 'weather')
    # for d in ds:
    #     dims = d['img'].shape
    #     for b in d['bboxes']:
    #         if b[0] < 0:
    #             xmin = min(xmin, b[0])
    #             d_xmins = dims         
    #         if b[1] < 0:
    #             ymin = min(ymin, b[1])
    #             d_ymins = dims 
    #         if b[2] > dims[1]:
    #             xmax = max(xmax, b[2])
    #             d_xmaxs = dims
    #         if b[3] > dims[0]:
    #             ymax = max(ymax, b[3])
    #             d_ymaxs = dims

    # ds = SODA('/data/auto', 'train', 'weather')
    # for d in ds:
    #     dims = d['img'].shape
    #     for b in d['bboxes']:
    #         if b[0] < 0:
    #             xmin = min(xmin, b[0])
    #             d_xmins = dims         
    #         if b[1] < 0:
    #             ymin = min(ymin, b[1])
    #             d_ymins = dims 
    #         if b[2] > dims[1]:
    #             xmax = max(xmax, b[2])
    #             d_xmaxs = dims
    #         if b[3] > dims[0]:
    #             ymax = max(ymax, b[3])
    #             d_ymaxs = dims
    
    # print('xmin', xmin)
    # print('d_xmin', d_xmins)
    # print('xmax', xmax)
    # print('d_xmax', d_xmaxs)
    # print('ymin', ymin)
    # print('d_ymin', d_ymins)
    # print('ymax', ymax)
    # print('d_ymax', d_ymaxs)

    # m=0
    # for i in ds:
    #     aux_func.save_fig(i['image'],f'exampleDAWN{m}')
    #     m += 1
    #     if m == 10:
    #         break
        
    
    # ds = DAWN('/data/auto', 'train', 'weather')
    # for i in ds:
    #     print(i['condition'])

    # data0 = [a['condition'] for a in ds if a['condition']==0]
    # print('0: ', len(data0))

    # data1 = [a['condition'] for a in ds if a['condition']==1]
    # print('1: ', len(data1))

    # data2= [a['condition'] for a in ds if a['condition']==2]
    # print('2: ', len(data2))

    # data3 = [a['condition'] for a in ds if a['condition']==3]
    # print('3: ', len(data3))

    # data4 = [a['condition'] for a in ds if a['condition']==4]
    # print('4: ', len(data4))

    # data5= [a['condition'] for a in ds if a['condition']==5]
    # print('5: ', len(data5))



    
    # for d in ds:
    #     plt.imshow(d['image'])
    #     plt.title(f"Condition: {BDD100K.conditions[args.condition][d['condition']]} ({d['condition']})")
    #     plt.show()