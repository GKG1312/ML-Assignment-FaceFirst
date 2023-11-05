import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils import generate_target
 
    
class MaskDataset(object):
    def __init__(self, csv_file, img_dir, label_dir, transforms=None):
        self.transforms = transforms
        self.list = csv_file
        self.img_dir = img_dir
        self.label_dir = label_dir
        # print(self.list)

    def __getitem__(self, idx):
        # load images ad masks
        # file_image = 'maksssksksss'+ str(idx) + '.png'
        # file_label = 'maksssksksss'+ str(idx) + '.xml'
        img_path = os.path.join(self.img_dir, self.list.iloc[idx][0])
        label_path = os.path.join(self.label_dir, self.list.iloc[idx][1])
        img = Image.open(img_path).convert("RGB")
        img_shape = np.asarray(img).shape
        #Generate Label
        target = generate_target(idx, label_path, img_shape)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return torch.tensor(np.array(img)/255).permute(2,0,1).type(torch.FloatTensor), target

    def __len__(self):
        return len(self.list)