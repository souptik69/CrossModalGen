import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import zipfile
from io import BytesIO
import json
    
# AT
class MSCOCODataset(Dataset):
    def __init__(self):
        super(MSCOCODataset, self).__init__()
        with open('/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/annotations/captions_val2017.json', 'r') as f:
            # self.id2cap = json.load(f)
            # self.id2cap = self.id2cap['annotations']
            data = json.load(f)
            annotations = data['annotations']
        self.id2cap = sorted(annotations, key=lambda x: x['image_id'])
        self.video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/val2017_features'
        # import pdb
        # pdb.set_trace()
        # print('22')


    # def __getitem__(self, index):
    #     video_name, cap = self.id2cap[index]['image_id'], self.id2cap[index]['caption']
    #     video_name = f"{video_name:012}"
    #     video_fea = self._load_fea(self.video_fea_base_path, video_name) 
    #     video_fea = np.expand_dims(video_fea, axis=0)
    #     length = video_fea.shape[0]

    #     # video_fea = torch.mean(video_fea,dim=0,keepdim=True)

    #     return video_fea, cap, index, length, index

    def __getitem__(self, index):
        video_name, cap = self.id2cap[index]['image_id'], self.id2cap[index]['caption']
        video_name = f"{video_name:012}"
        video_fea = self._load_fea(self.video_fea_base_path, video_name)
        video_fea = np.expand_dims(video_fea, axis=0)
        length = video_fea.shape[0]
        return video_fea, cap, index, length, index

    def _load_fea(self, fea_base_path, video_name):
        import pickle5 as pickle1
        
        class NumpyCompatUnpickler(pickle1.Unpickler):
            def find_class(self, module, name):

                if module.startswith('numpy') and name == '_reconstruct':
                    return np.core.multiarray._reconstruct
                # Redirect numpy._core to numpy
                if module.startswith('numpy._core'):
                    module = 'numpy'
                return super().find_class(module, name)

        fea_path = os.path.join(fea_base_path, "%s.zip"%video_name)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = NumpyCompatUnpickler(content).load()
        return fea

    def __len__(self):
        return len(self.id2cap)
    