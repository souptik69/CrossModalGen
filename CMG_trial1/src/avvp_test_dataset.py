import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import zipfile
from io import BytesIO

def generate_category_list():
    """Load category names from AVVP Categories file"""
    file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_Categories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

class AVVPTestDataset(Dataset):
    def __init__(self, meta_csv_path, fea_base_path, modality='audio'):
        super(AVVPTestDataset, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path, sep='\t')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in AVVPTest')
        print(f'{len(self.split_df)} samples are used for Test')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        # Get the first event for simplicity
        event_labels = one_video_df['event_labels'].split(',')
        onsets = one_video_df['onset'].split(',')
        offsets = one_video_df['offset'].split(',')
        
        category = event_labels[0]
        onset = int(onsets[0])
        offset = int(offsets[0])
        video_id = one_video_df['filename']
        
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        
        if fea.shape[0] < 10:
            cur_t = fea.shape[0]
            add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
            fea = np.concatenate([fea, add_arr], axis=0)
        elif fea.shape[0] > 10:
            fea = fea[:10, :]
        
        # Get the segment between onset and offset
        fea = fea[onset:offset, :]
        
        avc_label = np.ones(offset-onset)  # [offset-onset，1]
        avel_label = self._obtain_avel_label(onset, offset, avc_label, category)  # [offset-onset，26]
        
        return {
            'feature': torch.from_numpy(fea), 
            'label': torch.from_numpy(avel_label), 
            'length': offset-onset,
            'video_id': video_id,
            'category': category
        }
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip" % video_id)
        try:
            with zipfile.ZipFile(fea_path, mode='r') as zfile:
                for name in zfile.namelist():
                    if '.pkl' not in name:
                        continue
                    with zfile.open(name, mode='r') as fea_file:
                        content = BytesIO(fea_file.read())
                        fea = pickle.load(content)
            return fea
        except Exception as e:
            print(f"Error loading features for {video_id}: {e}")
            # Return a default tensor if there's an error
            if self.modality == 'audio':
                return np.zeros((10, 128))
            else:
                return np.zeros((10, 512))

    def _obtain_avel_label(self, onset, offset, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = offset-onset, len(self.all_categories)
        label = np.zeros((T, category_num + 1)) 
        bg_flag = 1 - avc_label

        label[:, class_id] = avc_label
        label[:, -1] = bg_flag

        return label 

    def __len__(self):
        return len(self.split_df)
        
    def get_categories(self):
        """Return the list of categories"""
        return self.all_categories
