import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pickle
import zipfile
from io import BytesIO
import pdb
import pickle5 as pickle1
import random
SEED = 57
random.seed(SEED)

def generate_category_list():
    file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/100kcategories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

#VGG pretraining for downstream, entire dataset
class VGGSoundDataset_AV(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path,split='train'):
        super(VGGSoundDataset_AV, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        self.avc_label_base_path = avc_label_base_path
        all_df = pd.read_csv(meta_csv_path)

        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        self.split_df = pd.concat([df_train,df_test,df_val])
        # self.split_df = pd.concat([df_train,df_val])
        

        print(f'{len(self.split_df)}/{len(all_df)} samples are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in Vggsound100K_AVT train, val, test')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, audio_id = one_video_df['category'], one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, audio_id) # [10, 7, 7, 512]
        avc_label = self._load_fea(self.avc_label_base_path, audio_id) # [10，1]
        avel_label = self._obtain_avel_label(avc_label, category) # [10，142]
        
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        sample = {'video_fea': video_fea, 'audio_fea': audio_fea, 'avel_label': avel_label}
        return sample

    def _load_fea(self, fea_base_path, audio_id):
        import pickle5 as pickle1
        
        class NumpyCompatUnpickler(pickle1.Unpickler):
            def find_class(self, module, name):

                if module.startswith('numpy') and name == '_reconstruct':
                    return np.core.multiarray._reconstruct
                # Redirect numpy._core to numpy
                if module.startswith('numpy._core'):
                    module = 'numpy'
                return super().find_class(module, name)
        
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = NumpyCompatUnpickler(content).load()
        return fea
    
    def _obtain_avel_label(self, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = 10, len(self.all_categories)

        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 141+1]
        bg_flag = 1 - avc_label
        
        # label[:, class_id] = avc_label
        label[:, class_id] = avc_label.reshape(-1)
        # label[:, -1] = bg_flag
        label[:, -1] = bg_flag.reshape(-1)
        return label 


    def __len__(self,):
        return len(self.split_df)


#VGG pretraining for downstream, entire dataset 40k
class VGGSoundDataset_AV_1(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'):
        super(VGGSoundDataset_AV_1, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        all_df = pd.read_csv(meta_csv_path)

        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        self.split_df = pd.concat([df_train,df_test,df_val])
        # self.split_df = pd.concat([df_train,df_val])
        

        print(f'{len(self.split_df)}/{len(all_df)} samples are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in Vggsound100K_AVT train, val, test')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        audio_id = one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, audio_id) # [10, 7, 7, 512]
        
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        sample = {'video_fea': video_fea, 'audio_fea': audio_fea}
        return sample

    def _load_fea(self, fea_base_path, audio_id):
        import pickle5 as pickle1
        
        class NumpyCompatUnpickler(pickle1.Unpickler):
            def find_class(self, module, name):

                if module.startswith('numpy') and name == '_reconstruct':
                    return np.core.multiarray._reconstruct
                # Redirect numpy._core to numpy
                if module.startswith('numpy._core'):
                    module = 'numpy'
                return super().find_class(module, name)
        
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = NumpyCompatUnpickler(content).load()
        return fea



    def __len__(self,):
        return len(self.split_df)


#VGG training for testing on VGG

class VGGSoundDataset_AV_new(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path,split='train'):
        super(VGGSoundDataset_AV_new, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        self.avc_label_base_path = avc_label_base_path
        all_df = pd.read_csv(meta_csv_path)

        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        # self.split_df = pd.concat([df_train,df_test,df_val])
        self.split_df = pd.concat([df_train,df_val])
        

        print(f'{len(self.split_df)}/{len(all_df)} samples are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in Vggsound100K_AVT train, val, test')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, audio_id = one_video_df['category'], one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, audio_id) # [10, 7, 7, 512]
        avc_label = self._load_fea(self.avc_label_base_path, audio_id) # [10，1]
        avel_label = self._obtain_avel_label(avc_label, category) # [10，142]
        
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        sample = {'video_fea': video_fea, 'audio_fea': audio_fea, 'avel_label': avel_label}
        return sample

    def _load_fea(self, fea_base_path, audio_id):
        import pickle5 as pickle1
        
        class NumpyCompatUnpickler(pickle1.Unpickler):
            def find_class(self, module, name):

                if module.startswith('numpy') and name == '_reconstruct':
                    return np.core.multiarray._reconstruct
                # Redirect numpy._core to numpy
                if module.startswith('numpy._core'):
                    module = 'numpy'
                return super().find_class(module, name)
        
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = NumpyCompatUnpickler(content).load()
        return fea
    
    def _obtain_avel_label(self, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = 10, len(self.all_categories)

        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 141+1]
        bg_flag = 1 - avc_label
        
        # label[:, class_id] = avc_label
        label[:, class_id] = avc_label.reshape(-1)
        # label[:, -1] = bg_flag
        label[:, -1] = bg_flag.reshape(-1)
        return label 


    def __len__(self,):
        return len(self.split_df)
    

##Testing on VGG 
class VGGSoundDataset_AV_test(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'):
        super(VGGSoundDataset_AV_test, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        self.avc_label_base_path = avc_label_base_path
        all_df = pd.read_csv(meta_csv_path)

        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        self.split_df = pd.concat([df_test])


        print(f'{len(self.split_df)}/{len(all_df)} samples are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in Vggsound100K_AVT')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, audio_id = one_video_df['category'], one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, audio_id) # [10, 7, 7, 512]
        avc_label = self._load_fea(self.avc_label_base_path, audio_id) # [10，1]
        avel_label = self._obtain_avel_label(avc_label, category) # [10，142]
        
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        sample = {'video_fea': video_fea, 'audio_fea': audio_fea, 'avel_label': avel_label}
        return sample

    def _load_fea(self, fea_base_path, audio_id):
        
        class NumpyCompatUnpickler(pickle1.Unpickler):
            def find_class(self, module, name):

                if module.startswith('numpy') and name == '_reconstruct':
                    return np.core.multiarray._reconstruct
                # Redirect numpy._core to numpy
                if module.startswith('numpy._core'):
                    module = 'numpy'
                return super().find_class(module, name)
        
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = NumpyCompatUnpickler(content).load()
        return fea
    
    def _obtain_avel_label(self, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = 10, len(self.all_categories)

        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 141+1]
        bg_flag = 1 - avc_label
        
        # label[:, class_id] = avc_label
        label[:, class_id] = avc_label.reshape(-1)
        # label[:, -1] = bg_flag
        label[:, -1] = bg_flag.reshape(-1)
        return label 


    def __len__(self,):
        return len(self.split_df)
