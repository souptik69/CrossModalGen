import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import zipfile
from io import BytesIO
import pickle5 as pickle1

def generate_category_list():
    file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_Categories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list


class AVVPDataset(Dataset):
    # for AVEL task
    def __init__(self, meta_csv_path, fea_base_path, split='train', modality='video'):
        super(AVVPDataset, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path,sep='\t')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} positive classes in AVVP, 1 negative classes in AVVP')
        print(f'{len(self.split_df)} samples are used for {split}')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        categorys, video_id = one_video_df['event_labels'].split(','), one_video_df['filename']
        onsets, offsets = one_video_df['onset'].split(','), one_video_df['offset'].split(',')
        onsets = list(map(int, onsets))
        offsets = list(map(int, offsets))
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]
        
        avel_label = self._obtain_avel_label(onsets, offsets, categorys) # [10，26]
        
        return torch.from_numpy(fea), \
               torch.from_numpy(avel_label), \
               video_id
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea
    
    def _obtain_avel_label(self, onsets, offsets, categorys):# avc_label: [1, 10]
        T, category_num = 10, len(self.all_categories)
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
        label[:, -1] = np.ones(T) 
        iter_num = len(categorys)
        for i in range(iter_num):
            avc_label = np.zeros(T)
            avc_label[onsets[i]:offsets[i]] = 1
            class_id = self.all_categories.index(categorys[i])
            bg_flag = 1 - avc_label
            for j in range(10):
                label[j, class_id] = int(label[j, class_id]) | int(avc_label[j])
            for j in range(10):
                label[j, -1] = int(label[j, -1]) & int(bg_flag[j])
        return label 

    def __len__(self,):
        return len(self.split_df)


class AVVPDatasetTrain(Dataset):
    # for AVEL task
    def __init__(self, meta_csv_path, fea_base_path, split='train', modality='video'):
        super(AVVPDatasetTrain, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path, sep='\t')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in AVVPTrain')
        print(f'{len(self.split_df)} samples are used for Train')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        categorys, video_id = one_video_df['event_labels'].split(','), one_video_df['filename']
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]

        avc_label = np.ones(10) # [10，1]
        avel_label = self._obtain_avel_label(avc_label, categorys) # [10，26]

        return torch.from_numpy(fea), \
               torch.from_numpy(avel_label)
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea


    def _obtain_avel_label(self, avc_label, categorys):# avc_label: [1, 10]
        T, category_num = 10, len(self.all_categories)

        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
        for category in categorys:
            class_id = self.all_categories.index(category)
            bg_flag = 1 - avc_label
            label[:, class_id] = avc_label
            label[:, -1] = bg_flag

        return label 

    def __len__(self,):
        return len(self.split_df)
    
class AVVPDatasetEval(Dataset):
    # for AVEL task
    def __init__(self, meta_csv_path, fea_base_path, split='train', modality='video'):
        super(AVVPDatasetEval, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path)
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in AVVPEval')
        print(f'{len(self.split_df)} samples are used for Eval')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, video_id = one_video_df['event_labels'], one_video_df['filename']
        onset, offset = one_video_df['onset'].astype(int), one_video_df['offset'].astype(int)
        
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]
        
        fea = fea[onset:offset, :]
        
        avc_label = np.ones(offset-onset) # [offset-onset，1]
        avel_label = self._obtain_avel_label(onset, offset, avc_label, category) # [offset-onset，26]
        sample = {'feature': torch.from_numpy(fea), 'label': torch.from_numpy(avel_label), 'length':offset-onset}

        return sample

        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea


    def _obtain_avel_label(self, onset, offset, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = offset-onset, len(self.all_categories)
        label = np.zeros((T, category_num + 1)) 
        bg_flag = 1 - avc_label

        label[:, class_id] = avc_label
        label[:, -1] = bg_flag

        return label 

    def __len__(self,):
        return len(self.split_df)


class AVVPTestDataset(Dataset):
    """
    Dataset class for testing the AVVP models without onset/offset information.
    Uses the simplified test CSV format with just filenames and event labels.
    """
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, modality='both'):
        """
        Initialize the AVVP test dataset.
        
        Args:
            meta_csv_path: Path to the test CSV file
            audio_fea_base_path: Path to the audio feature zip files
            video_fea_base_path: Path to the video feature zip files
            modality: Which modality to load ('audio', 'video', or 'both')
        """
        super(AVVPTestDataset, self).__init__()
        self.modality = modality
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        
        # Load the CSV file
        self.split_df = pd.read_csv(meta_csv_path, sep='\t')
        
        # Load all categories
        self.all_categories = generate_category_list()
        print(f'AVVP test dataset initialized with {len(self.all_categories)} categories')
        print(f'Found {len(self.split_df)} test samples')

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        # Get data for this sample
        one_video_df = self.split_df.iloc[index]
        video_id = one_video_df['filename']
        
        # Parse the event labels string into a list
        if pd.isna(one_video_df['event_labels']):
            categorys = []
        else:
            categorys = one_video_df['event_labels'].split(',')
        
        # Load features
        audio_fea = None
        video_fea = None
        
        if self.modality in ['audio', 'both']:
            audio_fea = self._load_fea(self.audio_fea_base_path, video_id[:11])
            # Ensure we have exactly 10 timesteps for audio
            if audio_fea.shape[0] < 10:
                cur_t = audio_fea.shape[0]
                add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
                audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
            elif audio_fea.shape[0] > 10:
                audio_fea = audio_fea[:10, :]
        
        if self.modality in ['video', 'both']:
            video_fea = self._load_fea(self.video_fea_base_path, video_id[:11])
            # Ensure we have exactly 10 timesteps for video
            if video_fea.shape[0] < 10:
                cur_t = video_fea.shape[0]
                add_arr = np.tile(video_fea[-1, :], (10-cur_t, 1))
                video_fea = np.concatenate([video_fea, add_arr], axis=0)
            elif video_fea.shape[0] > 10:
                video_fea = video_fea[:10, :]
        
        # Create one-hot encoded labels
        # For test data without onset/offset, we assume the event is present in the entire video
        avel_label = self._obtain_avel_label(categorys)
        
        return {
            'video_id': video_id,
            'audio_fea': torch.from_numpy(audio_fea) if audio_fea is not None else None,
            'video_fea': torch.from_numpy(video_fea) if video_fea is not None else None,
            'label': torch.from_numpy(avel_label),
            'categories': categorys
        }
        
    def _load_fea(self, fea_base_path, video_id):
        """Load features from zip file"""
        fea_path = os.path.join(fea_base_path, "%s.zip" % video_id)
        
        class NumpyCompatUnpickler(pickle1.Unpickler):
            def find_class(self, module, name):
                if module.startswith('numpy') and name == '_reconstruct':
                    return np.core.multiarray._reconstruct
                # Redirect numpy._core to numpy
                if module.startswith('numpy._core'):
                    module = 'numpy'
                return super().find_class(module, name)
                
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    try:
                        fea = pickle.load(content)
                    except:
                        content.seek(0)
                        fea = NumpyCompatUnpickler(content).load()
        return fea
    
    def _obtain_avel_label(self, categorys):
        """
        Convert category list to one-hot encoded tensor.
        For test data without onset/offset, assume events are present throughout the video.
        
        Args:
            categorys: List of category names for this sample
            
        Returns:
            One-hot encoded tensor of shape [10, num_categories+1]
        """
        T, category_num = 10, len(self.all_categories)
        label = np.zeros((T, category_num + 1))  # Add one slot for 'background'
        
        # Initially set all to background
        label[:, -1] = np.ones(T)
        
        # Set the active categories
        for category in categorys:
            if category in self.all_categories:  # Skip categories not in our list
                class_id = self.all_categories.index(category)
                label[:, class_id] = 1  # Mark this category as present for all timesteps
                label[:, -1] = 0  # If any category is present, it's not background
        
        return label
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.split_df)
    
    def get_categories(self):
        """Return the list of categories"""
        return self.all_categories