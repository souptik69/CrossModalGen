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
    
    def get_categories(self):
        return self.all_categories


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



class AVVPMultimodalDatasetSimplified(Dataset):
    """
    Simplified multimodal dataset that works with the existing simplified CSV format
    """
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'):
        super(AVVPMultimodalDatasetSimplified, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        
        # Load and filter the simplified CSV
        full_df = pd.read_csv(meta_csv_path, sep='\t')
        
        # Group by filename to get pairs
        self.paired_data = []
        for filename in full_df['filename'].unique():
            file_entries = full_df[full_df['filename'] == filename]
            visual_entries = file_entries[file_entries['modality'] == 'visual']
            audio_entries = file_entries[file_entries['modality'] == 'audio']
            
            if len(visual_entries) > 0 and len(audio_entries) > 0:
                # Take the first entry of each modality
                visual_entry = visual_entries.iloc[0]
                audio_entry = audio_entries.iloc[0]
                
                self.paired_data.append({
                    'filename': filename,
                    'visual_onset': visual_entry['onset'],
                    'visual_offset': visual_entry['offset'],
                    'visual_labels': visual_entry['event_labels'],
                    'audio_onset': audio_entry['onset'],
                    'audio_offset': audio_entry['offset'],
                    'audio_labels': audio_entry['event_labels']
                })
        
        self.all_categories = self.generate_category_list()
        print(f'Total {len(self.all_categories)} classes in Simplified Multimodal AVVP')
        print(f'{len(self.paired_data)} paired samples are used for {split}')

    def generate_category_list(self):
        """Generate category list"""
        # Same as above
        file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_Categories.txt'
        category_list = []
        try:
            with open(file_path, 'r') as fr:
                for line in fr.readlines():
                    category_list.append(line.strip())
        except FileNotFoundError:
            category_list = [
                'Acoustic_guitar', 'Baby_cry_infant_cry', 'Baby_laughter', 'Banjo', 'Basketball_bounce',
                'Blender', 'Car', 'Cat', 'Cello', 'Chainsaw', 'Cheering', 'Chicken_rooster', 
                'Clapping', 'Dog', 'Fire_alarm', 'Frying_(food)', 'Helicopter', 'Lawn_mower',
                'Motorcycle', 'Singing', 'Speech', 'Telephone_bell_ringing', 'Vacuum_cleaner',
                'Violin_fiddle', 'Accordion'
            ]
        return category_list

    def __getitem__(self, index):
        entry = self.paired_data[index]
        video_id = entry['filename']
        
        # Load features
        audio_fea = self._load_fea(self.audio_fea_base_path, video_id[:11])
        video_fea = self._load_fea(self.video_fea_base_path, video_id[:11])
        
        # Normalize temporal dimensions
        audio_fea = self._normalize_temporal_dim(audio_fea, target_length=10)
        video_fea = self._normalize_temporal_dim(video_fea, target_length=10)
        
        # Parse temporal and label information
        visual_onsets, visual_offsets = self._parse_onset_offset(entry['visual_onset'], entry['visual_offset'])
        audio_onsets, audio_offsets = self._parse_onset_offset(entry['audio_onset'], entry['audio_offset'])
        
        visual_labels = self._parse_event_labels(entry['visual_labels'])
        audio_labels = self._parse_event_labels(entry['audio_labels'])
        
        # Create combined labels (union of visual and audio labels)
        combined_labels = list(set(visual_labels + audio_labels))
        combined_onsets = visual_onsets + audio_onsets
        combined_offsets = visual_offsets + audio_offsets
        
        # Create AVEL labels
        avel_label = self._obtain_avel_label(combined_onsets, combined_offsets, combined_labels)
        
        return torch.from_numpy(audio_fea), torch.from_numpy(video_fea), \
               torch.from_numpy(avel_label), video_id

    def _parse_onset_offset(self, onset_str, offset_str):
        """Parse onset and offset strings"""
        if ',' in str(onset_str):
            onsets = [int(x) for x in str(onset_str).split(',')]
        else:
            onsets = [int(onset_str)]
        
        if ',' in str(offset_str):
            offsets = [int(x) for x in str(offset_str).split(',')]
        else:
            offsets = [int(offset_str)]
        
        return onsets, offsets

    def _parse_event_labels(self, labels_str):
        """Parse event labels"""
        if pd.isna(labels_str):
            return []
        return [label.strip() for label in str(labels_str).split(',')]

    def _normalize_temporal_dim(self, fea, target_length=10):
        """Normalize feature temporal dimension"""
        if fea.shape[0] < target_length:
            cur_t = fea.shape[0]
            add_arr = np.tile(fea[-1, :], (target_length - cur_t, 1))
            fea = np.concatenate([fea, add_arr], axis=0)
        elif fea.shape[0] > target_length:
            fea = fea[:target_length, :]
        return fea

    def _load_fea(self, fea_base_path, video_id):
        """Load features from zip file"""
        fea_path = os.path.join(fea_base_path, f"{video_id}.zip")
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def _obtain_avel_label(self, onsets, offsets, categorys):
        """Create AVEL labels"""
        T, category_num = 10, len(self.all_categories)
        label = np.zeros((T, category_num + 1))
        label[:, -1] = np.ones(T)
        
        for i in range(len(categorys)):
            if i < len(onsets) and i < len(offsets):
                avc_label = np.zeros(T)
                avc_label[onsets[i]:offsets[i]] = 1
                class_id = self.all_categories.index(categorys[i])
                bg_flag = 1 - avc_label
                
                for j in range(10):
                    label[j, class_id] = int(label[j, class_id]) | int(avc_label[j])
                    label[j, -1] = int(label[j, -1]) & int(bg_flag[j])
        
        return label

    def __len__(self):
        return len(self.paired_data)

