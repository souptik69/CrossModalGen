import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        # Now for the supervised task
        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True
        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]

        return visual_feat, audio_feat, label


    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num


class AVETestDataset(Dataset):
    """
    Dataset class for testing the AVE models.
    Loads audio and video features from H5 files and connects them with category information
    from the annotations file.
    """
    def __init__(self, data_root, annotations_path):
        super(AVETestDataset, self).__init__()
        # Paths to feature files
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, 'test_order.h5')
        
        # Load category information from annotations file
        self.categories = []
        seen_categories = set()
        
        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('&')
                category = parts[0]
                
                if category not in seen_categories:
                    seen_categories.add(category)
                    self.categories.append(category)
        
        print(f"AVETestDataset initialized with {len(self.categories)} categories")
        self.h5_isOpen = False
    
    def __getitem__(self, index):
        """Get a sample from the dataset"""
        if not self.h5_isOpen:
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True
        
        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]
        
        return visual_feat, audio_feat, label
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num
    
    def get_categories(self):
        
        return self.categories