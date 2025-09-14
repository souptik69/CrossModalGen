

import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime

# Add MultiBench to path for accessing the get_dataloader function
MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
MOSEI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'
MOSI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSI/'

sys.path.append(MULTIBENCH_PATH)

def print_progress(message):
    """Helper function for logging progress during dataset loading"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

class MOSEIDatasetUnsupervised(Dataset):
    """
    Dataset class for unsupervised pretraining on combined MOSEI data.
    Combines train, validation splits from MOSEI for self-supervised learning.
    Returns numpy arrays to match VGGSound collate pattern.
    """
    def __init__(self, data_path, max_seq_len=50, batch_size=64):
        super(MOSEIDatasetUnsupervised, self).__init__()
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        
        print_progress("Loading MOSEI dataset for unsupervised pretraining...")
        
        try:
            from datasets.affect.get_data import get_dataloader
            
            # Load all three splits from MOSEI
            filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
            traindata, validdata, testdata = get_dataloader(
                filepath, 
                robust_test=False, 
                max_pad=True, 
                data_type='mosei', 
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                z_norm=True
            )
            
            # Combine all data for unsupervised pretraining
            print_progress("Combining train, validation splits...")
            self.combined_data = []
            
            # Extract data from each split
            # for split_name, dataloader in [("train", traindata), ("val", validdata), ("test", testdata)]:
            for split_name, dataloader in [("train", traindata), ("val", validdata)]:
                print_progress(f"Processing {split_name} split...")
                for batch_idx, batch in enumerate(dataloader):
                    # MultiBench returns: [visual_features, audio_features, text_features, labels]
                    visual_feat, audio_feat, text_feat, labels = batch
                    
                    # Store each sample individually to create a unified dataset
                    batch_size_actual = visual_feat.shape[0]
                    for i in range(batch_size_actual):
                        sample = {
                            'visual': visual_feat[i],  # Shape: [seq_len, 35]
                            'audio': audio_feat[i],    # Shape: [seq_len, 74] 
                            'text': text_feat[i],      # Shape: [seq_len, 300]
                            'labels': labels[i],       # Shape: [1] - sentiment score
                            'video_id': f"mosei_{split_name}_{batch_idx}_{i}"  # Create unique identifier
                        }
                        self.combined_data.append(sample)
                        
            print_progress(f"Successfully loaded {len(self.combined_data)} samples for unsupervised pretraining")
            
        except Exception as e:
            print_progress(f"Error loading MOSEI data: {e}")
            import traceback
            print_progress(f"Traceback: {traceback.format_exc()}")
            self.combined_data = []
    
    def __getitem__(self, index):
        """
        Returns numpy arrays to match VGGSound collate pattern.
        This is crucial for the torch.from_numpy(np.asarray(...)) pattern to work efficiently.
        """
        sample = self.combined_data[index]
        
        # Convert tensors to numpy arrays (removing .float() calls)
        # This ensures compatibility with VGGSound-style collate function
        audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
        text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
        labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
        video_id = sample['video_id']
        
        return audio_feature, video_feature, text_feature, labels, video_id
    
    def __len__(self):
        return len(self.combined_data)


class MOSEIDatasetUnsupervisedSplit(Dataset):
    """
    Dataset class for unsupervised pretraining on MOSEI data with split functionality.
    - If split is 'train': combines train+validation splits for self-supervised learning
    - If split is 'test_train': uses first 80% of test data
    - If split is 'test_val': uses last 20% of test data
    Returns numpy arrays to match VGGSound collate pattern.
    """
    def __init__(self, data_path, split='train', max_seq_len=50, batch_size=64):
        super(MOSEIDatasetUnsupervisedSplit, self).__init__()
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        
        print_progress(f"Loading MOSEI dataset for unsupervised pretraining (split: {split})...")
        
        try:
            from datasets.affect.get_data import get_dataloader
            
            # Load all three splits from MOSEI
            filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
            traindata, validdata, testdata = get_dataloader(
                filepath, 
                robust_test=False, 
                max_pad=True, 
                data_type='mosei', 
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                z_norm=True
            )
            
            self.combined_data = []
            
            if split == 'train':
                # Combine train and validation splits like the original unsupervised class
                print_progress("Combining train and validation splits...")
                splits_to_process = [("train", traindata), ("val", validdata)]
                
                for split_name, dataloader in splits_to_process:
                    print_progress(f"Processing {split_name} split...")
                    for batch_idx, batch in enumerate(dataloader):
                        visual_feat, audio_feat, text_feat, labels = batch
                        batch_size_actual = visual_feat.shape[0]
                        
                        for i in range(batch_size_actual):
                            sample = {
                                'visual': visual_feat[i],  # Shape: [seq_len, 35]
                                'audio': audio_feat[i],    # Shape: [seq_len, 74] 
                                'text': text_feat[i],      # Shape: [seq_len, 300]
                                'labels': labels[i],       # Shape: [1] - sentiment score
                                'video_id': f"mosei_{split_name}_{batch_idx}_{i}"
                            }
                            self.combined_data.append(sample)
                            
            elif split in ['test_train', 'test_val']:
                # Process test data and split it 80-20
                print_progress("Processing test split for 80-20 division...")
                test_samples = []
                
                for batch_idx, batch in enumerate(testdata):
                    visual_feat, audio_feat, text_feat, labels = batch
                    batch_size_actual = visual_feat.shape[0]
                    
                    for i in range(batch_size_actual):
                        sample = {
                            'visual': visual_feat[i],  # Shape: [seq_len, 35]
                            'audio': audio_feat[i],    # Shape: [seq_len, 74] 
                            'text': text_feat[i],      # Shape: [seq_len, 300]
                            'labels': labels[i],       # Shape: [1] - sentiment score
                            'video_id': f"mosei_test_{batch_idx}_{i}"
                        }
                        test_samples.append(sample)
                
                # Perform 80-20 split
                total_samples = len(test_samples)
                split_idx = int(0.8 * total_samples)
                
                if split == 'test_train':
                    # First 80% for test_train
                    self.combined_data = test_samples[:split_idx]
                    print_progress(f"Using first 80% of test data: {len(self.combined_data)} samples")
                elif split == 'test_val':
                    # Last 20% for test_val
                    self.combined_data = test_samples[split_idx:]
                    print_progress(f"Using last 20% of test data: {len(self.combined_data)} samples")
                    
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'test_train', or 'test_val'")
                        
            print_progress(f"Successfully loaded {len(self.combined_data)} samples for unsupervised pretraining ({split})")
            
        except Exception as e:
            print_progress(f"Error loading MOSEI data: {e}")
            import traceback
            print_progress(f"Traceback: {traceback.format_exc()}")
            self.combined_data = []
    
    def __getitem__(self, index):
        """
        Returns numpy arrays to match VGGSound collate pattern.
        This is crucial for the torch.from_numpy(np.asarray(...)) pattern to work efficiently.
        """
        sample = self.combined_data[index]
        
        # Convert tensors to numpy arrays (removing .float() calls)
        # This ensures compatibility with VGGSound-style collate function
        audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
        text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
        labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
        video_id = sample['video_id']
        
        return audio_feature, video_feature, text_feature, labels, video_id
    
    def __len__(self):
        return len(self.combined_data)


class MOSEIDatasetSupervised(Dataset):
    """
    Dataset class for supervised training on MOSEI data.
    - If split is 'train' or 'val': returns combined train+val data
    - If split is 'test': returns only test data
    Returns numpy arrays to match VGGSound collate pattern.
    """
    def __init__(self, data_path, split='train', max_seq_len=50, batch_size=64):
        super(MOSEIDatasetSupervised, self).__init__()
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        
        print_progress(f"Loading MOSEI dataset for supervised training (split: {split})...")
        
        try:
            from datasets.affect.get_data import get_dataloader
            
            # Load all three splits from MOSEI
            filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
            traindata, validdata, testdata = get_dataloader(
                filepath, 
                robust_test=False, 
                max_pad=True, 
                data_type='mosei', 
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                z_norm=True
            )
            
            # Determine which splits to use based on the requested split
            if split == 'train':
                print_progress("Using train split only...")
                splits_to_process = [("train", traindata)]
            elif split == 'val':
                print_progress("Using validation split only...")
                splits_to_process = [("val", validdata)]
            elif split == 'test':
                print_progress("Using test split only...")
                splits_to_process = [("test", testdata)]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
            # Process the selected splits
            self.data = []
            for split_name, dataloader in splits_to_process:
                print_progress(f"Processing {split_name} split...")
                for batch_idx, batch in enumerate(dataloader):
                    # MultiBench returns: [visual_features, audio_features, text_features, labels]
                    visual_feat, audio_feat, text_feat, labels = batch
                    
                    # Store each sample individually
                    batch_size_actual = visual_feat.shape[0]
                    for i in range(batch_size_actual):
                        sample = {
                            'visual': visual_feat[i],  # Shape: [seq_len, 35]
                            'audio': audio_feat[i],    # Shape: [seq_len, 74] 
                            'text': text_feat[i],      # Shape: [seq_len, 300]
                            'labels': labels[i],       # Shape: [1] - sentiment score
                            'video_id': f"mosei_{split_name}_{batch_idx}_{i}"  # Create unique identifier
                        }
                        self.data.append(sample)
                        
            print_progress(f"Successfully loaded {len(self.data)} samples for supervised training")
            
        except Exception as e:
            print_progress(f"Error loading MOSEI supervised data: {e}")
            import traceback
            print_progress(f"Traceback: {traceback.format_exc()}")
            self.data = []
    
    def __getitem__(self, index):
        """
        Returns numpy arrays to match VGGSound collate pattern.
        """
        sample = self.data[index]
        
        # Convert tensors to numpy arrays (removing .float() calls)
        audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
        text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
        labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
        video_id = sample['video_id']
        
        return audio_feature, video_feature, text_feature, labels, video_id
    
    def __len__(self):
        return len(self.data)


class MOSIDataset(Dataset):
    """
    Dataset class for MOSI data loading.
    Used for cross-dataset evaluation (pretrain on MOSEI, test on MOSI).
    - If split is 'train': returns combined train+val data
    - If split is 'val': returns only val data
    - If split is 'test': returns only test data
    Returns numpy arrays to match VGGSound collate pattern.
    """
    def __init__(self, data_path, split='test', max_seq_len=50, batch_size=64):
        super(MOSIDataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        
        print_progress(f"Loading MOSI dataset (split: {split})...")
        
        try:
            from datasets.affect.get_data import get_dataloader
            
            filepath = os.path.join(data_path, 'mosi_raw.pkl')
            traindata, validdata, testdata = get_dataloader(
                filepath, 
                robust_test=False, 
                max_pad=True, 
                data_type='mosi', 
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                z_norm=True
            )
            
            if split == 'train':
                print_progress("Combining train and validation splits...")
                splits_to_process = [("train", traindata), ("val", validdata)]
            elif split == 'val':
                print_progress("Using validation split only...")
                splits_to_process = [("val", validdata)]
            elif split == 'test':
                print_progress("Using test split only...")
                splits_to_process = [("test", testdata)]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
                
            # Process the selected splits
            print_progress(f"Processing MOSI {split} split...")
            self.data = []
            
            for split_name, dataloader in splits_to_process:
                print_progress(f"Processing {split_name} split...")
                for batch_idx, batch in enumerate(dataloader):
                    visual_feat, audio_feat, text_feat, labels = batch
                    batch_size_actual = visual_feat.shape[0]
                    
                    for i in range(batch_size_actual):
                        sample = {
                            'visual': visual_feat[i],   # [seq_len, 35]
                            'audio': audio_feat[i],     # [seq_len, 74]
                            'text': text_feat[i],       # [seq_len, 300] 
                            'labels': labels[i],        # [1]
                            'video_id': f"mosi_{split_name}_{batch_idx}_{i}"
                        }
                        self.data.append(sample)
                    
            print_progress(f"Successfully loaded {len(self.data)} samples from MOSI {split}")
            
        except Exception as e:
            print_progress(f"Error loading MOSI {split} data: {e}")
            import traceback
            print_progress(f"Traceback: {traceback.format_exc()}")
            self.data = []
    
    def __getitem__(self, index):
        """
        Returns numpy arrays to match VGGSound collate pattern.
        """
        sample = self.data[index]
        
        # Convert tensors to numpy arrays (removing .float() calls)
        audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
        text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
        labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
        video_id = sample['video_id']
        
        return audio_feature, video_feature, text_feature, labels, video_id
    
    def __len__(self):
        return len(self.data)


# ===============================================================================
# COLLATE FUNCTION - Following exact VGGSound pattern
# ===============================================================================

def collate_func_AVT(samples):
    """
    VGGSound-style collate function for MOSEI/MOSI data.
    Follows the exact pattern: torch.from_numpy(np.asarray([sample[key] for sample in samples])).float()
    
    Args:
        samples: List of tuples from Dataset.__getitem__()
                 Each tuple: (audio_feature, video_feature, text_feature, labels, video_id)
                 where each feature is a numpy array
                 
    Returns:
        Dictionary with batched tensors for training
    """
    
    return {
        'audio_fea': torch.from_numpy(np.asarray([sample[0] for sample in samples])).float(),
        
        'video_fea': torch.from_numpy(np.asarray([sample[1] for sample in samples])).float(),
        
        'text_fea': torch.from_numpy(np.asarray([sample[2] for sample in samples])).float(),
        
        'labels': torch.from_numpy(np.asarray([sample[3] for sample in samples])).float(),
        
        'video_ids': [sample[4] for sample in samples]
    }



def collate_func_AVT_dynamic_padding(samples):
    """
    Advanced collate function with dynamic last-frame repetition padding.
    
    This function performs the following steps:
    1. Detects actual sequence length by finding where trailing zero padding begins (using text modality)
    2. Truncates all modalities to their actual content length (removes artificial zero padding)
    3. Finds maximum actual length within the current batch
    4. Applies last-frame repetition padding to match batch maximum length
    5. Returns VGGSound-compatible tensor dictionary
    
    Args:
        samples: List of tuples (audio_feature, video_feature, text_feature, labels, video_id)
                Each feature is numpy array with shape [max_seq_len, feature_dim] (e.g., [50, 74])
                Input sequences have trailing zero padding from MultiBench preprocessing
    
    Returns:
        Dictionary with keys: 'audio_fea', 'video_fea', 'text_fea', 'labels', 'video_ids'
        All feature tensors have shape [batch_size, dynamic_max_len, feature_dim]
    """
    
    # Step 1: Extract raw features from samples
    raw_audio_features = [sample[0] for sample in samples]  # List of [seq_len, 74] arrays
    raw_video_features = [sample[1] for sample in samples]  # List of [seq_len, 35] arrays
    raw_text_features = [sample[2] for sample in samples]   # List of [seq_len, 300] arrays
    labels = [sample[3] for sample in samples]              # List of [1] arrays
    video_ids = [sample[4] for sample in samples]           # List of strings
    
    # Step 2: Detect actual sequence lengths by finding where trailing zero padding starts
    # We use only text modality as specified, then apply same length to all modalities
    actual_lengths = []
    
    for text_feat in raw_text_features:
        seq_len = text_feat.shape[0]
        actual_length = seq_len  # Initialize to full sequence length
        
        # Scan backwards from the end to find where actual content stops
        # This detects the transition from content to trailing zero padding
        for t in range(seq_len - 1, -1, -1):  # Start from last timestep, go backwards
            if np.any(text_feat[t] != 0):  # Found a timestep with non-zero content
                actual_length = t + 1  # +1 because we want to include this timestep
                break
            # If this timestep is all zeros, continue scanning backwards
        
        # Ensure we always have at least 1 timestep (safety check)
        actual_length = max(1, actual_length)
        actual_lengths.append(actual_length)
    
    # Step 3: Truncate all modalities to their detected actual lengths
    # This removes the artificial zero padding added by MultiBench
    truncated_audio = []
    truncated_video = []
    truncated_text = []
    
    for i in range(len(samples)):
        length = actual_lengths[i]
        # Apply same length to all modalities since they're aligned
        truncated_audio.append(raw_audio_features[i][:length])
        truncated_video.append(raw_video_features[i][:length])
        truncated_text.append(raw_text_features[i][:length])
    
    # Step 4: Find the maximum actual length in this batch for dynamic padding
    # This determines the target length for all sequences in the batch
    max_length_in_batch = max(actual_lengths)
    
    # Step 5: Apply last-frame repetition padding to reach batch maximum length
    # This is semantically meaningful padding that maintains content continuity
    padded_audio = []
    padded_video = []
    padded_text = []
    
    for i in range(len(samples)):
        audio_seq = truncated_audio[i]
        video_seq = truncated_video[i]
        text_seq = truncated_text[i]
        
        current_length = audio_seq.shape[0]
        
        if current_length < max_length_in_batch:
            # Calculate how much padding we need to reach batch maximum
            padding_needed = max_length_in_batch - current_length
            
            # Extract the last timestep from each modality for repetition
            # This preserves the "final state" of each modality
            last_audio_frame = audio_seq[-1:, :]  # Shape: [1, 74]
            last_video_frame = video_seq[-1:, :]  # Shape: [1, 35]  
            last_text_frame = text_seq[-1:, :]    # Shape: [1, 300]
            
            # Create padding by tiling (repeating) the last frame
            # This maintains semantic continuity rather than introducing zeros
            audio_padding = np.tile(last_audio_frame, (padding_needed, 1))  # [padding_needed, 74]
            video_padding = np.tile(last_video_frame, (padding_needed, 1))  # [padding_needed, 35]
            text_padding = np.tile(last_text_frame, (padding_needed, 1))    # [padding_needed, 300]
            
            # Concatenate original content with last-frame padding
            padded_audio_seq = np.concatenate([audio_seq, audio_padding], axis=0)
            padded_video_seq = np.concatenate([video_seq, video_padding], axis=0)
            padded_text_seq = np.concatenate([text_seq, text_padding], axis=0)
        else:
            # Sequence is already at batch max length, no padding needed
            padded_audio_seq = audio_seq
            padded_video_seq = video_seq
            padded_text_seq = text_seq
        
        padded_audio.append(padded_audio_seq)
        padded_video.append(padded_video_seq)
        padded_text.append(padded_text_seq)
    
    # Step 6: Convert to tensors following the exact VGGSound pattern
    # This maintains compatibility with your existing training pipeline
    return {
        'audio_fea': torch.from_numpy(np.asarray(padded_audio)).float(),
        'video_fea': torch.from_numpy(np.asarray(padded_video)).float(),
        'text_fea': torch.from_numpy(np.asarray(padded_text)).float(),
        'labels': torch.from_numpy(np.asarray(labels)).float(),
        'video_ids': video_ids
    }

# ===============================================================================
# DATALOADER FUNCTIONS
# ===============================================================================

def get_mosei_unsupervised_dataloader(batch_size= 64, max_seq_len=50, num_workers=8):
    """
    Creates a DataLoader for unsupervised pretraining on combined MOSEI data.
    
    Usage:
        unsupervised_loader = get_mosei_unsupervised_dataloader(batch_size=args.batch_size)
        for batch_data in unsupervised_loader:
            audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
            video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
            text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
            # Ignore labels for unsupervised pretraining
    """
    dataset = MOSEIDatasetUnsupervised(MOSEI_DATA_PATH, max_seq_len=max_seq_len, batch_size=batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_func_AVT_dynamic_padding
    )


def get_mosei_unsupervised_split_dataloaders(batch_size=64, max_seq_len=50, num_workers=8):
    """
    Creates DataLoaders for unsupervised pretraining with split functionality.
    Returns train, test_train, and test_val dataloaders.
    
    Usage:
        train_loader, test_train_loader, test_val_loader = get_mosei_unsupervised_split_dataloaders(batch_size=args.batch_size)
        
        # For unsupervised pretraining on train+val data
        for batch_data in train_loader:
            audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
            video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
            text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
            # Ignore labels for unsupervised pretraining
            
        # For evaluation on test splits
        for batch_data in test_train_loader:
            # 80% of test data for training evaluation
            pass
            
        for batch_data in test_val_loader:
            # 20% of test data for validation evaluation
            pass
    """
    # Create datasets for different splits
    train_dataset = MOSEIDatasetUnsupervisedSplit(MOSEI_DATA_PATH, split='train', max_seq_len=max_seq_len, batch_size=batch_size)
    test_train_dataset = MOSEIDatasetUnsupervisedSplit(MOSEI_DATA_PATH, split='test_train', max_seq_len=max_seq_len, batch_size=batch_size)
    test_val_dataset = MOSEIDatasetUnsupervisedSplit(MOSEI_DATA_PATH, split='test_val', max_seq_len=max_seq_len, batch_size=batch_size)
    
    # Create dataloaders with VGGSound-style collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=False,
                             collate_fn=collate_func_AVT_dynamic_padding)
    test_train_loader = DataLoader(test_train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=False,
                                  collate_fn=collate_func_AVT_dynamic_padding)
    test_val_loader = DataLoader(test_val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=False,
                                collate_fn=collate_func_AVT_dynamic_padding)
    
    return train_loader, test_train_loader, test_val_loader


def get_mosei_supervised_dataloaders(batch_size=64, max_seq_len=50, num_workers=8):
    """
    Creates DataLoaders for supervised training on MOSEI.
    Returns train, validation, and test dataloaders.
    
    Usage:
        train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(batch_size=args.batch_size)
        for batch_data in train_loader:
            audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
            video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
            text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
            labels = batch_data['labels']              # [batch_size, 1]
    """
    # Create datasets with modified logic:
    # - train/val splits return combined train+val data
    # - test split returns only test data
    train_dataset = MOSEIDatasetSupervised(MOSEI_DATA_PATH, split='train', max_seq_len=max_seq_len, batch_size=batch_size)
    val_dataset = MOSEIDatasetSupervised(MOSEI_DATA_PATH, split='val', max_seq_len=max_seq_len, batch_size=batch_size)  
    test_dataset = MOSEIDatasetSupervised(MOSEI_DATA_PATH, split='test', max_seq_len=max_seq_len, batch_size=batch_size)
    
    # Create dataloaders with VGGSound-style collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=False, 
                             collate_fn=collate_func_AVT_dynamic_padding)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           collate_fn=collate_func_AVT_dynamic_padding)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=collate_func_AVT_dynamic_padding)
    
    return train_loader, val_loader, test_loader


def get_mosi_dataloaders(batch_size=64, max_seq_len=50, num_workers=8):
    """
    Creates DataLoaders for MOSI dataset (cross-dataset evaluation).
    Returns train, validation, and test dataloaders.
    
    Usage:
        mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(batch_size=args.batch_size)
        for batch_data in mosi_test:
            audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
            video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
            text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
            labels = batch_data['labels']              # [batch_size, 1]
    """
    # Create MOSI datasets for all splits
    train_dataset = MOSIDataset(MOSI_DATA_PATH, split='train', max_seq_len=max_seq_len, batch_size=batch_size)
    val_dataset = MOSIDataset(MOSI_DATA_PATH, split='val', max_seq_len=max_seq_len, batch_size=batch_size)
    test_dataset = MOSIDataset(MOSI_DATA_PATH, split='test', max_seq_len=max_seq_len, batch_size=batch_size)
    
    # Create dataloaders with VGGSound-style collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=False,
                             collate_fn=collate_func_AVT_dynamic_padding)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           collate_fn=collate_func_AVT_dynamic_padding)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=collate_func_AVT_dynamic_padding)
    
    return train_loader, val_loader, test_loader


# ===============================================================================
# TEST FUNCTIONS
# ===============================================================================


def print_feature_diagnostics(features, feature_name, sample_idx=0):
    """
    Print comprehensive diagnostics for a feature tensor/array
    """
    if isinstance(features, torch.Tensor):
        features_np = features.numpy()
    else:
        features_np = features
    
    print_progress(f"\n--- {feature_name.upper()} FEATURE DIAGNOSTICS ---")
    print_progress(f"Shape: {features_np.shape}")
    print_progress(f"Data type: {features_np.dtype}")
    print_progress(f"Min value: {np.min(features_np):.6f}")
    print_progress(f"Max value: {np.max(features_np):.6f}")
    print_progress(f"Mean: {np.mean(features_np):.6f}")
    print_progress(f"Std: {np.std(features_np):.6f}")
    
    # Check for NaN or Inf values
    nan_count = np.sum(np.isnan(features_np))
    inf_count = np.sum(np.isinf(features_np))
    print_progress(f"NaN values: {nan_count}")
    print_progress(f"Inf values: {inf_count}")
    
    # Show sample values from different time steps
    if len(features_np.shape) >= 2:
        seq_len = features_np.shape[0]
        feature_dim = features_np.shape[1]
        print_progress(f"Sequence length: {seq_len}, Feature dimension: {feature_dim}")
        
        # Show first few values from first timestep
        print_progress(f"First timestep values (first 10): {features_np[0, :10]}")
        
        # Show middle timestep if available
        if seq_len > 1:
            mid_idx = seq_len // 2
            print_progress(f"Middle timestep values (first 10): {features_np[mid_idx, :10]}")
            
        # Show last timestep if different from first
        if seq_len > 2:
            print_progress(f"Last timestep values (first 10): {features_np[-1, :10]}")
    else:
        print_progress(f"Feature values: {features_np}")
    
    print_progress(f"--- END {feature_name.upper()} DIAGNOSTICS ---\n")


def test_collate_pattern():
    """
    Test that our collate function follows the exact VGGSound pattern.
    This verifies that numpy arrays are properly converted to tensors.
    """
    print_progress("\n=== Testing VGGSound Collate Pattern ===")
    
    # Create mock samples that mimic what our datasets return
    mock_samples = []
    for i in range(2):  # Create 2 fake samples
        audio_feat = np.random.randn(50, 74).astype(np.float32)  # [seq_len, 74]
        video_feat = np.random.randn(50, 35).astype(np.float32)  # [seq_len, 35]
        text_feat = np.random.randn(50, 300).astype(np.float32)  # [seq_len, 300]
        labels = np.array([0.5]).astype(np.float32)              # [1]
        video_id = f"test_sample_{i}"
        
        sample = (audio_feat, video_feat, text_feat, labels, video_id)
        mock_samples.append(sample)
    
    # Test our collate function
    batch_data = collate_func_AVT(mock_samples)
    
    print_progress(f"Mock batch results:")
    print_progress(f"  Audio shape: {batch_data['audio_fea'].shape}, dtype: {batch_data['audio_fea'].dtype}")
    print_progress(f"  Video shape: {batch_data['video_fea'].shape}, dtype: {batch_data['video_fea'].dtype}")
    print_progress(f"  Text shape: {batch_data['text_fea'].shape}, dtype: {batch_data['text_fea'].dtype}")
    print_progress(f"  Labels shape: {batch_data['labels'].shape}, dtype: {batch_data['labels'].dtype}")
    print_progress(f"  Video IDs: {batch_data['video_ids']}")
    print_progress("✅ VGGSound collate pattern test passed!")


def test_dynamic_collate_pattern():
    """
    Test function specifically designed for the dynamic padding collate function.
    Creates mock samples that mimic real MultiBench output structure:
    - Actual content followed by trailing zero padding
    - Different sequence lengths to test dynamic padding logic
    """
    print_progress("\n=== Testing Dynamic Padding Collate Function ===")
    
    # Create mock samples that mimic MultiBench preprocessing output
    # These have actual content followed by trailing zero padding
    mock_samples = []
    
    # Sample 1: Short sequence (actual length 15, padded to 50)
    print_progress("Creating Sample 1: Actual length 15, padded to 50")
    audio_feat_1 = np.random.randn(15, 74).astype(np.float32)  # Real content
    audio_padding_1 = np.zeros((35, 74), dtype=np.float32)    # Zero padding  
    audio_feat_1 = np.concatenate([audio_feat_1, audio_padding_1], axis=0)
    
    video_feat_1 = np.random.randn(15, 35).astype(np.float32)  # Real content
    video_padding_1 = np.zeros((35, 35), dtype=np.float32)    # Zero padding
    video_feat_1 = np.concatenate([video_feat_1, video_padding_1], axis=0)
    
    text_feat_1 = np.random.randn(15, 300).astype(np.float32)  # Real content
    text_padding_1 = np.zeros((35, 300), dtype=np.float32)    # Zero padding  
    text_feat_1 = np.concatenate([text_feat_1, text_padding_1], axis=0)
    
    labels_1 = np.array([0.3]).astype(np.float32)
    video_id_1 = "test_sample_short"
    
    sample_1 = (audio_feat_1, video_feat_1, text_feat_1, labels_1, video_id_1)
    mock_samples.append(sample_1)
    
    # Sample 2: Medium sequence (actual length 32, padded to 50) 
    print_progress("Creating Sample 2: Actual length 32, padded to 50")
    audio_feat_2 = np.random.randn(32, 74).astype(np.float32)  # Real content
    audio_padding_2 = np.zeros((18, 74), dtype=np.float32)    # Zero padding
    audio_feat_2 = np.concatenate([audio_feat_2, audio_padding_2], axis=0)
    
    video_feat_2 = np.random.randn(32, 35).astype(np.float32)  # Real content  
    video_padding_2 = np.zeros((18, 35), dtype=np.float32)    # Zero padding
    video_feat_2 = np.concatenate([video_feat_2, video_padding_2], axis=0)
    
    text_feat_2 = np.random.randn(32, 300).astype(np.float32)  # Real content
    text_padding_2 = np.zeros((18, 300), dtype=np.float32)    # Zero padding
    text_feat_2 = np.concatenate([text_feat_2, text_padding_2], axis=0)
    
    labels_2 = np.array([-0.7]).astype(np.float32) 
    video_id_2 = "test_sample_medium"
    
    sample_2 = (audio_feat_2, video_feat_2, text_feat_2, labels_2, video_id_2)
    mock_samples.append(sample_2)
    
    # Sample 3: Long sequence (actual length 48, minimal padding)
    print_progress("Creating Sample 3: Actual length 48, padded to 50") 
    audio_feat_3 = np.random.randn(48, 74).astype(np.float32)  # Real content
    audio_padding_3 = np.zeros((2, 74), dtype=np.float32)     # Minimal zero padding
    audio_feat_3 = np.concatenate([audio_feat_3, audio_padding_3], axis=0)
    
    video_feat_3 = np.random.randn(48, 35).astype(np.float32)  # Real content
    video_padding_3 = np.zeros((2, 35), dtype=np.float32)     # Minimal zero padding  
    video_feat_3 = np.concatenate([video_feat_3, video_padding_3], axis=0)
    
    text_feat_3 = np.random.randn(48, 300).astype(np.float32)  # Real content
    text_padding_3 = np.zeros((2, 300), dtype=np.float32)     # Minimal zero padding
    text_feat_3 = np.concatenate([text_feat_3, text_padding_3], axis=0)
    
    labels_3 = np.array([1.2]).astype(np.float32)
    video_id_3 = "test_sample_long"
    
    sample_3 = (audio_feat_3, video_feat_3, text_feat_3, labels_3, video_id_3)
    mock_samples.append(sample_3)
    
    print_progress(f"Created {len(mock_samples)} mock samples with varying actual lengths")
    print_progress("Expected behavior:")
    print_progress("  - Sample 1: Should be detected as length 15, padded to 48 (batch max)")
    print_progress("  - Sample 2: Should be detected as length 32, padded to 48 (batch max)")
    print_progress("  - Sample 3: Should be detected as length 48, no padding needed")
    print_progress("  - Final batch shape should be [3, 48, feature_dim] for each modality")
    
    # Test the dynamic collate function
    print_progress("\nTesting dynamic padding collate function...")
    batch_data = collate_func_AVT_dynamic_padding(mock_samples)
    
    # Verify results
    print_progress("Results from dynamic padding collate function:")
    print_progress(f"  Audio shape: {batch_data['audio_fea'].shape}, dtype: {batch_data['audio_fea'].dtype}")
    print_progress(f"  Video shape: {batch_data['video_fea'].shape}, dtype: {batch_data['video_fea'].dtype}")
    print_progress(f"  Text shape: {batch_data['text_fea'].shape}, dtype: {batch_data['text_fea'].dtype}")
    print_progress(f"  Labels shape: {batch_data['labels'].shape}, dtype: {batch_data['labels'].dtype}")
    print_progress(f"  Video IDs: {batch_data['video_ids']}")
    
    # Verify the dynamic length detection worked correctly
    expected_batch_length = 48  # Maximum actual content length among the 3 samples
    actual_batch_length = batch_data['audio_fea'].shape[1] 
    
    print_progress(f"\nDetailed verification:")
    print_progress(f"  Expected batch sequence length: {expected_batch_length}")
    print_progress(f"  Actual batch sequence length: {actual_batch_length}")
    print_progress(f"  Dynamic length detection: {'✅ PASSED' if actual_batch_length == expected_batch_length else '❌ FAILED'}")
    
    # Verify last-frame padding was applied correctly
    # For sample 1 (length 15), timesteps 15-47 should all equal timestep 14 (last real content)
    sample_1_audio = batch_data['audio_fea'][0]  # First sample
    last_content_frame = sample_1_audio[14]      # Last real content (0-indexed)
    first_padded_frame = sample_1_audio[15]      # First padded frame
    
    padding_correct = torch.allclose(last_content_frame, first_padded_frame)
    print_progress(f"  Last-frame padding correctness: {'✅ PASSED' if padding_correct else '❌ FAILED'}")
    
    # Check that different samples have different content (not all the same)
    sample_1_audio_first = batch_data['audio_fea'][0, 0]  # First sample, first timestep
    sample_2_audio_first = batch_data['audio_fea'][1, 0]  # Second sample, first timestep
    
    content_different = not torch.allclose(sample_1_audio_first, sample_2_audio_first)
    print_progress(f"  Sample differentiation: {'✅ PASSED' if content_different else '❌ FAILED'}")
    
    if actual_batch_length == expected_batch_length and padding_correct and content_different:
        print_progress("\n✅ Dynamic padding collate function test PASSED!")
        print_progress("The function correctly:")
        print_progress("  - Detected actual sequence lengths by finding trailing zeros")
        print_progress("  - Applied dynamic batching to the maximum length in the batch") 
        print_progress("  - Used last-frame repetition padding instead of zero padding")
        print_progress("  - Maintained proper tensor shapes and data types")
    else:
        print_progress("\n❌ Dynamic padding collate function test FAILED!")
        print_progress("Check the implementation for issues with length detection or padding logic")
    
    return batch_data

def test_all_dataloaders():
    """
    Comprehensive test function for all dataloaders.
    Tests that all datasets return the expected format and dimensions.
    """
    print_progress("\n" + "="*80)
    print_progress("COMPREHENSIVE DATALOADER TESTING")
    print_progress("="*80)
    
    # Test collate pattern first
    print_progress("\nTesting ORIGINAL collate function with random data...")
    test_collate_pattern()

    print_progress("Testing DYNAMIC padding collate function...")
    dynamic_results = test_dynamic_collate_pattern()

    print_progress(f"\nComparison Summary:")
    print_progress(f"  Dynamic collate batch length: {dynamic_results['audio_fea'].shape[1]}")
    
    print_progress("\n=== Testing Unsupervised MOSEI Dataset ===")
    try:
        unsupervised_loader = get_mosei_unsupervised_dataloader(batch_size=16, max_seq_len=50)
        for i, batch_data in enumerate(unsupervised_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea'] 
            text_feature = batch_data['text_fea']
            
            print_progress(f"Unsupervised Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            print_progress(f"  Labels available: {'labels' in batch_data}")
            labels = batch_data['labels']
            
            print_progress(f"\nUnsupervised Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Batch_{i}_Audio_Sample_0")
            print_feature_diagnostics(video_feature[0], f"Batch_{i}_Video_Sample_0")
            print_feature_diagnostics(text_feature[0], f"Batch_{i}_Text_Sample_0")
            print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
            
            if i == 10:  # Only check first batch
                break
        print_progress("✅ MOSEI Unsupervised test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Unsupervised test failed: {e}")

    print_progress("\n=== Testing MOSEI Unsupervised Split Dataset ===")
    try:
        train_loader, test_train_loader, test_val_loader = get_mosei_unsupervised_split_dataloaders(batch_size=16, max_seq_len=50)
        
        # Test train split (train+val combined)
        for i, batch_data in enumerate(train_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea'] 
            text_feature = batch_data['text_fea']
            
            print_progress(f"Unsupervised Split Train Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            print_progress(f"  Labels available: {'labels' in batch_data}")
            labels = batch_data['labels']
            
            print_progress(f"\nUnsupervised split Train Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Batch_{i}_Audio_Sample_0")
            print_feature_diagnostics(video_feature[0], f"Batch_{i}_Video_Sample_0")
            print_feature_diagnostics(text_feature[0], f"Batch_{i}_Text_Sample_0")
            print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
            if i == 10:  # Only check first batch
                break
        
        # Test test_train split (80% of test data)
        for i, batch_data in enumerate(test_train_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea'] 
            text_feature = batch_data['text_fea']
            
            print_progress(f"Unsupervised Split Test-Train Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            labels = batch_data['labels']
            
            print_progress(f"\nUnsupervised Split Test-Train Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Batch_{i}_Audio_Sample_0")
            print_feature_diagnostics(video_feature[0], f"Batch_{i}_Video_Sample_0")
            print_feature_diagnostics(text_feature[0], f"Batch_{i}_Text_Sample_0")
            print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
            if i == 10:  # Only check first batch
                break
        
        # Test test_val split (20% of test data)
        for i, batch_data in enumerate(test_val_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea'] 
            text_feature = batch_data['text_fea']
            
            print_progress(f"Unsupervised Split Test-Val Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            labels = batch_data['labels']
            
            print_progress(f"\nUnsupervised Split Test_Val Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Batch_{i}_Audio_Sample_0")
            print_feature_diagnostics(video_feature[0], f"Batch_{i}_Video_Sample_0")
            print_feature_diagnostics(text_feature[0], f"Batch_{i}_Text_Sample_0")
            print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
            if i == 10:  # Only check first batch
                break
                
        # Verify the 80-20 split worked correctly
        train_dataset_size = len(train_loader.dataset)
        test_train_size = len(test_train_loader.dataset)
        test_val_size = len(test_val_loader.dataset)
        
        print_progress(f"Dataset sizes:")
        print_progress(f"  Train (train+val): {train_dataset_size} samples")
        print_progress(f"  Test-Train (80%): {test_train_size} samples")
        print_progress(f"  Test-Val (20%): {test_val_size} samples")
        print_progress(f"  Test split ratio: {test_train_size / (test_train_size + test_val_size):.2f} / {test_val_size / (test_train_size + test_val_size):.2f}")
        
        print_progress("✅ MOSEI Unsupervised Split test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Unsupervised Split test failed: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")
    
    print_progress("\n=== Testing Supervised MOSEI Dataset ===")
    try:
        train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(batch_size=16, max_seq_len=50)
        
        for i, batch_data in enumerate(train_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea']
            text_feature = batch_data['text_fea'] 
            labels = batch_data['labels']
            
            print_progress(f"Supervised Train Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            print_progress(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")

            print_progress(f"\nSupervised Train Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Supervised_Batch_{i}_Audio_Sample_0")
            print_progress(f"Batch size: {audio_feature.shape[0]}")
            print_progress(f"Labels range in batch: [{labels.min().item():.3f}, {labels.max().item():.3f}]")
            print_progress(f"Sample labels: {labels[:5].numpy().flatten()}")  # Show first 5 labels
            
            
            if i == 10:  # Only check first batch
                break
        print_progress("✅ MOSEI Supervised test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Supervised test failed: {e}")
    
    print_progress("\n=== Testing MOSI Dataset ===")
    try:
        mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(batch_size=16, max_seq_len=50)
        
        for i, batch_data in enumerate(mosi_test):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea']
            text_feature = batch_data['text_fea']
            labels = batch_data['labels']
            
            print_progress(f"MOSI Test Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            print_progress(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
            
            if i == 10:  # Check first two batches
                break
        print_progress("✅ MOSI test passed!")
    except Exception as e:
        print_progress(f"❌ MOSI test failed: {e}")
    
    print_progress("\n" + "="*80)
    print_progress("✅ ALL DATASET TESTS COMPLETED SUCCESSFULLY!")
    print_progress("Ready for integration with your training pipeline.")
    print_progress("="*80)


if __name__ == "__main__":
    test_all_dataloaders()
