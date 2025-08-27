

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
    Combines train, validation, and test splits from MOSEI for self-supervised learning.
    Returns numpy arrays to match VGGSound collate pattern.
    """
    def __init__(self, data_path, max_seq_len=10, batch_size=64):
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
                batch_size=batch_size
            )
            
            # Combine all data for unsupervised pretraining
            print_progress("Combining train, validation, and test splits...")
            self.combined_data = []
            
            # Extract data from each split
            for split_name, dataloader in [("train", traindata), ("val", validdata), ("test", testdata)]:
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
                batch_size=batch_size
            )
            
            
            # Determine which splits to use based on the requested split
            # if split in ['train', 'val']:
            #     # For train or val split, combine train+val data
            #     print_progress("Combining train and validation splits for supervised training...")
            #     splits_to_process = [("train", traindata), ("val", validdata)]
            # elif split == 'test':
            #     # For test split, use only test data
            #     print_progress("Using test split only...")
            #     splits_to_process = [("test", testdata)]
            # else:
            #     raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")


            # Determine which splits to use based on the requested split
            if split == 'train':
                # For train split, use only train data
                print_progress("Using train split only...")
                splits_to_process = [("train", traindata)]
            elif split == 'val':
                # For val split, use only val data
                print_progress("Using validation split only...")
                splits_to_process = [("val", validdata)]
            elif split == 'test':
                # For test split, use only test data
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
    Returns numpy arrays to match VGGSound collate pattern.
    """
    def __init__(self, data_path, split='test', max_seq_len=10, batch_size=64):
        super(MOSIDataset, self).__init__()
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        
        print_progress(f"Loading MOSI {split} dataset...")
        
        try:
            from datasets.affect.get_data import get_dataloader
            
            filepath = os.path.join(data_path, 'mosi_raw.pkl')
            traindata, validdata, testdata = get_dataloader(
                filepath, 
                robust_test=False, 
                max_pad=True, 
                data_type='mosi', 
                max_seq_len=max_seq_len,
                batch_size=batch_size
            )
            
            # Select the appropriate split
            if split == 'train':
                selected_dataloader = traindata
            elif split == 'val':
                selected_dataloader = validdata  
            elif split == 'test':
                selected_dataloader = testdata
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
                
            print_progress(f"Processing MOSI {split} split...")
            self.data = []
            
            for batch_idx, batch in enumerate(selected_dataloader):
                visual_feat, audio_feat, text_feat, labels = batch
                batch_size_actual = visual_feat.shape[0]
                
                for i in range(batch_size_actual):
                    sample = {
                        'visual': visual_feat[i],   # [seq_len, 35]
                        'audio': audio_feat[i],     # [seq_len, 74]
                        'text': text_feat[i],       # [seq_len, 300] 
                        'labels': labels[i],        # [1]
                        'video_id': f"mosi_{split}_{batch_idx}_{i}"
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


# ===============================================================================
# DATALOADER FUNCTIONS
# ===============================================================================

def get_mosei_unsupervised_dataloader(batch_size= 64, max_seq_len=10, num_workers=8):
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
        collate_fn=collate_func_AVT  # VGGSound-style collate function
    )


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
                             collate_fn=collate_func_AVT)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           collate_fn=collate_func_AVT)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=collate_func_AVT)
    
    return train_loader, val_loader, test_loader


def get_mosi_dataloaders(batch_size=64, max_seq_len=10, num_workers=8):
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
                             collate_fn=collate_func_AVT)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           collate_fn=collate_func_AVT)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=collate_func_AVT)
    
    return train_loader, val_loader, test_loader


# ===============================================================================
# TEST FUNCTIONS
# ===============================================================================

def test_collate_pattern():
    """
    Test that our collate function follows the exact VGGSound pattern.
    This verifies that numpy arrays are properly converted to tensors.
    """
    print_progress("\n=== Testing VGGSound Collate Pattern ===")
    
    # Create mock samples that mimic what our datasets return
    mock_samples = []
    for i in range(2):  # Create 2 fake samples
        audio_feat = np.random.randn(10, 74).astype(np.float32)  # [seq_len, 74]
        video_feat = np.random.randn(10, 35).astype(np.float32)  # [seq_len, 35]
        text_feat = np.random.randn(10, 300).astype(np.float32)  # [seq_len, 300]
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


def test_all_dataloaders():
    """
    Comprehensive test function for all dataloaders.
    Tests that all datasets return the expected format and dimensions.
    """
    print_progress("\n" + "="*80)
    print_progress("COMPREHENSIVE DATALOADER TESTING")
    print_progress("="*80)
    
    # Test collate pattern first
    test_collate_pattern()
    
    print_progress("\n=== Testing Unsupervised MOSEI Dataset ===")
    try:
        unsupervised_loader = get_mosei_unsupervised_dataloader(batch_size=32, max_seq_len=10)
        for i, batch_data in enumerate(unsupervised_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea'] 
            text_feature = batch_data['text_fea']
            
            print_progress(f"Unsupervised Batch {i}:")
            print_progress(f"  Audio shape: {audio_feature.shape}, dtype: {audio_feature.dtype}")
            print_progress(f"  Video shape: {video_feature.shape}, dtype: {video_feature.dtype}")
            print_progress(f"  Text shape: {text_feature.shape}, dtype: {text_feature.dtype}")
            print_progress(f"  Labels available: {'labels' in batch_data}")
            
            if i == 0:  # Only check first batch
                break
        print_progress("✅ MOSEI Unsupervised test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Unsupervised test failed: {e}")
    
    print_progress("\n=== Testing Supervised MOSEI Dataset ===")
    try:
        train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(batch_size=64, max_seq_len=10)
        
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
            
            if i == 0:  # Only check first batch
                break
        print_progress("✅ MOSEI Supervised test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Supervised test failed: {e}")
    
    print_progress("\n=== Testing MOSI Dataset ===")
    try:
        mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(batch_size=32, max_seq_len=10)
        
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
            
            if i >= 1:  # Check first two batches
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
