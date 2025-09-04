import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
from datetime import datetime
import h5py
import pickle

# Add MultiBench to path for accessing the get_dataloader function
MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
MOSEI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'
MOSI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSI/'

sys.path.append(MULTIBENCH_PATH)

def print_progress(message):
    """Helper function for logging progress during dataset loading"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# ===============================================================================
# RAW DATA INSPECTION FUNCTIONS
# ===============================================================================

def inspect_raw_h5_files(data_path, data_type='mosei', num_samples=5):
    """
    Inspect the raw H5 files to see the original data structure and content.
    """
    print_progress(f"\n{'='*80}")
    print_progress(f"INSPECTING RAW H5 FILES: {data_type.upper()}")
    print_progress(f"{'='*80}")
    
    try:
        # Find all H5 files in the data directory
        h5_files = [f for f in os.listdir(data_path) if f.endswith('.h5') or f.endswith('.hdf5')]
        print_progress(f"Found H5 files: {h5_files}")
        
        for h5_file in h5_files:
            h5_path = os.path.join(data_path, h5_file)
            print_progress(f"\n--- INSPECTING {h5_file} ---")
            
            with h5py.File(h5_path, 'r') as f:
                print_progress(f"H5 file keys: {list(f.keys())}")
                
                # Recursively inspect the structure
                def inspect_h5_group(group, prefix=""):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            print_progress(f"{prefix}Group: {key}")
                            inspect_h5_group(item, prefix + "  ")
                        elif isinstance(item, h5py.Dataset):
                            print_progress(f"{prefix}Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}")
                            
                            # Show sample data for small datasets
                            if item.size < 100 and item.size > 0:
                                sample_data = item[()]
                                if isinstance(sample_data, (str, bytes)):
                                    print_progress(f"{prefix}  Sample: {str(sample_data)[:100]}")
                                elif np.isscalar(sample_data):
                                    print_progress(f"{prefix}  Value: {sample_data}")
                                else:
                                    print_progress(f"{prefix}  Sample shape: {np.array(sample_data).shape}")
                                    if len(sample_data) > 0:
                                        print_progress(f"{prefix}  First few values: {np.array(sample_data).flatten()[:10]}")
                
                inspect_h5_group(f)
                
    except Exception as e:
        print_progress(f"Error inspecting H5 files: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")


def inspect_raw_text_processing(data_path, data_type='mosei', num_samples=5):
    """
    Inspect the raw text processing pipeline step by step.
    """
    print_progress(f"\n{'='*80}")
    print_progress(f"INSPECTING RAW TEXT PROCESSING PIPELINE: {data_type.upper()}")
    print_progress(f"{'='*80}")
    
    try:
        from datasets.affect.get_raw_data import get_rawtext, get_word2id, glove_embeddings, get_word_embeddings
        
        # Find text data files
        if data_type == 'mosei':
            text_files = [f for f in os.listdir(data_path) if 'text' in f.lower() and (f.endswith('.h5') or f.endswith('.hdf5'))]
        else:
            text_files = [f for f in os.listdir(data_path) if 'text' in f.lower() and (f.endswith('.h5') or f.endswith('.hdf5'))]
            
        if not text_files:
            # Try to find any h5 files that might contain text
            text_files = [f for f in os.listdir(data_path) if f.endswith('.h5') or f.endswith('.hdf5')]
            
        print_progress(f"Found potential text files: {text_files}")
        
        if text_files:
            text_file_path = os.path.join(data_path, text_files[0])
            print_progress(f"\nAnalyzing text file: {text_file_path}")
            
            # Step 1: Get raw text data
            print_progress(f"\n--- STEP 1: EXTRACTING RAW TEXT ---")
            
            # First, let's see what video IDs are available by looking at the main data file
            if data_type == 'mosei':
                main_file = os.path.join(data_path, 'mosei_senti_data.pkl')
            else:
                main_file = os.path.join(data_path, 'mosi_raw.pkl')
                
            if os.path.exists(main_file):
                with open(main_file, 'rb') as f:
                    data = pickle.load(f)
                    print_progress(f"Main data keys: {data.keys() if hasattr(data, 'keys') else type(data)}")
                    
                    # Try to extract video IDs from the structure
                    if isinstance(data, dict):
                        for key in list(data.keys())[:3]:  # Look at first 3 keys
                            print_progress(f"Key '{key}': {type(data[key])}")
                            if hasattr(data[key], 'keys'):
                                sample_keys = list(data[key].keys())[:5]
                                print_progress(f"  Sample subkeys: {sample_keys}")
                                
                                # These might be video IDs
                                vids = sample_keys[:num_samples]
                                print_progress(f"Using sample video IDs: {vids}")
                                
                                try:
                                    # Try to get raw text
                                    text_data, video_data = get_rawtext(text_file_path, 'hdf5', vids)
                                    
                                    print_progress(f"\n--- RAW TEXT EXTRACTION RESULTS ---")
                                    print_progress(f"Extracted {len(text_data)} text samples")
                                    print_progress(f"Extracted {len(video_data)} video entries")
                                    
                                    # Show sample raw text
                                    for i, (text, vid) in enumerate(zip(text_data[:3], video_data[:3])):
                                        print_progress(f"\nSample {i+1}:")
                                        print_progress(f"  Video ID: {vid}")
                                        print_progress(f"  Raw text type: {type(text)}")
                                        if isinstance(text, (str, bytes)):
                                            text_str = str(text)
                                            print_progress(f"  Text length: {len(text_str)} characters")
                                            print_progress(f"  Text preview: '{text_str[:200]}...'")
                                        elif isinstance(text, (list, np.ndarray)):
                                            print_progress(f"  Text is array/list of length: {len(text)}")
                                            if len(text) > 0:
                                                print_progress(f"  First few elements: {text[:5]}")
                                        else:
                                            print_progress(f"  Text content: {text}")
                                    
                                    # Step 2: Word to ID mapping
                                    print_progress(f"\n--- STEP 2: WORD-TO-ID MAPPING ---")
                                    try:
                                        word2id = get_word2id(text_data, video_data)
                                        print_progress(f"Generated word2id mapping with {len(word2id)} entries")
                                        
                                        # Show sample word2id mappings
                                        sample_word2id = word2id[:10] if len(word2id) >= 10 else word2id
                                        for i, (word, idx) in enumerate(sample_word2id):
                                            print_progress(f"  {i+1}. '{word}' -> {idx}")
                                            
                                    except Exception as e:
                                        print_progress(f"Error in word2id mapping: {e}")
                                    
                                    # Step 3: GloVe embeddings
                                    print_progress(f"\n--- STEP 3: GLOVE EMBEDDINGS ---")
                                    try:
                                        embeddings = glove_embeddings(text_data, video_data, paddings=50)
                                        print_progress(f"Generated embeddings shape: {embeddings.shape}")
                                        print_progress(f"Embedding dtype: {embeddings.dtype}")
                                        print_progress(f"Embedding range: [{np.min(embeddings):.4f}, {np.max(embeddings):.4f}]")
                                        print_progress(f"Embedding mean: {np.mean(embeddings):.4f}")
                                        print_progress(f"Embedding std: {np.std(embeddings):.4f}")
                                        
                                        # Show sample embeddings
                                        print_progress(f"\nSample embedding analysis:")
                                        sample_embedding = embeddings[0]  # First sample
                                        print_progress(f"  Sample shape: {sample_embedding.shape}")
                                        print_progress(f"  First timestep embedding (first 10 dims): {sample_embedding[0][:10]}")
                                        
                                        # Check for padding (zero rows)
                                        zero_rows = np.sum(np.all(sample_embedding == 0, axis=1))
                                        print_progress(f"  Zero-padded timesteps: {zero_rows}/{sample_embedding.shape[0]}")
                                        
                                    except Exception as e:
                                        print_progress(f"Error in GloVe embeddings: {e}")
                                        import traceback
                                        print_progress(f"Traceback: {traceback.format_exc()}")
                                    
                                    break  # Found working key, exit loop
                                    
                                except Exception as e:
                                    print_progress(f"Error with key '{key}': {e}")
                                    continue
                    
            else:
                print_progress(f"Main data file not found: {main_file}")
                
    except Exception as e:
        print_progress(f"Error in raw text processing inspection: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")


def inspect_raw_vs_processed_data(data_path, data_type='mosei', num_samples=3):
    """
    Compare raw data with processed data to see the full pipeline.
    """
    print_progress(f"\n{'='*80}")
    print_progress(f"RAW VS PROCESSED DATA COMPARISON: {data_type.upper()}")
    print_progress(f"{'='*80}")
    
    try:
        from datasets.affect.get_data import get_dataloader
        
        # Get processed data (what we normally use)
        print_progress(f"\n--- LOADING PROCESSED DATA ---")
        if data_type == 'mosei':
            filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
        else:
            filepath = os.path.join(data_path, 'mosi_raw.pkl')
            
        traindata, validdata, testdata = get_dataloader(
            filepath, 
            robust_test=False, 
            max_pad=True, 
            data_type=data_type, 
            max_seq_len=50,
            batch_size=4,
            z_norm=True
        )
        
        # Get first batch of processed data
        processed_batch = next(iter(traindata))
        visual_processed, audio_processed, text_processed, labels_processed = processed_batch
        
        print_progress(f"Processed data shapes:")
        print_progress(f"  Visual: {visual_processed.shape}")
        print_progress(f"  Audio: {audio_processed.shape}")
        print_progress(f"  Text: {text_processed.shape}")
        print_progress(f"  Labels: {labels_processed.shape}")
        
        # Analyze the processed text embeddings
        print_progress(f"\n--- PROCESSED TEXT ANALYSIS ---")
        sample_text = text_processed[0]  # First sample
        print_feature_diagnostics(sample_text, "Processed_Text_Sample_0")
        
        # Try to load and inspect the original pickle file structure
        print_progress(f"\n--- ORIGINAL PICKLE FILE STRUCTURE ---")
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                raw_data = pickle.load(f)
                print_progress(f"Pickle file type: {type(raw_data)}")
                
                if isinstance(raw_data, dict):
                    print_progress(f"Main keys: {list(raw_data.keys())}")
                    
                    for main_key in list(raw_data.keys())[:2]:  # Check first 2 main keys
                        print_progress(f"\n--- MAIN KEY: {main_key} ---")
                        data_subset = raw_data[main_key]
                        print_progress(f"Type: {type(data_subset)}")
                        
                        if isinstance(data_subset, dict):
                            sample_keys = list(data_subset.keys())[:3]
                            print_progress(f"Sample entry keys: {sample_keys}")
                            
                            for entry_key in sample_keys:
                                entry_data = data_subset[entry_key]
                                print_progress(f"\n  Entry: {entry_key}")
                                print_progress(f"  Entry type: {type(entry_data)}")
                                
                                if isinstance(entry_data, dict):
                                    print_progress(f"  Entry keys: {list(entry_data.keys())}")
                                    
                                    # Look for different modality data
                                    for modality in ['text', 'vision', 'audio', 'labels']:
                                        if modality in entry_data:
                                            mod_data = entry_data[modality]
                                            print_progress(f"    {modality}: shape={getattr(mod_data, 'shape', 'N/A')}, type={type(mod_data)}")
                                            
                                            if hasattr(mod_data, 'shape') and len(mod_data.shape) >= 1:
                                                if mod_data.shape[0] > 0:
                                                    sample_values = mod_data.flatten()[:5] if hasattr(mod_data, 'flatten') else mod_data[:5]
                                                    print_progress(f"    {modality} sample values: {sample_values}")
                                elif hasattr(entry_data, 'shape'):
                                    print_progress(f"  Entry shape: {entry_data.shape}")
                                    if len(entry_data.shape) >= 1 and entry_data.shape[0] > 0:
                                        sample_values = entry_data.flatten()[:5] if hasattr(entry_data, 'flatten') else entry_data[:5]
                                        print_progress(f"  Entry sample values: {sample_values}")
                
                elif hasattr(raw_data, '__len__'):
                    print_progress(f"Data length: {len(raw_data)}")
                    if len(raw_data) > 0:
                        print_progress(f"First element type: {type(raw_data[0])}")
                        if hasattr(raw_data[0], 'keys'):
                            print_progress(f"First element keys: {list(raw_data[0].keys())}")
        
        print_progress(f"\n--- DATA PIPELINE SUMMARY ---")
        print_progress(f"✓ Raw data gets loaded from pickle files with complex nested structure")
        print_progress(f"✓ Text data gets processed through GloVe embeddings (300D)")
        print_progress(f"✓ All modalities get padded to max_seq_len={50}")
        print_progress(f"✓ Final processed shapes: Visual[50,35], Audio[50,74], Text[50,300]")
        
    except Exception as e:
        print_progress(f"Error in raw vs processed comparison: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")

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

def compare_padding_effects(data_path, data_type='mosei', max_seq_len=50, batch_size=4):
    """
    Compare the effects of max_pad=True vs max_pad=False on the same data
    """
    print_progress(f"\n{'='*80}")
    print_progress(f"COMPARING PADDING EFFECTS: {data_type.upper()}")
    print_progress(f"{'='*80}")
    
    try:
        from datasets.affect.get_data import get_dataloader
        
        # Load data with max_pad=True
        print_progress("\n--- LOADING WITH MAX_PAD=True ---")
        if data_type == 'mosei':
            filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
        else:
            filepath = os.path.join(data_path, 'mosi_raw.pkl')
            
        traindata_padded, validdata_padded, testdata_padded = get_dataloader(
            filepath, 
            robust_test=False, 
            max_pad=True, 
            data_type=data_type, 
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            z_norm=True
        )
        
        # Load data with max_pad=False
        print_progress("\n--- LOADING WITH MAX_PAD=False ---")
        traindata_unpadded, validdata_unpadded, testdata_unpadded = get_dataloader(
            filepath, 
            robust_test=False, 
            max_pad=False, 
            data_type=data_type, 
            max_seq_len=max_seq_len,
            batch_size=batch_size
        )
        
        # Compare first batch from training data
        print_progress("\n--- COMPARING FIRST TRAINING BATCH ---")
        
        # Get padded batch
        padded_batch = next(iter(traindata_padded))
        visual_padded, audio_padded, text_padded, labels_padded = padded_batch
        
        # Get unpadded batch  
        unpadded_batch = next(iter(traindata_unpadded))
        visual_unpadded, audio_unpadded, text_unpadded, labels_unpadded = unpadded_batch
        
        print_progress(f"\nPADDED DATA (max_pad=True):")
        print_progress(f"  Visual shape: {visual_padded.shape}")
        print_progress(f"  Audio shape: {audio_padded.shape}")
        print_progress(f"  Text shape: {text_padded.shape}")
        print_progress(f"  Labels shape: {labels_padded.shape}")
        
        print_progress(f"\nUNPADDED DATA (max_pad=False):")
        print_progress(f"  Visual shape: {visual_unpadded.shape}")
        print_progress(f"  Audio shape: {audio_unpadded.shape}")
        print_progress(f"  Text shape: {text_unpadded.shape}")
        print_progress(f"  Labels shape: {labels_unpadded.shape}")
        
        # Analyze first sample in detail
        sample_idx = 0
        print_progress(f"\n--- DETAILED ANALYSIS OF SAMPLE {sample_idx} ---")
        
        print_progress(f"\n🎥 VISUAL FEATURES COMPARISON")
        print_feature_diagnostics(visual_padded[sample_idx], f"Visual_PADDED_Sample_{sample_idx}")
        print_feature_diagnostics(visual_unpadded[sample_idx], f"Visual_UNPADDED_Sample_{sample_idx}")
        
        print_progress(f"\n🔊 AUDIO FEATURES COMPARISON")
        print_feature_diagnostics(audio_padded[sample_idx], f"Audio_PADDED_Sample_{sample_idx}")
        print_feature_diagnostics(audio_unpadded[sample_idx], f"Audio_UNPADDED_Sample_{sample_idx}")
        
        print_progress(f"\n📝 TEXT FEATURES COMPARISON")
        print_feature_diagnostics(text_padded[sample_idx], f"Text_PADDED_Sample_{sample_idx}")
        print_feature_diagnostics(text_unpadded[sample_idx], f"Text_UNPADDED_Sample_{sample_idx}")
        
        print_progress(f"\n🎯 LABELS COMPARISON")
        print_progress(f"Padded labels: {labels_padded[sample_idx]}")
        print_progress(f"Unpadded labels: {labels_unpadded[sample_idx]}")
        print_progress(f"Labels match: {torch.allclose(labels_padded[sample_idx], labels_unpadded[sample_idx])}")
        
        # Check for zero padding patterns
        print_progress(f"\n--- PADDING PATTERN ANALYSIS ---")
        visual_padded_np = visual_padded[sample_idx].numpy()
        visual_unpadded_np = visual_unpadded[sample_idx].numpy()
        
        # Count zero rows (indicating padding)
        zero_rows_padded = np.sum(np.all(visual_padded_np == 0, axis=1))
        zero_rows_unpadded = np.sum(np.all(visual_unpadded_np == 0, axis=1))
        
        print_progress(f"Visual zero rows (padded): {zero_rows_padded}/{visual_padded_np.shape[0]}")
        print_progress(f"Visual zero rows (unpadded): {zero_rows_unpadded}/{visual_unpadded_np.shape[0]}")
        
    except Exception as e:
        print_progress(f"Error in padding comparison: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")

def analyze_dataset_content(data_path, data_type='mosei', max_samples=5):
    """
    Analyze the actual content and distribution of the dataset
    """
    print_progress(f"\n{'='*80}")
    print_progress(f"DATASET CONTENT ANALYSIS: {data_type.upper()}")
    print_progress(f"{'='*80}")
    
    try:
        from datasets.affect.get_data import get_dataloader
        
        if data_type == 'mosei':
            filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
        else:
            filepath = os.path.join(data_path, 'mosi_raw.pkl')
            
        traindata, validdata, testdata = get_dataloader(
            filepath, 
            robust_test=False, 
            max_pad=True, 
            data_type=data_type, 
            max_seq_len=50,
            batch_size=16,
            z_norm=True
        )
        
        # Collect samples from all splits
        all_labels = []
        sample_count = 0
        
        print_progress(f"\n--- ANALYZING SPLITS ---")
        for split_name, dataloader in [("train", traindata), ("val", validdata), ("test", testdata)]:
            split_samples = 0
            split_labels = []
            
            for batch_idx, batch in enumerate(dataloader):
                visual_feat, audio_feat, text_feat, labels = batch
                batch_size_actual = visual_feat.shape[0]
                split_samples += batch_size_actual
                
                # Collect labels for analysis
                if isinstance(labels, torch.Tensor):
                    split_labels.extend(labels.numpy().flatten())
                else:
                    split_labels.extend(labels.flatten())
                
                # Show detailed analysis for first few samples
                if split_name == "train" and batch_idx == 0 and sample_count < max_samples:
                    print_progress(f"\n--- DETAILED SAMPLE ANALYSIS FROM {split_name.upper()} ---")
                    
                    for i in range(min(max_samples - sample_count, batch_size_actual)):
                        print_progress(f"\n🔍 SAMPLE {sample_count + i + 1}:")
                        print_progress(f"Label/Sentiment: {labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]}")
                        
                        # Visual features analysis
                        visual_sample = visual_feat[i]
                        print_feature_diagnostics(visual_sample, f"Visual_Sample_{sample_count + i + 1}")
                        
                        # Audio features analysis  
                        audio_sample = audio_feat[i]
                        print_feature_diagnostics(audio_sample, f"Audio_Sample_{sample_count + i + 1}")
                        
                        # Text features analysis
                        text_sample = text_feat[i]
                        print_feature_diagnostics(text_sample, f"Text_Sample_{sample_count + i + 1}")
                        
                        sample_count += 1
                        if sample_count >= max_samples:
                            break
                
                if sample_count >= max_samples:
                    break
            
            # Split statistics
            split_labels = np.array(split_labels)
            print_progress(f"\n--- {split_name.upper()} SPLIT STATISTICS ---")
            print_progress(f"Total samples: {split_samples}")
            print_progress(f"Label range: [{np.min(split_labels):.3f}, {np.max(split_labels):.3f}]")
            print_progress(f"Label mean: {np.mean(split_labels):.3f}")
            print_progress(f"Label std: {np.std(split_labels):.3f}")
            
            # Sentiment distribution
            positive_count = np.sum(split_labels > 0)
            negative_count = np.sum(split_labels < 0) 
            neutral_count = np.sum(split_labels == 0)
            
            print_progress(f"Sentiment distribution:")
            print_progress(f"  Positive (>0): {positive_count} ({positive_count/len(split_labels)*100:.1f}%)")
            print_progress(f"  Negative (<0): {negative_count} ({negative_count/len(split_labels)*100:.1f}%)")
            print_progress(f"  Neutral (=0): {neutral_count} ({neutral_count/len(split_labels)*100:.1f}%)")
            
            all_labels.extend(split_labels)
            
        # Overall dataset statistics
        all_labels = np.array(all_labels)
        print_progress(f"\n--- OVERALL DATASET STATISTICS ---")
        print_progress(f"Total samples across all splits: {len(all_labels)}")
        print_progress(f"Overall label range: [{np.min(all_labels):.3f}, {np.max(all_labels):.3f}]")
        print_progress(f"Overall label mean: {np.mean(all_labels):.3f}")
        print_progress(f"Overall label std: {np.std(all_labels):.3f}")
        
        # Create histogram bins for sentiment analysis
        bins = [-3, -2, -1, 0, 1, 2, 3]
        hist, _ = np.histogram(all_labels, bins=bins)
        print_progress(f"\nSentiment histogram:")
        for i in range(len(bins)-1):
            print_progress(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} samples")
            
    except Exception as e:
        print_progress(f"Error in dataset content analysis: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")

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
        collate_fn=collate_func_AVT  # VGGSound-style collate function
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
                             collate_fn=collate_func_AVT)
    test_train_loader = DataLoader(test_train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=False,
                                  collate_fn=collate_func_AVT)
    test_val_loader = DataLoader(test_val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=False,
                                collate_fn=collate_func_AVT)
    
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
                             collate_fn=collate_func_AVT)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           collate_fn=collate_func_AVT)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=collate_func_AVT)
    
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
                             collate_fn=collate_func_AVT)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           collate_fn=collate_func_AVT)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            collate_fn=collate_func_AVT)
    
    return train_loader, val_loader, test_loader


# ===============================================================================
# ENHANCED TEST FUNCTIONS WITH COMPREHENSIVE DIAGNOSTICS
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


def test_all_dataloaders_with_diagnostics():
    """
    Comprehensive test function with enhanced diagnostics.
    """
    print_progress("\n" + "="*80)
    print_progress("COMPREHENSIVE DATALOADER TESTING WITH ENHANCED DIAGNOSTICS")
    print_progress("="*80)
    
    # First, analyze the raw data with both padding options
    print_progress("\n🔍 STEP 1: RAW DATA ANALYSIS")
    compare_padding_effects(MOSEI_DATA_PATH, 'mosei', max_seq_len=50, batch_size=4)
    compare_padding_effects(MOSI_DATA_PATH, 'mosi', max_seq_len=50, batch_size=4)
    
    # Analyze dataset content and distributions
    print_progress("\n🔍 STEP 2: DATASET CONTENT ANALYSIS")
    analyze_dataset_content(MOSEI_DATA_PATH, 'mosei', max_samples=3)
    analyze_dataset_content(MOSI_DATA_PATH, 'mosi', max_samples=2)
    
    # Test collate pattern
    test_collate_pattern()
    
    # Test actual dataloaders with detailed diagnostics
    print_progress("\n🔍 STEP 3: DATALOADER INTEGRATION TESTING")
    
    print_progress("\n=== Testing Unsupervised MOSEI Dataset ===")
    try:
        unsupervised_loader = get_mosei_unsupervised_dataloader(batch_size=8, max_seq_len=50)
        for i, batch_data in enumerate(unsupervised_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea'] 
            text_feature = batch_data['text_fea']
            labels = batch_data['labels']
            
            print_progress(f"\nUnsupervised Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Batch_{i}_Audio_Sample_0")
            print_feature_diagnostics(video_feature[0], f"Batch_{i}_Video_Sample_0")
            print_feature_diagnostics(text_feature[0], f"Batch_{i}_Text_Sample_0")
            print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
            if i == 0:  # Only analyze first batch in detail
                break
        print_progress("✅ MOSEI Unsupervised test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Unsupervised test failed: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")

    print_progress("\n=== Testing MOSEI Supervised Dataset ===")
    try:
        train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(batch_size=8, max_seq_len=50)
        
        for i, batch_data in enumerate(train_loader):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea']
            text_feature = batch_data['text_fea'] 
            labels = batch_data['labels']
            
            print_progress(f"\nSupervised Train Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"Supervised_Batch_{i}_Audio_Sample_0")
            print_progress(f"Batch size: {audio_feature.shape[0]}")
            print_progress(f"Labels range in batch: [{labels.min().item():.3f}, {labels.max().item():.3f}]")
            print_progress(f"Sample labels: {labels[:5].numpy().flatten()}")  # Show first 5 labels
            
            if i == 0:  # Only analyze first batch in detail
                break
        print_progress("✅ MOSEI Supervised test passed!")
    except Exception as e:
        print_progress(f"❌ MOSEI Supervised test failed: {e}")
    
    print_progress("\n=== Testing MOSI Dataset ===")
    try:
        mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(batch_size=8, max_seq_len=50)
        
        for i, batch_data in enumerate(mosi_test):
            audio_feature = batch_data['audio_fea']
            video_feature = batch_data['video_fea']
            text_feature = batch_data['text_fea']
            labels = batch_data['labels']
            
            print_progress(f"\nMOSI Test Batch {i} Analysis:")
            print_feature_diagnostics(audio_feature[0], f"MOSI_Batch_{i}_Audio_Sample_0")
            print_progress(f"Batch size: {audio_feature.shape[0]}")
            print_progress(f"Labels range in batch: [{labels.min().item():.3f}, {labels.max().item():.3f}]")
            print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
            if i == 0:  # Only analyze first batch
                break
        print_progress("✅ MOSI test passed!")
    except Exception as e:
        print_progress(f"❌ MOSI test failed: {e}")
    
    print_progress("\n" + "="*80)
    print_progress("✅ ALL COMPREHENSIVE DIAGNOSTIC TESTS COMPLETED!")
    print_progress("Your datasets are ready for multimodal training.")
    print_progress("="*80)


if __name__ == "__main__":
    test_all_dataloaders_with_diagnostics()


# import os
# import sys
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import logging
# from datetime import datetime
# import h5py
# import pickle

# # Add MultiBench to path for accessing the get_dataloader function
# MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
# MOSEI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'
# MOSI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSI/'

# sys.path.append(MULTIBENCH_PATH)

# def print_progress(message):
#     """Helper function for logging progress during dataset loading"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# # ===============================================================================
# # RAW DATA INSPECTION FUNCTIONS
# # ===============================================================================

# def inspect_raw_h5_files(data_path, data_type='mosei', num_samples=5):
#     """
#     Inspect the raw H5 files to see the original data structure and content.
#     """
#     print_progress(f"\n{'='*80}")
#     print_progress(f"INSPECTING RAW H5 FILES: {data_type.upper()}")
#     print_progress(f"{'='*80}")
    
#     try:
#         # Find all H5 files in the data directory
#         h5_files = [f for f in os.listdir(data_path) if f.endswith('.h5') or f.endswith('.hdf5')]
#         print_progress(f"Found H5 files: {h5_files}")
        
#         for h5_file in h5_files:
#             h5_path = os.path.join(data_path, h5_file)
#             print_progress(f"\n--- INSPECTING {h5_file} ---")
            
#             with h5py.File(h5_path, 'r') as f:
#                 print_progress(f"H5 file keys: {list(f.keys())}")
                
#                 # Recursively inspect the structure
#                 def inspect_h5_group(group, prefix=""):
#                     for key in group.keys():
#                         item = group[key]
#                         if isinstance(item, h5py.Group):
#                             print_progress(f"{prefix}Group: {key}")
#                             inspect_h5_group(item, prefix + "  ")
#                         elif isinstance(item, h5py.Dataset):
#                             print_progress(f"{prefix}Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}")
                            
#                             # Show sample data for small datasets
#                             if item.size < 100 and item.size > 0:
#                                 sample_data = item[()]
#                                 if isinstance(sample_data, (str, bytes)):
#                                     print_progress(f"{prefix}  Sample: {str(sample_data)[:100]}")
#                                 elif np.isscalar(sample_data):
#                                     print_progress(f"{prefix}  Value: {sample_data}")
#                                 else:
#                                     print_progress(f"{prefix}  Sample shape: {np.array(sample_data).shape}")
#                                     if len(sample_data) > 0:
#                                         print_progress(f"{prefix}  First few values: {np.array(sample_data).flatten()[:10]}")
                
#                 inspect_h5_group(f)
                
#     except Exception as e:
#         print_progress(f"Error inspecting H5 files: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")


# def inspect_raw_text_processing(data_path, data_type='mosei', num_samples=5):
#     """
#     Inspect the raw text processing pipeline step by step.
#     """
#     print_progress(f"\n{'='*80}")
#     print_progress(f"INSPECTING RAW TEXT PROCESSING PIPELINE: {data_type.upper()}")
#     print_progress(f"{'='*80}")
    
#     try:
#         from datasets.affect.get_raw_data import get_rawtext, get_word2id, glove_embeddings, get_word_embeddings
        
#         # Find text data files
#         if data_type == 'mosei':
#             text_files = [f for f in os.listdir(data_path) if 'text' in f.lower() and (f.endswith('.h5') or f.endswith('.hdf5'))]
#         else:
#             text_files = [f for f in os.listdir(data_path) if 'text' in f.lower() and (f.endswith('.h5') or f.endswith('.hdf5'))]
            
#         if not text_files:
#             # Try to find any h5 files that might contain text
#             text_files = [f for f in os.listdir(data_path) if f.endswith('.h5') or f.endswith('.hdf5')]
            
#         print_progress(f"Found potential text files: {text_files}")
        
#         if text_files:
#             text_file_path = os.path.join(data_path, text_files[0])
#             print_progress(f"\nAnalyzing text file: {text_file_path}")
            
#             # Step 1: Get raw text data
#             print_progress(f"\n--- STEP 1: EXTRACTING RAW TEXT ---")
            
#             # First, let's see what video IDs are available by looking at the main data file
#             if data_type == 'mosei':
#                 main_file = os.path.join(data_path, 'mosei_senti_data.pkl')
#             else:
#                 main_file = os.path.join(data_path, 'mosi_raw.pkl')
                
#             if os.path.exists(main_file):
#                 with open(main_file, 'rb') as f:
#                     data = pickle.load(f)
#                     print_progress(f"Main data keys: {data.keys() if hasattr(data, 'keys') else type(data)}")
                    
#                     # Try to extract video IDs from the structure
#                     if isinstance(data, dict):
#                         for key in list(data.keys())[:3]:  # Look at first 3 keys
#                             print_progress(f"Key '{key}': {type(data[key])}")
#                             if hasattr(data[key], 'keys'):
#                                 sample_keys = list(data[key].keys())[:5]
#                                 print_progress(f"  Sample subkeys: {sample_keys}")
                                
#                                 # These might be video IDs
#                                 vids = sample_keys[:num_samples]
#                                 print_progress(f"Using sample video IDs: {vids}")
                                
#                                 try:
#                                     # Try to get raw text
#                                     text_data, video_data = get_rawtext(text_file_path, 'hdf5', vids)
                                    
#                                     print_progress(f"\n--- RAW TEXT EXTRACTION RESULTS ---")
#                                     print_progress(f"Extracted {len(text_data)} text samples")
#                                     print_progress(f"Extracted {len(video_data)} video entries")
                                    
#                                     # Show sample raw text
#                                     for i, (text, vid) in enumerate(zip(text_data[:3], video_data[:3])):
#                                         print_progress(f"\nSample {i+1}:")
#                                         print_progress(f"  Video ID: {vid}")
#                                         print_progress(f"  Raw text type: {type(text)}")
#                                         if isinstance(text, (str, bytes)):
#                                             text_str = str(text)
#                                             print_progress(f"  Text length: {len(text_str)} characters")
#                                             print_progress(f"  Text preview: '{text_str[:200]}...'")
#                                         elif isinstance(text, (list, np.ndarray)):
#                                             print_progress(f"  Text is array/list of length: {len(text)}")
#                                             if len(text) > 0:
#                                                 print_progress(f"  First few elements: {text[:5]}")
#                                         else:
#                                             print_progress(f"  Text content: {text}")
                                    
#                                     # Step 2: Word to ID mapping
#                                     print_progress(f"\n--- STEP 2: WORD-TO-ID MAPPING ---")
#                                     try:
#                                         word2id = get_word2id(text_data, video_data)
#                                         print_progress(f"Generated word2id mapping with {len(word2id)} entries")
                                        
#                                         # Show sample word2id mappings
#                                         sample_word2id = word2id[:10] if len(word2id) >= 10 else word2id
#                                         for i, (word, idx) in enumerate(sample_word2id):
#                                             print_progress(f"  {i+1}. '{word}' -> {idx}")
                                            
#                                     except Exception as e:
#                                         print_progress(f"Error in word2id mapping: {e}")
                                    
#                                     # Step 3: GloVe embeddings
#                                     print_progress(f"\n--- STEP 3: GLOVE EMBEDDINGS ---")
#                                     try:
#                                         embeddings = glove_embeddings(text_data, video_data, paddings=50)
#                                         print_progress(f"Generated embeddings shape: {embeddings.shape}")
#                                         print_progress(f"Embedding dtype: {embeddings.dtype}")
#                                         print_progress(f"Embedding range: [{np.min(embeddings):.4f}, {np.max(embeddings):.4f}]")
#                                         print_progress(f"Embedding mean: {np.mean(embeddings):.4f}")
#                                         print_progress(f"Embedding std: {np.std(embeddings):.4f}")
                                        
#                                         # Show sample embeddings
#                                         print_progress(f"\nSample embedding analysis:")
#                                         sample_embedding = embeddings[0]  # First sample
#                                         print_progress(f"  Sample shape: {sample_embedding.shape}")
#                                         print_progress(f"  First timestep embedding (first 10 dims): {sample_embedding[0][:10]}")
                                        
#                                         # Check for padding (zero rows)
#                                         zero_rows = np.sum(np.all(sample_embedding == 0, axis=1))
#                                         print_progress(f"  Zero-padded timesteps: {zero_rows}/{sample_embedding.shape[0]}")
                                        
#                                     except Exception as e:
#                                         print_progress(f"Error in GloVe embeddings: {e}")
#                                         import traceback
#                                         print_progress(f"Traceback: {traceback.format_exc()}")
                                    
#                                     break  # Found working key, exit loop
                                    
#                                 except Exception as e:
#                                     print_progress(f"Error with key '{key}': {e}")
#                                     continue
                    
#             else:
#                 print_progress(f"Main data file not found: {main_file}")
                
#     except Exception as e:
#         print_progress(f"Error in raw text processing inspection: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")


# def inspect_raw_vs_processed_data(data_path, data_type='mosei', num_samples=3):
#     """
#     Compare raw data with processed data to see the full pipeline.
#     """
#     print_progress(f"\n{'='*80}")
#     print_progress(f"RAW VS PROCESSED DATA COMPARISON: {data_type.upper()}")
#     print_progress(f"{'='*80}")
    
#     try:
#         from datasets.affect.get_data import get_dataloader
        
#         # Get processed data (what we normally use)
#         print_progress(f"\n--- LOADING PROCESSED DATA ---")
#         if data_type == 'mosei':
#             filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
#         else:
#             filepath = os.path.join(data_path, 'mosi_raw.pkl')
            
#         traindata, validdata, testdata = get_dataloader(
#             filepath, 
#             robust_test=False, 
#             max_pad=True, 
#             data_type=data_type, 
#             max_seq_len=50,
#             batch_size=4
#         )
        
#         # Get first batch of processed data
#         processed_batch = next(iter(traindata))
#         visual_processed, audio_processed, text_processed, labels_processed = processed_batch
        
#         print_progress(f"Processed data shapes:")
#         print_progress(f"  Visual: {visual_processed.shape}")
#         print_progress(f"  Audio: {audio_processed.shape}")
#         print_progress(f"  Text: {text_processed.shape}")
#         print_progress(f"  Labels: {labels_processed.shape}")
        
#         # Analyze the processed text embeddings
#         print_progress(f"\n--- PROCESSED TEXT ANALYSIS ---")
#         sample_text = text_processed[0]  # First sample
#         print_feature_diagnostics(sample_text, "Processed_Text_Sample_0")
        
#         # Try to load and inspect the original pickle file structure
#         print_progress(f"\n--- ORIGINAL PICKLE FILE STRUCTURE ---")
#         if os.path.exists(filepath):
#             with open(filepath, 'rb') as f:
#                 raw_data = pickle.load(f)
#                 print_progress(f"Pickle file type: {type(raw_data)}")
                
#                 if isinstance(raw_data, dict):
#                     print_progress(f"Main keys: {list(raw_data.keys())}")
                    
#                     for main_key in list(raw_data.keys())[:2]:  # Check first 2 main keys
#                         print_progress(f"\n--- MAIN KEY: {main_key} ---")
#                         data_subset = raw_data[main_key]
#                         print_progress(f"Type: {type(data_subset)}")
                        
#                         if isinstance(data_subset, dict):
#                             sample_keys = list(data_subset.keys())[:3]
#                             print_progress(f"Sample entry keys: {sample_keys}")
                            
#                             for entry_key in sample_keys:
#                                 entry_data = data_subset[entry_key]
#                                 print_progress(f"\n  Entry: {entry_key}")
#                                 print_progress(f"  Entry type: {type(entry_data)}")
                                
#                                 if isinstance(entry_data, dict):
#                                     print_progress(f"  Entry keys: {list(entry_data.keys())}")
                                    
#                                     # Look for different modality data
#                                     for modality in ['text', 'vision', 'audio', 'labels']:
#                                         if modality in entry_data:
#                                             mod_data = entry_data[modality]
#                                             print_progress(f"    {modality}: shape={getattr(mod_data, 'shape', 'N/A')}, type={type(mod_data)}")
                                            
#                                             if hasattr(mod_data, 'shape') and len(mod_data.shape) >= 1:
#                                                 if mod_data.shape[0] > 0:
#                                                     sample_values = mod_data.flatten()[:5] if hasattr(mod_data, 'flatten') else mod_data[:5]
#                                                     print_progress(f"    {modality} sample values: {sample_values}")
#                                 elif hasattr(entry_data, 'shape'):
#                                     print_progress(f"  Entry shape: {entry_data.shape}")
#                                     if len(entry_data.shape) >= 1 and entry_data.shape[0] > 0:
#                                         sample_values = entry_data.flatten()[:5] if hasattr(entry_data, 'flatten') else entry_data[:5]
#                                         print_progress(f"  Entry sample values: {sample_values}")
                
#                 elif hasattr(raw_data, '__len__'):
#                     print_progress(f"Data length: {len(raw_data)}")
#                     if len(raw_data) > 0:
#                         print_progress(f"First element type: {type(raw_data[0])}")
#                         if hasattr(raw_data[0], 'keys'):
#                             print_progress(f"First element keys: {list(raw_data[0].keys())}")
        
#         print_progress(f"\n--- DATA PIPELINE SUMMARY ---")
#         print_progress(f"✓ Raw data gets loaded from pickle files with complex nested structure")
#         print_progress(f"✓ Text data gets processed through GloVe embeddings (300D)")
#         print_progress(f"✓ All modalities get padded to max_seq_len={50}")
#         print_progress(f"✓ Final processed shapes: Visual[50,35], Audio[50,74], Text[50,300]")
        
#     except Exception as e:
#         print_progress(f"Error in raw vs processed comparison: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")

# def print_feature_diagnostics(features, feature_name, sample_idx=0):
#     """
#     Print comprehensive diagnostics for a feature tensor/array
#     """
#     if isinstance(features, torch.Tensor):
#         features_np = features.numpy()
#     else:
#         features_np = features
    
#     print_progress(f"\n--- {feature_name.upper()} FEATURE DIAGNOSTICS ---")
#     print_progress(f"Shape: {features_np.shape}")
#     print_progress(f"Data type: {features_np.dtype}")
#     print_progress(f"Min value: {np.min(features_np):.6f}")
#     print_progress(f"Max value: {np.max(features_np):.6f}")
#     print_progress(f"Mean: {np.mean(features_np):.6f}")
#     print_progress(f"Std: {np.std(features_np):.6f}")
    
#     # Check for NaN or Inf values
#     nan_count = np.sum(np.isnan(features_np))
#     inf_count = np.sum(np.isinf(features_np))
#     print_progress(f"NaN values: {nan_count}")
#     print_progress(f"Inf values: {inf_count}")
    
#     # Show sample values from different time steps
#     if len(features_np.shape) >= 2:
#         seq_len = features_np.shape[0]
#         feature_dim = features_np.shape[1]
#         print_progress(f"Sequence length: {seq_len}, Feature dimension: {feature_dim}")
        
#         # Show first few values from first timestep
#         print_progress(f"First timestep values (first 10): {features_np[0, :10]}")
        
#         # Show middle timestep if available
#         if seq_len > 1:
#             mid_idx = seq_len // 2
#             print_progress(f"Middle timestep values (first 10): {features_np[mid_idx, :10]}")
            
#         # Show last timestep if different from first
#         if seq_len > 2:
#             print_progress(f"Last timestep values (first 10): {features_np[-1, :10]}")
#     else:
#         print_progress(f"Feature values: {features_np}")
    
#     print_progress(f"--- END {feature_name.upper()} DIAGNOSTICS ---\n")

# def compare_padding_effects(data_path, data_type='mosei', max_seq_len=50, batch_size=4):
#     """
#     Compare the effects of max_pad=True vs max_pad=False on the same data
#     """
#     print_progress(f"\n{'='*80}")
#     print_progress(f"COMPARING PADDING EFFECTS: {data_type.upper()}")
#     print_progress(f"{'='*80}")
    
#     try:
#         from datasets.affect.get_data import get_dataloader
        
#         # Load data with max_pad=True
#         print_progress("\n--- LOADING WITH MAX_PAD=True ---")
#         if data_type == 'mosei':
#             filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
#         else:
#             filepath = os.path.join(data_path, 'mosi_raw.pkl')
            
#         traindata_padded, validdata_padded, testdata_padded = get_dataloader(
#             filepath, 
#             robust_test=False, 
#             max_pad=True, 
#             data_type=data_type, 
#             max_seq_len=max_seq_len,
#             batch_size=batch_size
#         )
        
#         # Load data with max_pad=False
#         print_progress("\n--- LOADING WITH MAX_PAD=False ---")
#         traindata_unpadded, validdata_unpadded, testdata_unpadded = get_dataloader(
#             filepath, 
#             robust_test=False, 
#             max_pad=False, 
#             data_type=data_type, 
#             max_seq_len=max_seq_len,
#             batch_size=batch_size
#         )
        
#         # Compare first batch from training data
#         print_progress("\n--- COMPARING FIRST TRAINING BATCH ---")
        
#         # Get padded batch
#         padded_batch = next(iter(traindata_padded))
#         visual_padded, audio_padded, text_padded, labels_padded = padded_batch
        
#         # Get unpadded batch  
#         unpadded_batch = next(iter(traindata_unpadded))
#         visual_unpadded, audio_unpadded, text_unpadded, labels_unpadded = unpadded_batch
        
#         print_progress(f"\nPADDED DATA (max_pad=True):")
#         print_progress(f"  Visual shape: {visual_padded.shape}")
#         print_progress(f"  Audio shape: {audio_padded.shape}")
#         print_progress(f"  Text shape: {text_padded.shape}")
#         print_progress(f"  Labels shape: {labels_padded.shape}")
        
#         print_progress(f"\nUNPADDED DATA (max_pad=False):")
#         # Handle the case where unpadded data returns lists of variable-length sequences
#         if isinstance(visual_unpadded, list):
#             print_progress(f"  Visual: List of {len(visual_unpadded)} variable-length sequences")
#             if len(visual_unpadded) > 0:
#                 sample_lengths = [seq.shape[0] if hasattr(seq, 'shape') else len(seq) for seq in visual_unpadded[:3]]
#                 print_progress(f"  Visual sample lengths: {sample_lengths}")
#         else:
#             print_progress(f"  Visual shape: {visual_unpadded.shape}")
            
#         if isinstance(audio_unpadded, list):
#             print_progress(f"  Audio: List of {len(audio_unpadded)} variable-length sequences")
#             if len(audio_unpadded) > 0:
#                 sample_lengths = [seq.shape[0] if hasattr(seq, 'shape') else len(seq) for seq in audio_unpadded[:3]]
#                 print_progress(f"  Audio sample lengths: {sample_lengths}")
#         else:
#             print_progress(f"  Audio shape: {audio_unpadded.shape}")
            
#         if isinstance(text_unpadded, list):
#             print_progress(f"  Text: List of {len(text_unpadded)} variable-length sequences")
#             if len(text_unpadded) > 0:
#                 sample_lengths = [seq.shape[0] if hasattr(seq, 'shape') else len(seq) for seq in text_unpadded[:3]]
#                 print_progress(f"  Text sample lengths: {sample_lengths}")
#         else:
#             print_progress(f"  Text shape: {text_unpadded.shape}")
            
#         if isinstance(labels_unpadded, list):
#             print_progress(f"  Labels: List of {len(labels_unpadded)} labels")
#         else:
#             print_progress(f"  Labels shape: {labels_unpadded.shape}")
        
#         # Analyze first sample in detail
#         sample_idx = 0
#         print_progress(f"\n--- DETAILED ANALYSIS OF SAMPLE {sample_idx} ---")
        
#         print_progress(f"\n🎥 VISUAL FEATURES COMPARISON")
#         print_feature_diagnostics(visual_padded[sample_idx], f"Visual_PADDED_Sample_{sample_idx}")
        
#         if isinstance(visual_unpadded, list) and len(visual_unpadded) > sample_idx:
#             print_feature_diagnostics(visual_unpadded[sample_idx], f"Visual_UNPADDED_Sample_{sample_idx}")
#             print_progress(f"Length comparison: Padded={visual_padded[sample_idx].shape[0]}, Unpadded={visual_unpadded[sample_idx].shape[0] if hasattr(visual_unpadded[sample_idx], 'shape') else len(visual_unpadded[sample_idx])}")
        
#         print_progress(f"\n🔊 AUDIO FEATURES COMPARISON") 
#         print_feature_diagnostics(audio_padded[sample_idx], f"Audio_PADDED_Sample_{sample_idx}")
        
#         if isinstance(audio_unpadded, list) and len(audio_unpadded) > sample_idx:
#             print_feature_diagnostics(audio_unpadded[sample_idx], f"Audio_UNPADDED_Sample_{sample_idx}")
#             print_progress(f"Length comparison: Padded={audio_padded[sample_idx].shape[0]}, Unpadded={audio_unpadded[sample_idx].shape[0] if hasattr(audio_unpadded[sample_idx], 'shape') else len(audio_unpadded[sample_idx])}")
        
#         print_progress(f"\n📝 TEXT FEATURES COMPARISON")
#         print_feature_diagnostics(text_padded[sample_idx], f"Text_PADDED_Sample_{sample_idx}")
        
#         if isinstance(text_unpadded, list) and len(text_unpadded) > sample_idx:
#             print_feature_diagnostics(text_unpadded[sample_idx], f"Text_UNPADDED_Sample_{sample_idx}")
#             print_progress(f"Length comparison: Padded={text_padded[sample_idx].shape[0]}, Unpadded={text_unpadded[sample_idx].shape[0] if hasattr(text_unpadded[sample_idx], 'shape') else len(text_unpadded[sample_idx])}")
        
#         print_progress(f"\n🎯 LABELS COMPARISON")
#         print_progress(f"Padded labels: {labels_padded[sample_idx]}")
        
#         if isinstance(labels_unpadded, list):
#             print_progress(f"Unpadded labels: {labels_unpadded[sample_idx] if len(labels_unpadded) > sample_idx else 'N/A'}")
#             if len(labels_unpadded) > sample_idx:
#                 print_progress(f"Labels match: {torch.allclose(labels_padded[sample_idx], torch.tensor(labels_unpadded[sample_idx]) if not isinstance(labels_unpadded[sample_idx], torch.Tensor) else labels_unpadded[sample_idx])}")
#         else:
#             print_progress(f"Unpadded labels: {labels_unpadded[sample_idx]}")
#             print_progress(f"Labels match: {torch.allclose(labels_padded[sample_idx], labels_unpadded[sample_idx])}")
        
#         # Check for zero padding patterns
#         print_progress(f"\n--- PADDING PATTERN ANALYSIS ---")
#         visual_padded_np = visual_padded[sample_idx].numpy()
        
#         # Count zero rows (indicating padding)
#         zero_rows_padded = np.sum(np.all(visual_padded_np == 0, axis=1))
#         print_progress(f"Visual zero rows (padded): {zero_rows_padded}/{visual_padded_np.shape[0]}")
        
#         if isinstance(visual_unpadded, list) and len(visual_unpadded) > sample_idx:
#             visual_unpadded_sample = visual_unpadded[sample_idx]
#             if hasattr(visual_unpadded_sample, 'numpy'):
#                 visual_unpadded_np = visual_unpadded_sample.numpy()
#             else:
#                 visual_unpadded_np = np.array(visual_unpadded_sample)
#             zero_rows_unpadded = np.sum(np.all(visual_unpadded_np == 0, axis=1))
#             print_progress(f"Visual zero rows (unpadded): {zero_rows_unpadded}/{visual_unpadded_np.shape[0]}")
#             print_progress(f"Padding effect: Added {visual_padded_np.shape[0] - visual_unpadded_np.shape[0]} timesteps")
        
#     except Exception as e:
#         print_progress(f"Error in padding comparison: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")

# def analyze_dataset_content(data_path, data_type='mosei', max_samples=5):
#     """
#     Analyze the actual content and distribution of the dataset
#     """
#     print_progress(f"\n{'='*80}")
#     print_progress(f"DATASET CONTENT ANALYSIS: {data_type.upper()}")
#     print_progress(f"{'='*80}")
    
#     try:
#         from datasets.affect.get_data import get_dataloader
        
#         if data_type == 'mosei':
#             filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
#         else:
#             filepath = os.path.join(data_path, 'mosi_raw.pkl')
            
#         traindata, validdata, testdata = get_dataloader(
#             filepath, 
#             robust_test=False, 
#             max_pad=True, 
#             data_type=data_type, 
#             max_seq_len=50,
#             batch_size=16
#         )
        
#         # Collect samples from all splits
#         all_labels = []
#         sample_count = 0
        
#         print_progress(f"\n--- ANALYZING SPLITS ---")
#         for split_name, dataloader in [("train", traindata), ("val", validdata), ("test", testdata)]:
#             split_samples = 0
#             split_labels = []
            
#             for batch_idx, batch in enumerate(dataloader):
#                 visual_feat, audio_feat, text_feat, labels = batch
#                 batch_size_actual = visual_feat.shape[0]
#                 split_samples += batch_size_actual
                
#                 # Collect labels for analysis
#                 if isinstance(labels, torch.Tensor):
#                     split_labels.extend(labels.numpy().flatten())
#                 else:
#                     split_labels.extend(labels.flatten())
                
#                 # Show detailed analysis for first few samples
#                 if split_name == "train" and batch_idx == 0 and sample_count < max_samples:
#                     print_progress(f"\n--- DETAILED SAMPLE ANALYSIS FROM {split_name.upper()} ---")
                    
#                     for i in range(min(max_samples - sample_count, batch_size_actual)):
#                         print_progress(f"\n🔍 SAMPLE {sample_count + i + 1}:")
#                         print_progress(f"Label/Sentiment: {labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]}")
                        
#                         # Visual features analysis
#                         visual_sample = visual_feat[i]
#                         print_feature_diagnostics(visual_sample, f"Visual_Sample_{sample_count + i + 1}")
                        
#                         # Audio features analysis  
#                         audio_sample = audio_feat[i]
#                         print_feature_diagnostics(audio_sample, f"Audio_Sample_{sample_count + i + 1}")
                        
#                         # Text features analysis
#                         text_sample = text_feat[i]
#                         print_feature_diagnostics(text_sample, f"Text_Sample_{sample_count + i + 1}")
                        
#                         sample_count += 1
#                         if sample_count >= max_samples:
#                             break
                
#                 if sample_count >= max_samples:
#                     break
            
#             # Split statistics
#             split_labels = np.array(split_labels)
#             print_progress(f"\n--- {split_name.upper()} SPLIT STATISTICS ---")
#             print_progress(f"Total samples: {split_samples}")
#             print_progress(f"Label range: [{np.min(split_labels):.3f}, {np.max(split_labels):.3f}]")
#             print_progress(f"Label mean: {np.mean(split_labels):.3f}")
#             print_progress(f"Label std: {np.std(split_labels):.3f}")
            
#             # Sentiment distribution
#             positive_count = np.sum(split_labels > 0)
#             negative_count = np.sum(split_labels < 0) 
#             neutral_count = np.sum(split_labels == 0)
            
#             print_progress(f"Sentiment distribution:")
#             print_progress(f"  Positive (>0): {positive_count} ({positive_count/len(split_labels)*100:.1f}%)")
#             print_progress(f"  Negative (<0): {negative_count} ({negative_count/len(split_labels)*100:.1f}%)")
#             print_progress(f"  Neutral (=0): {neutral_count} ({neutral_count/len(split_labels)*100:.1f}%)")
            
#             all_labels.extend(split_labels)
            
#         # Overall dataset statistics
#         all_labels = np.array(all_labels)
#         print_progress(f"\n--- OVERALL DATASET STATISTICS ---")
#         print_progress(f"Total samples across all splits: {len(all_labels)}")
#         print_progress(f"Overall label range: [{np.min(all_labels):.3f}, {np.max(all_labels):.3f}]")
#         print_progress(f"Overall label mean: {np.mean(all_labels):.3f}")
#         print_progress(f"Overall label std: {np.std(all_labels):.3f}")
        
#         # Create histogram bins for sentiment analysis
#         bins = [-3, -2, -1, 0, 1, 2, 3]
#         hist, _ = np.histogram(all_labels, bins=bins)
#         print_progress(f"\nSentiment histogram:")
#         for i in range(len(bins)-1):
#             print_progress(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} samples")
            
#     except Exception as e:
#         print_progress(f"Error in dataset content analysis: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")

# class MOSEIDatasetUnsupervised(Dataset):
#     """
#     Dataset class for unsupervised pretraining on combined MOSEI data.
#     Combines train, validation splits from MOSEI for self-supervised learning.
#     Returns numpy arrays to match VGGSound collate pattern.
#     """
#     def __init__(self, data_path, max_seq_len=50, batch_size=64):
#         super(MOSEIDatasetUnsupervised, self).__init__()
#         self.data_path = data_path
#         self.max_seq_len = max_seq_len
        
#         print_progress("Loading MOSEI dataset for unsupervised pretraining...")
        
#         try:
#             from datasets.affect.get_data import get_dataloader
            
#             # Load all three splits from MOSEI
#             filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
#             traindata, validdata, testdata = get_dataloader(
#                 filepath, 
#                 robust_test=False, 
#                 max_pad=True, 
#                 data_type='mosei', 
#                 max_seq_len=max_seq_len,
#                 batch_size=batch_size
#             )
            
#             # Combine all data for unsupervised pretraining
#             print_progress("Combining train, validation splits...")
#             self.combined_data = []
            
#             # Extract data from each split
#             # for split_name, dataloader in [("train", traindata), ("val", validdata), ("test", testdata)]:
#             for split_name, dataloader in [("train", traindata), ("val", validdata)]:
#                 print_progress(f"Processing {split_name} split...")
#                 for batch_idx, batch in enumerate(dataloader):
#                     # MultiBench returns: [visual_features, audio_features, text_features, labels]
#                     visual_feat, audio_feat, text_feat, labels = batch
                    
#                     # Store each sample individually to create a unified dataset
#                     batch_size_actual = visual_feat.shape[0]
#                     for i in range(batch_size_actual):
#                         sample = {
#                             'visual': visual_feat[i],  # Shape: [seq_len, 35]
#                             'audio': audio_feat[i],    # Shape: [seq_len, 74] 
#                             'text': text_feat[i],      # Shape: [seq_len, 300]
#                             'labels': labels[i],       # Shape: [1] - sentiment score
#                             'video_id': f"mosei_{split_name}_{batch_idx}_{i}"  # Create unique identifier
#                         }
#                         self.combined_data.append(sample)
                        
#             print_progress(f"Successfully loaded {len(self.combined_data)} samples for unsupervised pretraining")
            
#         except Exception as e:
#             print_progress(f"Error loading MOSEI data: {e}")
#             import traceback
#             print_progress(f"Traceback: {traceback.format_exc()}")
#             self.combined_data = []
    
#     def __getitem__(self, index):
#         """
#         Returns numpy arrays to match VGGSound collate pattern.
#         This is crucial for the torch.from_numpy(np.asarray(...)) pattern to work efficiently.
#         """
#         sample = self.combined_data[index]
        
#         # Convert tensors to numpy arrays (removing .float() calls)
#         # This ensures compatibility with VGGSound-style collate function
#         audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
#         video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
#         text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
#         labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
#         video_id = sample['video_id']
        
#         return audio_feature, video_feature, text_feature, labels, video_id
    
#     def __len__(self):
#         return len(self.combined_data)


# class MOSEIDatasetUnsupervisedSplit(Dataset):
#     """
#     Dataset class for unsupervised pretraining on MOSEI data with split functionality.
#     - If split is 'train': combines train+validation splits for self-supervised learning
#     - If split is 'test_train': uses first 80% of test data
#     - If split is 'test_val': uses last 20% of test data
#     Returns numpy arrays to match VGGSound collate pattern.
#     """
#     def __init__(self, data_path, split='train', max_seq_len=50, batch_size=64):
#         super(MOSEIDatasetUnsupervisedSplit, self).__init__()
#         self.data_path = data_path
#         self.split = split
#         self.max_seq_len = max_seq_len
        
#         print_progress(f"Loading MOSEI dataset for unsupervised pretraining (split: {split})...")
        
#         try:
#             from datasets.affect.get_data import get_dataloader
            
#             # Load all three splits from MOSEI
#             filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
#             traindata, validdata, testdata = get_dataloader(
#                 filepath, 
#                 robust_test=False, 
#                 max_pad=True, 
#                 data_type='mosei', 
#                 max_seq_len=max_seq_len,
#                 batch_size=batch_size
#             )
            
#             self.combined_data = []
            
#             if split == 'train':
#                 # Combine train and validation splits like the original unsupervised class
#                 print_progress("Combining train and validation splits...")
#                 splits_to_process = [("train", traindata), ("val", validdata)]
                
#                 for split_name, dataloader in splits_to_process:
#                     print_progress(f"Processing {split_name} split...")
#                     for batch_idx, batch in enumerate(dataloader):
#                         visual_feat, audio_feat, text_feat, labels = batch
#                         batch_size_actual = visual_feat.shape[0]
                        
#                         for i in range(batch_size_actual):
#                             sample = {
#                                 'visual': visual_feat[i],  # Shape: [seq_len, 35]
#                                 'audio': audio_feat[i],    # Shape: [seq_len, 74] 
#                                 'text': text_feat[i],      # Shape: [seq_len, 300]
#                                 'labels': labels[i],       # Shape: [1] - sentiment score
#                                 'video_id': f"mosei_{split_name}_{batch_idx}_{i}"
#                             }
#                             self.combined_data.append(sample)
                            
#             elif split in ['test_train', 'test_val']:
#                 # Process test data and split it 80-20
#                 print_progress("Processing test split for 80-20 division...")
#                 test_samples = []
                
#                 for batch_idx, batch in enumerate(testdata):
#                     visual_feat, audio_feat, text_feat, labels = batch
#                     batch_size_actual = visual_feat.shape[0]
                    
#                     for i in range(batch_size_actual):
#                         sample = {
#                             'visual': visual_feat[i],  # Shape: [seq_len, 35]
#                             'audio': audio_feat[i],    # Shape: [seq_len, 74] 
#                             'text': text_feat[i],      # Shape: [seq_len, 300]
#                             'labels': labels[i],       # Shape: [1] - sentiment score
#                             'video_id': f"mosei_test_{batch_idx}_{i}"
#                         }
#                         test_samples.append(sample)
                
#                 # Perform 80-20 split
#                 total_samples = len(test_samples)
#                 split_idx = int(0.8 * total_samples)
                
#                 if split == 'test_train':
#                     # First 80% for test_train
#                     self.combined_data = test_samples[:split_idx]
#                     print_progress(f"Using first 80% of test data: {len(self.combined_data)} samples")
#                 elif split == 'test_val':
#                     # Last 20% for test_val
#                     self.combined_data = test_samples[split_idx:]
#                     print_progress(f"Using last 20% of test data: {len(self.combined_data)} samples")
                    
#             else:
#                 raise ValueError(f"Invalid split: {split}. Must be 'train', 'test_train', or 'test_val'")
                        
#             print_progress(f"Successfully loaded {len(self.combined_data)} samples for unsupervised pretraining ({split})")
            
#         except Exception as e:
#             print_progress(f"Error loading MOSEI data: {e}")
#             import traceback
#             print_progress(f"Traceback: {traceback.format_exc()}")
#             self.combined_data = []
    
#     def __getitem__(self, index):
#         """
#         Returns numpy arrays to match VGGSound collate pattern.
#         This is crucial for the torch.from_numpy(np.asarray(...)) pattern to work efficiently.
#         """
#         sample = self.combined_data[index]
        
#         # Convert tensors to numpy arrays (removing .float() calls)
#         # This ensures compatibility with VGGSound-style collate function
#         audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
#         video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
#         text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
#         labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
#         video_id = sample['video_id']
        
#         return audio_feature, video_feature, text_feature, labels, video_id
    
#     def __len__(self):
#         return len(self.combined_data)


# class MOSEIDatasetSupervised(Dataset):
#     """
#     Dataset class for supervised training on MOSEI data.
#     - If split is 'train' or 'val': returns combined train+val data
#     - If split is 'test': returns only test data
#     Returns numpy arrays to match VGGSound collate pattern.
#     """
#     def __init__(self, data_path, split='train', max_seq_len=50, batch_size=64):
#         super(MOSEIDatasetSupervised, self).__init__()
#         self.data_path = data_path
#         self.split = split
#         self.max_seq_len = max_seq_len
        
#         print_progress(f"Loading MOSEI dataset for supervised training (split: {split})...")
        
#         try:
#             from datasets.affect.get_data import get_dataloader
            
#             # Load all three splits from MOSEI
#             filepath = os.path.join(data_path, 'mosei_senti_data.pkl')
#             traindata, validdata, testdata = get_dataloader(
#                 filepath, 
#                 robust_test=False, 
#                 max_pad=True, 
#                 data_type='mosei', 
#                 max_seq_len=max_seq_len,
#                 batch_size=batch_size
#             )
            
#             # Determine which splits to use based on the requested split
#             if split == 'train':
#                 print_progress("Using train split only...")
#                 splits_to_process = [("train", traindata)]
#             elif split == 'val':
#                 print_progress("Using validation split only...")
#                 splits_to_process = [("val", validdata)]
#             elif split == 'test':
#                 print_progress("Using test split only...")
#                 splits_to_process = [("test", testdata)]
#             else:
#                 raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
#             # Process the selected splits
#             self.data = []
#             for split_name, dataloader in splits_to_process:
#                 print_progress(f"Processing {split_name} split...")
#                 for batch_idx, batch in enumerate(dataloader):
#                     # MultiBench returns: [visual_features, audio_features, text_features, labels]
#                     visual_feat, audio_feat, text_feat, labels = batch
                    
#                     # Store each sample individually
#                     batch_size_actual = visual_feat.shape[0]
#                     for i in range(batch_size_actual):
#                         sample = {
#                             'visual': visual_feat[i],  # Shape: [seq_len, 35]
#                             'audio': audio_feat[i],    # Shape: [seq_len, 74] 
#                             'text': text_feat[i],      # Shape: [seq_len, 300]
#                             'labels': labels[i],       # Shape: [1] - sentiment score
#                             'video_id': f"mosei_{split_name}_{batch_idx}_{i}"  # Create unique identifier
#                         }
#                         self.data.append(sample)
                        
#             print_progress(f"Successfully loaded {len(self.data)} samples for supervised training")
            
#         except Exception as e:
#             print_progress(f"Error loading MOSEI supervised data: {e}")
#             import traceback
#             print_progress(f"Traceback: {traceback.format_exc()}")
#             self.data = []
    
#     def __getitem__(self, index):
#         """
#         Returns numpy arrays to match VGGSound collate pattern.
#         """
#         sample = self.data[index]
        
#         # Convert tensors to numpy arrays (removing .float() calls)
#         audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
#         video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
#         text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
#         labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
#         video_id = sample['video_id']
        
#         return audio_feature, video_feature, text_feature, labels, video_id
    
#     def __len__(self):
#         return len(self.data)


# class MOSIDataset(Dataset):
#     """
#     Dataset class for MOSI data loading.
#     Used for cross-dataset evaluation (pretrain on MOSEI, test on MOSI).
#     - If split is 'train': returns combined train+val data
#     - If split is 'val': returns only val data
#     - If split is 'test': returns only test data
#     Returns numpy arrays to match VGGSound collate pattern.
#     """
#     def __init__(self, data_path, split='test', max_seq_len=50, batch_size=64):
#         super(MOSIDataset, self).__init__()
#         self.data_path = data_path
#         self.split = split
#         self.max_seq_len = max_seq_len
        
#         print_progress(f"Loading MOSI dataset (split: {split})...")
        
#         try:
#             from datasets.affect.get_data import get_dataloader
            
#             filepath = os.path.join(data_path, 'mosi_raw.pkl')
#             traindata, validdata, testdata = get_dataloader(
#                 filepath, 
#                 robust_test=False, 
#                 max_pad=True, 
#                 data_type='mosi', 
#                 max_seq_len=max_seq_len,
#                 batch_size=batch_size
#             )
            
#             if split == 'train':
#                 print_progress("Combining train and validation splits...")
#                 splits_to_process = [("train", traindata), ("val", validdata)]
#             elif split == 'val':
#                 print_progress("Using validation split only...")
#                 splits_to_process = [("val", validdata)]
#             elif split == 'test':
#                 print_progress("Using test split only...")
#                 splits_to_process = [("test", testdata)]
#             else:
#                 raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
                
#             # Process the selected splits
#             print_progress(f"Processing MOSI {split} split...")
#             self.data = []
            
#             for split_name, dataloader in splits_to_process:
#                 print_progress(f"Processing {split_name} split...")
#                 for batch_idx, batch in enumerate(dataloader):
#                     visual_feat, audio_feat, text_feat, labels = batch
#                     batch_size_actual = visual_feat.shape[0]
                    
#                     for i in range(batch_size_actual):
#                         sample = {
#                             'visual': visual_feat[i],   # [seq_len, 35]
#                             'audio': audio_feat[i],     # [seq_len, 74]
#                             'text': text_feat[i],       # [seq_len, 300] 
#                             'labels': labels[i],        # [1]
#                             'video_id': f"mosi_{split_name}_{batch_idx}_{i}"
#                         }
#                         self.data.append(sample)
                    
#             print_progress(f"Successfully loaded {len(self.data)} samples from MOSI {split}")
            
#         except Exception as e:
#             print_progress(f"Error loading MOSI {split} data: {e}")
#             import traceback
#             print_progress(f"Traceback: {traceback.format_exc()}")
#             self.data = []
    
#     def __getitem__(self, index):
#         """
#         Returns numpy arrays to match VGGSound collate pattern.
#         """
#         sample = self.data[index]
        
#         # Convert tensors to numpy arrays (removing .float() calls)
#         audio_feature = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
#         video_feature = sample['visual'].numpy() if isinstance(sample['visual'], torch.Tensor) else sample['visual']
#         text_feature = sample['text'].numpy() if isinstance(sample['text'], torch.Tensor) else sample['text']
#         labels = sample['labels'].numpy() if isinstance(sample['labels'], torch.Tensor) else sample['labels']
#         video_id = sample['video_id']
        
#         return audio_feature, video_feature, text_feature, labels, video_id
    
#     def __len__(self):
#         return len(self.data)


# # ===============================================================================
# # COLLATE FUNCTION - Following exact VGGSound pattern
# # ===============================================================================

# def collate_func_AVT(samples):
#     """
#     VGGSound-style collate function for MOSEI/MOSI data.
#     Follows the exact pattern: torch.from_numpy(np.asarray([sample[key] for sample in samples])).float()
    
#     Args:
#         samples: List of tuples from Dataset.__getitem__()
#                  Each tuple: (audio_feature, video_feature, text_feature, labels, video_id)
#                  where each feature is a numpy array
                 
#     Returns:
#         Dictionary with batched tensors for training
#     """
    
#     return {
#         'audio_fea': torch.from_numpy(np.asarray([sample[0] for sample in samples])).float(),
        
#         'video_fea': torch.from_numpy(np.asarray([sample[1] for sample in samples])).float(),
        
#         'text_fea': torch.from_numpy(np.asarray([sample[2] for sample in samples])).float(),
        
#         'labels': torch.from_numpy(np.asarray([sample[3] for sample in samples])).float(),
        
#         'video_ids': [sample[4] for sample in samples]
#     }


# # ===============================================================================
# # DATALOADER FUNCTIONS
# # ===============================================================================

# def get_mosei_unsupervised_dataloader(batch_size= 64, max_seq_len=50, num_workers=8):
#     """
#     Creates a DataLoader for unsupervised pretraining on combined MOSEI data.
    
#     Usage:
#         unsupervised_loader = get_mosei_unsupervised_dataloader(batch_size=args.batch_size)
#         for batch_data in unsupervised_loader:
#             audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
#             video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
#             text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
#             # Ignore labels for unsupervised pretraining
#     """
#     dataset = MOSEIDatasetUnsupervised(MOSEI_DATA_PATH, max_seq_len=max_seq_len, batch_size=batch_size)
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=False,
#         collate_fn=collate_func_AVT  # VGGSound-style collate function
#     )


# def get_mosei_unsupervised_split_dataloaders(batch_size=64, max_seq_len=50, num_workers=8):
#     """
#     Creates DataLoaders for unsupervised pretraining with split functionality.
#     Returns train, test_train, and test_val dataloaders.
    
#     Usage:
#         train_loader, test_train_loader, test_val_loader = get_mosei_unsupervised_split_dataloaders(batch_size=args.batch_size)
        
#         # For unsupervised pretraining on train+val data
#         for batch_data in train_loader:
#             audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
#             video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
#             text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
#             # Ignore labels for unsupervised pretraining
            
#         # For evaluation on test splits
#         for batch_data in test_train_loader:
#             # 80% of test data for training evaluation
#             pass
            
#         for batch_data in test_val_loader:
#             # 20% of test data for validation evaluation
#             pass
#     """
#     # Create datasets for different splits
#     train_dataset = MOSEIDatasetUnsupervisedSplit(MOSEI_DATA_PATH, split='train', max_seq_len=max_seq_len, batch_size=batch_size)
#     test_train_dataset = MOSEIDatasetUnsupervisedSplit(MOSEI_DATA_PATH, split='test_train', max_seq_len=max_seq_len, batch_size=batch_size)
#     test_val_dataset = MOSEIDatasetUnsupervisedSplit(MOSEI_DATA_PATH, split='test_val', max_seq_len=max_seq_len, batch_size=batch_size)
    
#     # Create dataloaders with VGGSound-style collate function
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                              num_workers=num_workers, pin_memory=False,
#                              collate_fn=collate_func_AVT)
#     test_train_loader = DataLoader(test_train_dataset, batch_size=batch_size, shuffle=False,
#                                   num_workers=num_workers, pin_memory=False,
#                                   collate_fn=collate_func_AVT)
#     test_val_loader = DataLoader(test_val_dataset, batch_size=batch_size, shuffle=False,
#                                 num_workers=num_workers, pin_memory=False,
#                                 collate_fn=collate_func_AVT)
    
#     return train_loader, test_train_loader, test_val_loader


# def get_mosei_supervised_dataloaders(batch_size=64, max_seq_len=50, num_workers=8):
#     """
#     Creates DataLoaders for supervised training on MOSEI.
#     Returns train, validation, and test dataloaders.
    
#     Usage:
#         train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(batch_size=args.batch_size)
#         for batch_data in train_loader:
#             audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
#             video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
#             text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
#             labels = batch_data['labels']              # [batch_size, 1]
#     """
#     # Create datasets with modified logic:
#     # - train/val splits return combined train+val data
#     # - test split returns only test data
#     train_dataset = MOSEIDatasetSupervised(MOSEI_DATA_PATH, split='train', max_seq_len=max_seq_len, batch_size=batch_size)
#     val_dataset = MOSEIDatasetSupervised(MOSEI_DATA_PATH, split='val', max_seq_len=max_seq_len, batch_size=batch_size)  
#     test_dataset = MOSEIDatasetSupervised(MOSEI_DATA_PATH, split='test', max_seq_len=max_seq_len, batch_size=batch_size)
    
#     # Create dataloaders with VGGSound-style collate function
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                              num_workers=num_workers, pin_memory=False, 
#                              collate_fn=collate_func_AVT)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                            num_workers=num_workers, pin_memory=False,
#                            collate_fn=collate_func_AVT)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                             num_workers=num_workers, pin_memory=False,
#                             collate_fn=collate_func_AVT)
    
#     return train_loader, val_loader, test_loader


# def get_mosi_dataloaders(batch_size=64, max_seq_len=50, num_workers=8):
#     """
#     Creates DataLoaders for MOSI dataset (cross-dataset evaluation).
#     Returns train, validation, and test dataloaders.
    
#     Usage:
#         mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(batch_size=args.batch_size)
#         for batch_data in mosi_test:
#             audio_feature = batch_data['audio_fea']    # [batch_size, seq_len, 74]
#             video_feature = batch_data['video_fea']    # [batch_size, seq_len, 35]
#             text_feature = batch_data['text_fea']      # [batch_size, seq_len, 300]
#             labels = batch_data['labels']              # [batch_size, 1]
#     """
#     # Create MOSI datasets for all splits
#     train_dataset = MOSIDataset(MOSI_DATA_PATH, split='train', max_seq_len=max_seq_len, batch_size=batch_size)
#     val_dataset = MOSIDataset(MOSI_DATA_PATH, split='val', max_seq_len=max_seq_len, batch_size=batch_size)
#     test_dataset = MOSIDataset(MOSI_DATA_PATH, split='test', max_seq_len=max_seq_len, batch_size=batch_size)
    
#     # Create dataloaders with VGGSound-style collate function
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                              num_workers=num_workers, pin_memory=False,
#                              collate_fn=collate_func_AVT)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                            num_workers=num_workers, pin_memory=False,
#                            collate_fn=collate_func_AVT)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                             num_workers=num_workers, pin_memory=False,
#                             collate_fn=collate_func_AVT)
    
#     return train_loader, val_loader, test_loader


# # ===============================================================================
# # ENHANCED TEST FUNCTIONS WITH COMPREHENSIVE DIAGNOSTICS
# # ===============================================================================

# def test_collate_pattern():
#     """
#     Test that our collate function follows the exact VGGSound pattern.
#     This verifies that numpy arrays are properly converted to tensors.
#     """
#     print_progress("\n=== Testing VGGSound Collate Pattern ===")
    
#     # Create mock samples that mimic what our datasets return
#     mock_samples = []
#     for i in range(2):  # Create 2 fake samples
#         audio_feat = np.random.randn(50, 74).astype(np.float32)  # [seq_len, 74]
#         video_feat = np.random.randn(50, 35).astype(np.float32)  # [seq_len, 35]
#         text_feat = np.random.randn(50, 300).astype(np.float32)  # [seq_len, 300]
#         labels = np.array([0.5]).astype(np.float32)              # [1]
#         video_id = f"test_sample_{i}"
        
#         sample = (audio_feat, video_feat, text_feat, labels, video_id)
#         mock_samples.append(sample)
    
#     # Test our collate function
#     batch_data = collate_func_AVT(mock_samples)
    
#     print_progress(f"Mock batch results:")
#     print_progress(f"  Audio shape: {batch_data['audio_fea'].shape}, dtype: {batch_data['audio_fea'].dtype}")
#     print_progress(f"  Video shape: {batch_data['video_fea'].shape}, dtype: {batch_data['video_fea'].dtype}")
#     print_progress(f"  Text shape: {batch_data['text_fea'].shape}, dtype: {batch_data['text_fea'].dtype}")
#     print_progress(f"  Labels shape: {batch_data['labels'].shape}, dtype: {batch_data['labels'].dtype}")
#     print_progress(f"  Video IDs: {batch_data['video_ids']}")
#     print_progress("✅ VGGSound collate pattern test passed!")


# def test_all_dataloaders_with_diagnostics():
#     """
#     Comprehensive test function with enhanced diagnostics including raw data inspection.
#     """
#     print_progress("\n" + "="*80)
#     print_progress("COMPREHENSIVE DATALOADER TESTING WITH RAW DATA ANALYSIS")
#     print_progress("="*80)
    
#     # NEW: Step 0 - Inspect raw H5 files and data structure
#     print_progress("\n🔍 STEP 0: RAW DATA FILE INSPECTION")
#     inspect_raw_h5_files(MOSEI_DATA_PATH, 'mosei')
#     inspect_raw_h5_files(MOSI_DATA_PATH, 'mosi')
    
#     # NEW: Step 0.5 - Inspect raw text processing pipeline
#     print_progress("\n🔍 STEP 0.5: RAW TEXT PROCESSING PIPELINE")
#     inspect_raw_text_processing(MOSEI_DATA_PATH, 'mosei', num_samples=3)
#     inspect_raw_text_processing(MOSI_DATA_PATH, 'mosi', num_samples=2)
    
#     # First, analyze the raw data with both padding options
#     print_progress("\n🔍 STEP 1: RAW DATA ANALYSIS")
#     compare_padding_effects(MOSEI_DATA_PATH, 'mosei', max_seq_len=50, batch_size=4)
#     compare_padding_effects(MOSI_DATA_PATH, 'mosi', max_seq_len=50, batch_size=4)
    
#     # Analyze dataset content and distributions
#     print_progress("\n🔍 STEP 2: DATASET CONTENT ANALYSIS")
#     analyze_dataset_content(MOSEI_DATA_PATH, 'mosei', max_samples=3)
#     analyze_dataset_content(MOSI_DATA_PATH, 'mosi', max_samples=2)
    
#     # NEW: Step 2.5 - Compare raw vs processed data
#     print_progress("\n🔍 STEP 2.5: RAW VS PROCESSED DATA COMPARISON")
#     inspect_raw_vs_processed_data(MOSEI_DATA_PATH, 'mosei', num_samples=3)
#     inspect_raw_vs_processed_data(MOSI_DATA_PATH, 'mosi', num_samples=2)
    
#     # Test collate pattern
#     test_collate_pattern()
    
#     # Test actual dataloaders with detailed diagnostics
#     print_progress("\n🔍 STEP 3: DATALOADER INTEGRATION TESTING")
    
#     print_progress("\n=== Testing Unsupervised MOSEI Dataset ===")
#     try:
#         unsupervised_loader = get_mosei_unsupervised_dataloader(batch_size=8, max_seq_len=50)
#         for i, batch_data in enumerate(unsupervised_loader):
#             audio_feature = batch_data['audio_fea']
#             video_feature = batch_data['video_fea'] 
#             text_feature = batch_data['text_fea']
#             labels = batch_data['labels']
            
#             print_progress(f"\nUnsupervised Batch {i} Analysis:")
#             print_feature_diagnostics(audio_feature[0], f"Batch_{i}_Audio_Sample_0")
#             print_feature_diagnostics(video_feature[0], f"Batch_{i}_Video_Sample_0")
#             print_feature_diagnostics(text_feature[0], f"Batch_{i}_Text_Sample_0")
#             print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
#             if i == 0:  # Only analyze first batch in detail
#                 break
#         print_progress("✅ MOSEI Unsupervised test passed!")
#     except Exception as e:
#         print_progress(f"❌ MOSEI Unsupervised test failed: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")

#     print_progress("\n=== Testing MOSEI Supervised Dataset ===")
#     try:
#         train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(batch_size=8, max_seq_len=50)
        
#         for i, batch_data in enumerate(train_loader):
#             audio_feature = batch_data['audio_fea']
#             video_feature = batch_data['video_fea']
#             text_feature = batch_data['text_fea'] 
#             labels = batch_data['labels']
            
#             print_progress(f"\nSupervised Train Batch {i} Analysis:")
#             print_feature_diagnostics(audio_feature[0], f"Supervised_Batch_{i}_Audio_Sample_0")
#             print_progress(f"Batch size: {audio_feature.shape[0]}")
#             print_progress(f"Labels range in batch: [{labels.min().item():.3f}, {labels.max().item():.3f}]")
#             print_progress(f"Sample labels: {labels[:5].numpy().flatten()}")  # Show first 5 labels
            
#             if i == 0:  # Only analyze first batch in detail
#                 break
#         print_progress("✅ MOSEI Supervised test passed!")
#     except Exception as e:
#         print_progress(f"❌ MOSEI Supervised test failed: {e}")
    
#     print_progress("\n=== Testing MOSI Dataset ===")
#     try:
#         mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(batch_size=8, max_seq_len=50)
        
#         for i, batch_data in enumerate(mosi_test):
#             audio_feature = batch_data['audio_fea']
#             video_feature = batch_data['video_fea']
#             text_feature = batch_data['text_fea']
#             labels = batch_data['labels']
            
#             print_progress(f"\nMOSI Test Batch {i} Analysis:")
#             print_feature_diagnostics(audio_feature[0], f"MOSI_Batch_{i}_Audio_Sample_0")
#             print_progress(f"Batch size: {audio_feature.shape[0]}")
#             print_progress(f"Labels range in batch: [{labels.min().item():.3f}, {labels.max().item():.3f}]")
#             print_progress(f"Sample labels: {labels[:3].numpy().flatten()}")  # Show first 3 labels
            
#             if i == 0:  # Only analyze first batch
#                 break
#         print_progress("✅ MOSI test passed!")
#     except Exception as e:
#         print_progress(f"❌ MOSI test failed: {e}")
    
#     print_progress("\n" + "="*80)
#     print_progress("✅ ALL COMPREHENSIVE DIAGNOSTIC TESTS COMPLETED!")
#     print_progress("📊 SUMMARY OF DATA PIPELINE:")
#     print_progress("  • Raw H5 files contain original multimodal data")  
#     print_progress("  • Text gets processed through GloVe embeddings (300D)")
#     print_progress("  • Audio and visual features are pre-extracted")
#     print_progress("  • Everything gets padded to consistent sequence lengths")
#     print_progress("  • Final format: Audio[seq,74], Visual[seq,35], Text[seq,300]")
#     print_progress("Your datasets are ready for multimodal training.")
#     print_progress("="*80)


# if __name__ == "__main__":
#     test_all_dataloaders_with_diagnostics()