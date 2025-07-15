# import torch
# import sys
# import os
# import pickle
# import numpy as np
# from datetime import datetime
# from transformers import BertTokenizer
# from collections import Counter

# # Paths
# MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
# MOSEI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'
# MOSI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSI/'

# sys.path.append(MULTIBENCH_PATH)

# def print_progress(message):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# def inspect_dataset(filepath, data_type, max_seq_len=10):
#     """Inspect dataset properties and structure"""
#     print_progress(f"Inspecting {data_type.upper()} dataset...")
    
#     try:
#         from datasets.affect.get_data import get_dataloader
        
#         # Load dataloaders
#         traindata, validdata, testdata = get_dataloader(
#             filepath, 
#             robust_test=False, 
#             max_pad=True, 
#             data_type=data_type, 
#             max_seq_len=max_seq_len,
#             batch_size=4  # Small batch for inspection
#         )
        
#         print_progress(f"Dataset sizes:")
#         print_progress(f"  - Train: {len(traindata.dataset)}")
#         print_progress(f"  - Valid: {len(validdata.dataset)}")
#         print_progress(f"  - Test: {len(testdata.dataset)}")
        

#         # Inspect first batch
#         for batch_idx, batch in enumerate(traindata):
#             print_progress(f"First batch structure:")
            
#             if isinstance(batch, tuple):
#                 print_progress(f"  Batch is tuple with {len(batch)} elements:")
#                 for i, element in enumerate(batch):
#                     if isinstance(element, torch.Tensor):
#                         print_progress(f"  - Element {i}: {element.shape} (dtype: {element.dtype})")
#                     elif isinstance(element, dict):
#                         print_progress(f"  - Element {i} (dict):")
#                         for key, value in element.items():
#                             if isinstance(value, torch.Tensor):
#                                 print_progress(f"    - {key}: {value.shape} (dtype: {value.dtype})")
#                             else:
#                                 print_progress(f"    - {key}: {type(value)}")
#                     else:
#                         print_progress(f"  - Element {i}: {type(element)}")
#             elif isinstance(batch, dict):
#                 for key, value in batch.items():
#                     if isinstance(value, torch.Tensor):
#                         print_progress(f"  - {key}: {value.shape} (dtype: {value.dtype})")
#                     else:
#                         print_progress(f"  - {key}: {type(value)}")
#             else:
#                 print_progress(f"  Batch type: {type(batch)}")
            
#             # Check individual sample
#             sample = traindata.dataset[0]
#             print_progress(f"Individual sample structure:")
#             if isinstance(sample, tuple):
#                 print_progress(f"  Sample is tuple with {len(sample)} elements:")
#                 for i, element in enumerate(sample):
#                     if isinstance(element, np.ndarray):
#                         print_progress(f"  - Element {i}: {element.shape} (dtype: {element.dtype})")
#                     elif isinstance(element, str):
#                         print_progress(f"  - Element {i}: '{element[:100]}...' (text length: {len(element)})")
#                     else:
#                         print_progress(f"  - Element {i}: {type(element)}")
#             elif isinstance(sample, dict):
#                 for key, value in sample.items():
#                     if isinstance(value, np.ndarray):
#                         print_progress(f"  - {key}: {value.shape} (dtype: {value.dtype})")
#                     elif isinstance(value, str):
#                         print_progress(f"  - {key}: '{value[:100]}...' (text length: {len(value)})")
#                     else:
#                         print_progress(f"  - {key}: {type(value)}")
#             else:
#                 print_progress(f"  Sample type: {type(sample)}")
#             break
            
#         return traindata, validdata, testdata
        
#     except Exception as e:
#         print_progress(f"Error inspecting {data_type}: {e}")
#         return None, None, None

# def extract_all_text(traindata, validdata, testdata):
#     """Extract all text from datasets"""
#     all_texts = []
    
#     datasets = [("train", traindata), ("valid", validdata), ("test", testdata)]
    
#     for split_name, dataloader in datasets:
#         if dataloader is None:
#             continue
            
#         print_progress(f"Extracting text from {split_name} set...")
        
#         for i in range(len(dataloader.dataset)):
#             sample = dataloader.dataset[i]
#             if 'text' in sample:
#                 text = sample['text']
#                 if isinstance(text, str):
#                     all_texts.append(text)
#                 elif isinstance(text, list):
#                     all_texts.extend(text)
                    
#     print_progress(f"Extracted {len(all_texts)} text samples")
#     return all_texts

# def create_word_index_mapping(all_texts, save_path, dataset_name):
#     """Create word index mapping using BERT tokenizer"""
#     print_progress(f"Creating word index mapping for {dataset_name}...")
    
#     # Initialize BERT tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
#     # Collect all tokens
#     all_tokens = []
#     word_counts = Counter()
    
#     for text in all_texts:
#         if isinstance(text, str):
#             # Tokenize with BERT
#             tokens = tokenizer.tokenize(text.lower())
#             all_tokens.extend(tokens)
#             word_counts.update(tokens)
    
#     print_progress(f"Found {len(word_counts)} unique tokens")
#     print_progress(f"Total tokens: {len(all_tokens)}")
    
#     # Create word to index mapping
#     # Start from 1 (0 reserved for padding/unknown)
#     word2idx = {'<PAD>': 0, '<UNK>': 1}
#     idx2word = {0: '<PAD>', 1: '<UNK>'}
    
#     # Add tokens by frequency (most frequent first)
#     for idx, (word, count) in enumerate(word_counts.most_common(), start=2):
#         word2idx[word] = idx
#         idx2word[idx] = word
    
#     # Create token ID to index mapping (for BERT compatibility)
#     token_id2idx = {}
#     for word, idx in word2idx.items():
#         if word not in ['<PAD>', '<UNK>']:
#             token_id = tokenizer.convert_tokens_to_ids(word)
#             token_id2idx[token_id] = idx
    
#     print_progress(f"Created mappings with {len(word2idx)} words")
#     print_progress(f"Token ID mapping size: {len(token_id2idx)}")
    
#     # Save mappings
#     mappings = {
#         'word2idx': word2idx,
#         'idx2word': idx2word,
#         'token_id2idx': token_id2idx,
#         'vocab_size': len(word2idx),
#         'word_counts': dict(word_counts.most_common(100))  # Save top 100 for inspection
#     }
    
#     # Save to multiple formats
#     mapping_files = [
#         ('word_mappings.pkl', mappings),
#         ('word2idx.pkl', word2idx),
#         ('idx2word.pkl', idx2word),
#         ('token_id2idx.pkl', token_id2idx),  # This is what the collate function needs
#     ]
    
#     for filename, data in mapping_files:
#         filepath = os.path.join(save_path, filename)
#         with open(filepath, 'wb') as f:
#             pickle.dump(data, f)
#         print_progress(f"Saved {filepath}")
    
#     return mappings

# def main():
#     print_progress("Starting data inspection and word index creation...")
    
#     # Create output directories
#     os.makedirs(MOSEI_DATA_PATH, exist_ok=True)
#     os.makedirs(MOSI_DATA_PATH, exist_ok=True)
    
#     # Dataset configurations
#     datasets_config = [
#         {
#             'name': 'MOSEI',
#             'filepath': os.path.join(MOSEI_DATA_PATH, 'mosei_senti_data.pkl'),
#             'data_type': 'mosei',
#             'save_path': MOSEI_DATA_PATH
#         },
#         {
#             'name': 'MOSI', 
#             'filepath': os.path.join(MOSI_DATA_PATH, 'mosi_raw.pkl'),
#             'data_type': 'mosi',
#             'save_path': MOSI_DATA_PATH
#         }
#     ]
    
#     for config in datasets_config:
#         print_progress(f"\n{'='*50}")
#         print_progress(f"Processing {config['name']} dataset")
#         print_progress(f"{'='*50}")
        
#         # Check if file exists
#         if not os.path.exists(config['filepath']):
#             print_progress(f"File not found: {config['filepath']}")
#             continue
            
#         # Inspect dataset
#         traindata, validdata, testdata = inspect_dataset(
#             config['filepath'], 
#             config['data_type']
#         )
        
#         if traindata is None:
#             print_progress(f"Failed to load {config['name']} dataset")
#             continue
            
#         # Extract text
#         all_texts = extract_all_text(traindata, validdata, testdata)
        
#         if not all_texts:
#             print_progress(f"No text found in {config['name']} dataset")
#             continue
            
#         # Create word mappings
#         mappings = create_word_index_mapping(
#             all_texts, 
#             config['save_path'], 
#             config['name']
#         )
        
#         # Print sample mappings
#         print_progress(f"Sample word mappings:")
#         sample_words = list(mappings['word2idx'].items())[:10]
#         for word, idx in sample_words:
#             print_progress(f"  '{word}' -> {idx}")
        
#         print_progress(f"Most frequent words:")
#         for word, count in list(mappings['word_counts'].items())[:10]:
#             print_progress(f"  '{word}': {count}")
    
#     print_progress("Data inspection and word index creation complete!")

# if __name__ == "__main__":
#     main()

# import torch
# import sys
# import os
# import numpy as np
# from datetime import datetime

# # Paths
# MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
# MOSEI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'
# MOSI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSI/'

# sys.path.append(MULTIBENCH_PATH)

# def print_progress(message):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# def inspect_dataset_detailed(filepath, data_type, max_seq_len=10):
#     """Detailed inspection of dataset properties and structure"""
#     print_progress(f"Inspecting {data_type.upper()} dataset in detail...")
    
#     try:
#         from datasets.affect.get_data import get_dataloader
        
#         # Load dataloaders
#         traindata, validdata, testdata = get_dataloader(
#             filepath, 
#             robust_test=False, 
#             max_pad=True, 
#             data_type=data_type, 
#             max_seq_len=max_seq_len,
#             batch_size=4  # Small batch for inspection
#         )
        
#         print_progress(f"Dataset sizes:")
#         print_progress(f"  - Train: {len(traindata.dataset)}")
#         print_progress(f"  - Valid: {len(validdata.dataset)}")
#         print_progress(f"  - Test: {len(testdata.dataset)}")
        
#         # Inspect individual samples (since they are lists)
#         print_progress(f"\n{'='*60}")
#         print_progress(f"INDIVIDUAL SAMPLE ANALYSIS")
#         print_progress(f"{'='*60}")
        
#         # Check multiple samples to understand structure
#         sample_indices = [0, 1, 2, 10, 100] if len(traindata.dataset) > 100 else [0, 1, 2]
        
#         for idx in sample_indices:
#             if idx >= len(traindata.dataset):
#                 continue
                
#             print_progress(f"\n--- Sample {idx} ---")
#             sample = traindata.dataset[idx]
            
#             print_progress(f"Sample type: {type(sample)}")
#             print_progress(f"Sample length: {len(sample)}")
            
#             # Inspect each element in the list
#             for i, element in enumerate(sample):
#                 print_progress(f"  Element {i}:")
#                 print_progress(f"    Type: {type(element)}")
                
#                 if isinstance(element, np.ndarray):
#                     print_progress(f"    Shape: {element.shape}")
#                     print_progress(f"    Dtype: {element.dtype}")
#                     print_progress(f"    Min value: {element.min():.4f}")
#                     print_progress(f"    Max value: {element.max():.4f}")
#                     print_progress(f"    Mean value: {element.mean():.4f}")
                    
#                     # Show a few actual values for small arrays
#                     if element.size <= 20:
#                         print_progress(f"    Values: {element.flatten()}")
#                     elif len(element.shape) == 1:
#                         print_progress(f"    First 5 values: {element[:5]}")
#                         print_progress(f"    Last 5 values: {element[-5:]}")
#                     elif len(element.shape) == 2:
#                         print_progress(f"    First row: {element[0]}")
#                         if element.shape[0] > 1:
#                             print_progress(f"    Last row: {element[-1]}")
                    
#                 elif isinstance(element, str):
#                     print_progress(f"    Text length: {len(element)}")
#                     print_progress(f"    Content: '{element[:200]}{'...' if len(element) > 200 else ''}'")
                    
#                 elif isinstance(element, (int, float)):
#                     print_progress(f"    Value: {element}")
                    
#                 elif isinstance(element, list):
#                     print_progress(f"    List length: {len(element)}")
#                     if len(element) > 0:
#                         print_progress(f"    First element type: {type(element[0])}")
#                         if isinstance(element[0], str):
#                             print_progress(f"    First few elements: {element[:3]}")
                        
#                 else:
#                     print_progress(f"    Content: {element}")
        
#         # Inspect batch structure
#         print_progress(f"\n{'='*60}")
#         print_progress(f"BATCH ANALYSIS")
#         print_progress(f"{'='*60}")
        
#         for batch_idx, batch in enumerate(traindata):
#             print_progress(f"Batch {batch_idx}:")
#             print_progress(f"  Batch type: {type(batch)}")
            
#             if isinstance(batch, tuple):
#                 print_progress(f"  Batch has {len(batch)} elements:")
#                 for i, element in enumerate(batch):
#                     if isinstance(element, torch.Tensor):
#                         print_progress(f"    Element {i} (Tensor):")
#                         print_progress(f"      Shape: {element.shape}")
#                         print_progress(f"      Dtype: {element.dtype}")
#                         print_progress(f"      Min: {element.min().item():.4f}")
#                         print_progress(f"      Max: {element.max().item():.4f}")
#                         print_progress(f"      Mean: {element.mean().item():.4f}")
                        
#                         # Likely meanings based on shape patterns
#                         if i == 0 and element.shape[-1] == 35:
#                             print_progress(f"      Likely: Audio features (35D)")
#                         elif i == 1 and element.shape[-1] == 74:
#                             print_progress(f"      Likely: Visual features (74D)")
#                         elif i == 2 and element.shape[-1] == 300:
#                             print_progress(f"      Likely: Text features (300D - possibly GloVe)")
#                         elif i == 3 and len(element.shape) == 2 and element.shape[-1] == 1:
#                             print_progress(f"      Likely: Sentiment label/score")
#                             print_progress(f"      Sample values: {element[:4].flatten()}")
#                     else:
#                         print_progress(f"    Element {i}: {type(element)}")
            
#             # Only process first batch for detailed inspection
#             break
            
#         # Check if there are any text strings in the raw dataset
#         print_progress(f"\n{'='*60}")
#         print_progress(f"SEARCHING FOR TEXT DATA")
#         print_progress(f"{'='*60}")
        
#         text_found = False
#         for idx in range(min(10, len(traindata.dataset))):
#             sample = traindata.dataset[idx]
#             for i, element in enumerate(sample):
#                 if isinstance(element, str):
#                     print_progress(f"Found text in sample {idx}, element {i}: '{element[:100]}...'")
#                     text_found = True
#                 elif isinstance(element, list):
#                     for j, subelement in enumerate(element):
#                         if isinstance(subelement, str):
#                             print_progress(f"Found text in sample {idx}, element {i}[{j}]: '{subelement[:100]}...'")
#                             text_found = True
#                             break
        
#         if not text_found:
#             print_progress("No raw text strings found in first 10 samples")
#             print_progress("Text might be pre-processed into numerical features")
        
#         return traindata, validdata, testdata
        
#     except Exception as e:
#         print_progress(f"Error inspecting {data_type}: {e}")
#         import traceback
#         print_progress(f"Traceback: {traceback.format_exc()}")
#         return None, None, None

# def analyze_feature_dimensions(traindata, data_type):
#     """Analyze what the different feature dimensions might represent"""
#     print_progress(f"\n{'='*60}")
#     print_progress(f"FEATURE DIMENSION ANALYSIS FOR {data_type.upper()}")
#     print_progress(f"{'='*60}")
    
#     # Get one batch to analyze
#     for batch in traindata:
#         if isinstance(batch, tuple) and len(batch) == 4:
#             audio_features, visual_features, text_features, labels = batch
            
#             print_progress(f"Audio Features Analysis:")
#             print_progress(f"  Shape: {audio_features.shape}")
#             print_progress(f"  Expected: [batch_size, sequence_length, audio_dim]")
#             print_progress(f"  Audio dim: {audio_features.shape[-1]} (likely COVAREP features)")
            
#             print_progress(f"\nVisual Features Analysis:")
#             print_progress(f"  Shape: {visual_features.shape}")
#             print_progress(f"  Expected: [batch_size, sequence_length, visual_dim]")
#             print_progress(f"  Visual dim: {visual_features.shape[-1]} (likely facial action units)")
            
#             print_progress(f"\nText Features Analysis:")
#             print_progress(f"  Shape: {text_features.shape}")
#             print_progress(f"  Expected: [batch_size, sequence_length, text_dim]")
#             print_progress(f"  Text dim: {text_features.shape[-1]} (likely GloVe embeddings)")
            
#             print_progress(f"\nLabels Analysis:")
#             print_progress(f"  Shape: {labels.shape}")
#             print_progress(f"  Label values (first 4): {labels[:4].flatten()}")
#             print_progress(f"  Min: {labels.min().item():.4f}")
#             print_progress(f"  Max: {labels.max().item():.4f}")
#             print_progress(f"  Mean: {labels.mean().item():.4f}")
            
#         break

# def main():
#     print_progress("Starting detailed data inspection...")
    
#     # Dataset configurations
#     datasets_config = [
#         {
#             'name': 'MOSEI',
#             'filepath': os.path.join(MOSEI_DATA_PATH, 'mosei_senti_data.pkl'),
#             'data_type': 'mosei',
#         },
#         {
#             'name': 'MOSI', 
#             'filepath': os.path.join(MOSI_DATA_PATH, 'mosi_raw.pkl'),
#             'data_type': 'mosi',
#         }
#     ]
    
#     for config in datasets_config:
#         print_progress(f"\n{'='*80}")
#         print_progress(f"PROCESSING {config['name']} DATASET")
#         print_progress(f"{'='*80}")
        
#         # Check if file exists
#         if not os.path.exists(config['filepath']):
#             print_progress(f"File not found: {config['filepath']}")
#             continue
            
#         # Detailed inspection
#         traindata, validdata, testdata = inspect_dataset_detailed(
#             config['filepath'], 
#             config['data_type']
#         )
        
#         if traindata is None:
#             print_progress(f"Failed to load {config['name']} dataset")
#             continue
        
#         # Analyze feature dimensions
#         analyze_feature_dimensions(traindata, config['data_type'])
    
#     print_progress("\nDetailed data inspection complete!")

# if __name__ == "__main__":
#     main()



import torch
import sys
import os
import numpy as np
from datetime import datetime

# Paths
MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
MOSEI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'
MOSI_DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSI/'

sys.path.append(MULTIBENCH_PATH)

def print_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def inspect_batch_structure(traindata, data_type):
    """Inspect batch structure and show sample values"""
    print_progress(f"\n{'='*60}")
    print_progress(f"BATCH STRUCTURE ANALYSIS FOR {data_type.upper()}")
    print_progress(f"{'='*60}")
    
    # Get first batch
    for batch in traindata:
        inputs = batch
        break
    
    print_progress(f"Batch type: {type(inputs)}")
    print_progress(f"Number of elements in batch: {len(inputs)}")
    
    # Analyze each element in the batch
    for i, element in enumerate(inputs):
        print_progress(f"\n--- Element {i} ---")
        print_progress(f"  Type: {type(element)}")
        
        if isinstance(element, torch.Tensor):
            print_progress(f"  Shape: {element.shape}")
            print_progress(f"  Dtype: {element.dtype}")
            print_progress(f"  Min: {element.min().item():.4f}")
            print_progress(f"  Max: {element.max().item():.4f}")
            print_progress(f"  Mean: {element.mean().item():.4f}")
            
            # Show sample values for small tensors or labels
            if len(element.shape) == 2 and element.shape[-1] == 1:
                print_progress(f"  Sample values (first 5): {element[:5].flatten()}")
            elif element.numel() <= 50:
                print_progress(f"  Values: {element.flatten()}")
                
        elif isinstance(element, (list, tuple)):
            print_progress(f"  Length: {len(element)}")
            for j, subelement in enumerate(element):
                print_progress(f"    Sub-element {j}:")
                print_progress(f"      Type: {type(subelement)}")
                if isinstance(subelement, torch.Tensor):
                    print_progress(f"      Shape: {subelement.shape}")
                    print_progress(f"      Dtype: {subelement.dtype}")
                    print_progress(f"      Min: {subelement.min().item():.4f}")
                    print_progress(f"      Max: {subelement.max().item():.4f}")
                    print_progress(f"      Mean: {subelement.mean().item():.4f}")
                    
                    # Show sample values for sequence lengths or small tensors
                    if len(subelement.shape) == 1:
                        print_progress(f"      Sample values: {subelement}")
        else:
            print_progress(f"  Content: {element}")
    
    return inputs

def analyze_dataset_specifics(inputs, data_type):
    """Analyze dataset-specific patterns based on notebook observations"""
    print_progress(f"\n{'='*60}")
    print_progress(f"DATASET-SPECIFIC ANALYSIS FOR {data_type.upper()}")
    print_progress(f"{'='*60}")
    
    if data_type.lower() == 'mosi':
        print_progress("MOSI Dataset Structure:")
        print_progress(f"  - Element 0 (Visual): {inputs[0].shape} - 35D features")
        print_progress(f"  - Element 1 (Audio): {inputs[1].shape} - 74D features") 
        print_progress(f"  - Element 2 (Text): {inputs[2].shape} - 300D features")
        print_progress(f"  - Element 3 (Labels): {inputs[3].shape} - Sentiment scores")
        print_progress(f"  - Sample label: {inputs[3][0].item()}")
        
    elif data_type.lower() == 'mosei':
        print_progress("MOSEI Dataset Structure:")
        print_progress(f"  - Element 0: Feature tensors")
        print_progress(f"    - Visual: {inputs[0].shape} - 35D features")
        print_progress(f"    - Audio: {inputs[1].shape} - 74D features")
        print_progress(f"    - Text: {inputs[2].shape} - 300D features")
        print_progress(f"    - Labels: {inputs[3].shape} - Sentiment scores")
        print_progress(f"  - Element 3: Labels")
        print_progress(f"    - Shape: {inputs[3].shape}")
        print_progress(f"    - Sample labels: {inputs[3][:].flatten()}")

def inspect_dataset(filepath, data_type, max_seq_len=10):
    """Main dataset inspection function"""
    print_progress(f"Inspecting {data_type.upper()} dataset...")
    
    try:
        from datasets.affect.get_data import get_dataloader
        
        # Load dataloaders
        traindata, validdata, testdata = get_dataloader(
            filepath, 
            robust_test=False, 
            max_pad=True, 
            data_type=data_type, 
            max_seq_len=max_seq_len,
            batch_size=32  # Use same batch size as notebook
        )
        
        print_progress(f"Dataset sizes:")
        print_progress(f"  - Train: {len(traindata.dataset)}")
        print_progress(f"  - Valid: {len(validdata.dataset)}")
        print_progress(f"  - Test: {len(testdata.dataset)}")
        
        # Inspect batch structure
        inputs = inspect_batch_structure(traindata, data_type)
        
        # Dataset-specific analysis
        analyze_dataset_specifics(inputs, data_type)
        
        return traindata, validdata, testdata
        
    except Exception as e:
        print_progress(f"Error inspecting {data_type}: {e}")
        import traceback
        print_progress(f"Traceback: {traceback.format_exc()}")
        return None, None, None

def main():
    print_progress("Starting focused data inspection...")
    
    # Dataset configurations
    datasets_config = [
        {
            'name': 'MOSI',
            'filepath': os.path.join(MOSI_DATA_PATH, 'mosi_raw.pkl'),
            'data_type': 'mosi',
        },
        {
            'name': 'MOSEI', 
            'filepath': os.path.join(MOSEI_DATA_PATH, 'mosei_senti_data.pkl'),
            'data_type': 'mosei',
        }
    ]
    
    for config in datasets_config:
        print_progress(f"\n{'='*80}")
        print_progress(f"PROCESSING {config['name']} DATASET")
        print_progress(f"{'='*80}")
        
        # Check if file exists
        if not os.path.exists(config['filepath']):
            print_progress(f"File not found: {config['filepath']}")
            continue
            
        # Inspect dataset
        traindata, validdata, testdata = inspect_dataset(
            config['filepath'], 
            config['data_type']
        )
        
        if traindata is None:
            print_progress(f"Failed to load {config['name']} dataset")
            continue
    
    print_progress("\nFocused data inspection complete!")

if __name__ == "__main__":
    main()