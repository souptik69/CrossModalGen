#!/usr/bin/env python
"""
Custom test script to thoroughly examine the text processing and query generation pipeline.
This script checks what sentences are being processed, the filtering process, and embedding dimensions.
"""

import os
import numpy as np
import torch
import pickle
import pandas as pd
from transformers import BertTokenizer, BertModel
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Test text query pipeline in detail')
    parser.add_argument('--pickle_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl',
                        help='Path to the pickle file containing id2idx mapping')
    parser.add_argument('--csv_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsoundCategories2Prompts.csv',
                        help='Path to the CSV file with categories and prompts')
    parser.add_argument('--meta_csv_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv',
                        help='Path to the main dataset CSV')
    parser.add_argument('--output_dir', type=str,
                        default='text_query_test_results',
                        help='Directory to save the test results')
    parser.add_argument('--num_samples', type=int, default=150,
                        help='Number of samples to test')
    return parser.parse_args()

def load_data(pickle_path, csv_path, meta_csv_path):
    """Load all necessary data files"""
    print("Loading data files...")
    
    # Load id2idx mapping
    try:
        with open(pickle_path, 'rb') as fp:
            id2idx = pickle.load(fp)
        print(f"✓ Loaded id2idx mapping with {len(id2idx)} entries")
    except Exception as e:
        print(f"✗ Error loading pickle file: {e}")
        return None, None, None
    
    # Load category to prompt mapping
    try:
        label2prompt = pd.read_csv(csv_path)
        print(f"✓ Loaded label2prompt CSV with {len(label2prompt)} categories")
        print(f"  Columns: {list(label2prompt.columns)}")
    except Exception as e:
        print(f"✗ Error loading CSV file: {e}")
        return None, None, None
    
    # Load main dataset
    try:
        meta_df = pd.read_csv(meta_csv_path)
        print(f"✓ Loaded main dataset with {len(meta_df)} samples")
        print(f"  Columns: {list(meta_df.columns)}")
        print(f"  Categories: {meta_df['category'].nunique()} unique")
    except Exception as e:
        print(f"✗ Error loading meta CSV file: {e}")
        return None, None, None
    
    return id2idx, label2prompt, meta_df

def analyze_id2idx_mapping(id2idx, tokenizer):
    """Analyze the id2idx mapping and its relationship with BERT tokenizer"""
    print("\n" + "="*60)
    print("ANALYZING ID2IDX MAPPING")
    print("="*60)
    
    print(f"Total entries in id2idx: {len(id2idx)}")
    
    # Check vocabulary overlap with BERT
    bert_vocab_size = tokenizer.vocab_size
    print(f"BERT vocabulary size: {bert_vocab_size}")
    
    # Sample some entries
    sample_entries = list(id2idx.items())[:20]
    print(f"\nFirst 20 entries in id2idx:")
    for bert_id, mapped_idx in sample_entries:
        try:
            token = tokenizer.convert_ids_to_tokens(bert_id)
            print(f"  BERT ID {bert_id:5d} -> Mapped ID {mapped_idx:5d} | Token: '{token}'")
        except:
            print(f"  BERT ID {bert_id:5d} -> Mapped ID {mapped_idx:5d} | Token: <INVALID>")
    
    # Check for special tokens
    special_tokens = {
        '[CLS]': tokenizer.cls_token_id,
        '[SEP]': tokenizer.sep_token_id,
        '[PAD]': tokenizer.pad_token_id,
        '[UNK]': tokenizer.unk_token_id,
        '[MASK]': tokenizer.mask_token_id
    }
    
    print(f"\nSpecial tokens in id2idx:")
    for name, token_id in special_tokens.items():
        in_mapping = token_id in id2idx
        print(f"  {name} (ID {token_id}): {'✓' if in_mapping else '✗'} in id2idx")

def process_text_sample(text, tokenizer, model, id2idx, sample_idx):
    """Process a single text sample and show detailed breakdown"""
    print(f"\n" + "-"*80)
    print(f"PROCESSING SAMPLE {sample_idx + 1}: '{text}'")
    print("-"*80)
    
    # Step 1: Tokenization
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    token_ids = inputs.input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"Original text: '{text}'")
    print(f"Tokenized into {len(tokens)} tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    
    # Step 2: Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Step 3: Remove special tokens
    non_special_tokens = tokens[1:-1]  # Remove [CLS] and [SEP]
    non_special_embeddings = embeddings[1:-1]
    non_special_ids = token_ids[1:-1]
    
    print(f"After removing special tokens: {len(non_special_tokens)} tokens")
    print(f"Non-special tokens: {non_special_tokens}")
    
    # Step 4: Apply id2idx filtering
    filtered_tokens = []
    filtered_embeddings = []
    filtered_mapped_ids = []
    
    print(f"\nFiltering process:")
    for i, (token, token_id, emb) in enumerate(zip(non_special_tokens, non_special_ids, non_special_embeddings)):
        in_id2idx = token_id in id2idx and token_id != 0
        mapped_id = id2idx.get(token_id, -1) if in_id2idx else -1
        
        print(f"  Token {i+1:2d}: '{token:15s}' | BERT ID: {token_id:5d} | In id2idx: {'✓' if in_id2idx else '✗'} | Mapped ID: {mapped_id}")
        
        if in_id2idx:
            filtered_tokens.append(token)
            filtered_embeddings.append(emb)
            filtered_mapped_ids.append(mapped_id)
    
    print(f"\nFiltering results:")
    print(f"  Original tokens: {len(non_special_tokens)}")
    print(f"  Filtered tokens: {len(filtered_tokens)}")
    print(f"  Retention rate: {len(filtered_tokens)/len(non_special_tokens)*100:.1f}%")
    print(f"  Filtered tokens: {filtered_tokens}")
    print(f"  Mapped IDs: {filtered_mapped_ids}")
    
    # Step 5: Analyze embeddings
    if len(filtered_embeddings) > 0:
        embeddings_array = np.array(filtered_embeddings)
        print(f"\nEmbedding analysis:")
        print(f"  Shape: {embeddings_array.shape}")
        print(f"  Mean: {embeddings_array.mean():.6f}")
        print(f"  Std:  {embeddings_array.std():.6f}")
        print(f"  Min:  {embeddings_array.min():.6f}")
        print(f"  Max:  {embeddings_array.max():.6f}")
        
        # Show first few dimensions of first token
        if len(filtered_embeddings) > 0:
            first_emb = filtered_embeddings[0]
            print(f"  First token '{filtered_tokens[0]}' embedding:")
            print(f"    First 10 dims: {first_emb[:10]}")
            print(f"    Last 10 dims:  {first_emb[-10:]}")
    else:
        print(f"\n⚠️  WARNING: No valid embeddings after filtering!")
        embeddings_array = np.array([])
    
    return {
        'text': text,
        'original_tokens': tokens,
        'non_special_tokens': non_special_tokens,
        'filtered_tokens': filtered_tokens,
        'filtered_embeddings': embeddings_array,
        'filtered_mapped_ids': filtered_mapped_ids,
        'retention_rate': len(filtered_tokens)/len(non_special_tokens) if len(non_special_tokens) > 0 else 0
    }

def test_collate_function(samples_data, id2idx, tokenizer, model):
    """Test the collate function with processed samples"""
    print(f"\n" + "="*60)
    print("TESTING COLLATE FUNCTION")
    print("="*60)
    
    text_prompts = [sample['text'] for sample in samples_data]
    bsz = len(text_prompts)
    
    print(f"Batch size: {bsz}")
    print(f"Text prompts:")
    for i, text in enumerate(text_prompts):
        print(f"  {i+1}: '{text}'")
    
    # Replicate the collate function logic
    query = []
    query_words = []
    
    for i, text in enumerate(text_prompts):
        print(f"\nProcessing text {i+1} in collate function...")
        
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        token_ids = inputs.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        non_special_tokens = tokens[1:-1]
        non_special_embeddings = embeddings[1:-1]
        
        words = []
        words_emb = []
        
        for token, emb in zip(non_special_tokens, non_special_embeddings):
            idx = tokenizer.convert_tokens_to_ids(token)
            if idx in id2idx and idx != 0:
                words_emb.append(emb)
                words.append(id2idx[idx])
        
        print(f"  Kept {len(words)} tokens out of {len(non_special_tokens)}")
        
        query.append(np.asarray(words_emb))
        query_words.append(words)
    
    # Create tensors
    query_len = [10 for _ in range(bsz)]  # max_num_words: 10
    max_len = max(query_len)
    
    query1 = np.zeros([bsz, max_len, 768]).astype(np.float32)
    query_idx = np.zeros([bsz, max_len]).astype(np.float32)
    
    for i, sample in enumerate(query):
        keep = min(sample.shape[0], query1.shape[1])
        print(f"Sample {i+1}: {sample.shape[0]} tokens, keeping {keep}")
        
        if keep > 0:
            query1[i, :keep] = sample[:keep]
            query_idx[i, :keep] = query_words[i][:keep]
    
    query_len = np.asarray(query_len)
    query_tensor = torch.from_numpy(query1).float()
    query_idx_tensor = torch.from_numpy(query_idx).long()
    query_len_tensor = torch.from_numpy(query_len).long()
    
    print(f"\nFinal tensors:")
    print(f"  query shape: {query_tensor.shape}")
    print(f"  query_idx shape: {query_idx_tensor.shape}")
    print(f"  query_len: {query_len_tensor.tolist()}")
    
    # Analyze final query tensor
    print(f"\nQuery tensor analysis:")
    non_zero_mask = query_tensor != 0
    non_zero_count = non_zero_mask.sum().item()
    total_elements = query_tensor.numel()
    
    print(f"  Total elements: {total_elements}")
    print(f"  Non-zero elements: {non_zero_count}")
    print(f"  Non-zero percentage: {non_zero_count/total_elements*100:.2f}%")
    print(f"  Mean (all): {query_tensor.mean():.6f}")
    print(f"  Std (all):  {query_tensor.std():.6f}")
    
    if non_zero_count > 0:
        non_zero_values = query_tensor[non_zero_mask]
        print(f"  Mean (non-zero): {non_zero_values.mean():.6f}")
        print(f"  Std (non-zero):  {non_zero_values.std():.6f}")
    
    # Show first few embeddings for each sample
    print(f"\nFirst embedding of each sample:")
    for i in range(bsz):
        if query_len_tensor[i] > 0:
            first_emb = query_tensor[i, 0, :]
            non_zero_first = (first_emb != 0).sum().item()
            print(f"  Sample {i+1}: Non-zero dims: {non_zero_first}/768")
            print(f"    First 5 dims: {first_emb[:5].tolist()}")
        else:
            print(f"  Sample {i+1}: No valid embeddings")
    
    return query_tensor, query_idx_tensor, query_len_tensor

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    id2idx, label2prompt, meta_df = load_data(args.pickle_path, args.csv_path, args.meta_csv_path)
    if id2idx is None:
        print("Failed to load data. Exiting.")
        return
    
    # Initialize BERT
    print(f"\nInitializing BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print(f"✓ BERT initialized")
    
    # Analyze id2idx mapping
    analyze_id2idx_mapping(id2idx, tokenizer)
    
    # Sample some categories and their prompts
    sample_categories = meta_df['category'].value_counts().head(args.num_samples).index.tolist()
    sample_texts = []
    
    print(f"\n" + "="*60)
    print(f"SAMPLING {args.num_samples} CATEGORIES")
    print("="*60)
    
    for i, category in enumerate(sample_categories):
        try:
            prompt_row = label2prompt.loc[label2prompt['label'] == category]
            if len(prompt_row) > 0:
                prompt = prompt_row.iloc[0, 1]  # Assuming prompt is in second column
                sample_texts.append(prompt)
                print(f"{i+1:2d}. Category: '{category}' -> Prompt: '{prompt}'")
            else:
                print(f"{i+1:2d}. Category: '{category}' -> ⚠️  No prompt found")
        except Exception as e:
            print(f"{i+1:2d}. Category: '{category}' -> ✗ Error: {e}")
    
    if not sample_texts:
        print("No valid text samples found. Exiting.")
        return
    
    # Process each text sample
    start_time = time.time()
    samples_data = []
    
    for i, text in enumerate(sample_texts):
        sample_data = process_text_sample(text, tokenizer, model, id2idx, i)
        samples_data.append(sample_data)
    
    # Test collate function
    query_tensor, query_idx_tensor, query_len_tensor = test_collate_function(
        samples_data, id2idx, tokenizer, model)
    
    processing_time = time.time() - start_time
    
    # Generate summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_samples = len(samples_data)
    total_retention = sum(sample['retention_rate'] for sample in samples_data) / total_samples
    empty_samples = sum(1 for sample in samples_data if len(sample['filtered_tokens']) == 0)
    
    print(f"Total samples processed: {total_samples}")
    print(f"Average token retention rate: {total_retention*100:.1f}%")
    print(f"Samples with no valid tokens: {empty_samples}/{total_samples}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, 'detailed_text_query_analysis.txt')
    with open(output_file, 'w') as f:
        f.write("DETAILED TEXT QUERY PIPELINE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  Pickle path: {args.pickle_path}\n")
        f.write(f"  CSV path: {args.csv_path}\n")
        f.write(f"  Meta CSV path: {args.meta_csv_path}\n")
        f.write(f"  Number of samples: {args.num_samples}\n\n")
        
        f.write(f"Data loading results:\n")
        f.write(f"  id2idx entries: {len(id2idx)}\n")
        f.write(f"  Label2prompt entries: {len(label2prompt)}\n")
        f.write(f"  Meta dataset size: {len(meta_df)}\n\n")
        
        f.write(f"Processing results:\n")
        f.write(f"  Total samples: {total_samples}\n")
        f.write(f"  Average retention rate: {total_retention*100:.1f}%\n")
        f.write(f"  Empty samples: {empty_samples}\n")
        f.write(f"  Processing time: {processing_time:.2f}s\n\n")
        
        for i, sample in enumerate(samples_data):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Text: '{sample['text']}'\n")
            f.write(f"  Original tokens: {len(sample['non_special_tokens'])}\n")
            f.write(f"  Filtered tokens: {len(sample['filtered_tokens'])}\n")
            f.write(f"  Retention rate: {sample['retention_rate']*100:.1f}%\n")
            f.write(f"  Filtered tokens: {sample['filtered_tokens']}\n")
            if len(sample['filtered_embeddings']) > 0:
                f.write(f"  First embedding preview: {sample['filtered_embeddings'][0][:5]}\n")
            f.write("\n")
        
        f.write(f"Final query tensor:\n")
        f.write(f"  Shape: {query_tensor.shape}\n")
        f.write(f"  Non-zero percentage: {(query_tensor != 0).float().mean()*100:.2f}%\n")
        f.write(f"  Query lengths: {query_len_tensor.tolist()}\n")
    
    print(f"\n✓ Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()