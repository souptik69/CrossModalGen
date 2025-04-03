#!/usr/bin/env python
"""
Test script for BERT embeddings using the Transformers library.
This script follows the same structure as the bert_embedding version
for direct comparison.
"""

import os
import numpy as np
import pandas as pd
import torch
import pickle
import argparse
from transformers import BertTokenizer, BertModel

def parse_args():
    parser = argparse.ArgumentParser(description='Test Transformers BERT embeddings')
    parser.add_argument('--prompts_csv', type=str, 
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsoundCategories2Prompts.csv',
                        help='Path to the prompts CSV file')
    parser.add_argument('--pickle_path', type=str,
                        default='/project/ag-jafra/Souptik/CMG_New/CMG/cnt.pkl',
                        help='Path to the pickle file containing id2idx mapping')
    parser.add_argument('--output_dir', type=str,
                        default='transformers_test_results',
                        help='Directory to save the test results')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of sample sentences to test')
    return parser.parse_args()

def load_id2idx(pickle_path):
    """Load the id2idx mapping from pickle file"""
    try:
        with open(pickle_path, 'rb') as fp:
            id2idx = pickle.load(fp)
        return id2idx
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return {}

def get_transformers_embeddings(sentences, tokenizer, model, id2idx):
    """Get BERT embeddings using the Transformers library"""
    print(f"\nProcessing {len(sentences)} sentences with Transformers:")
    
    all_words = []
    all_embeddings = []
    all_filtered_words = []
    all_filtered_embeddings = []
    
    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: '{sentence}'")
        
        # Tokenize and get embeddings
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Get the last hidden state for each token
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Get tokens (words)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        print(f"Total tokens: {len(tokens)}")
        
        # Store original tokens and embeddings
        all_words.append(tokens)
        all_embeddings.append(embeddings)
        
        # Filter words based on id2idx
        # For transformers, we need to map the tokens to the bert_embedding vocab indices
        filtered_words = []
        filtered_embeddings = []
        
        print("Token details:")
        for j, (token, emb) in enumerate(zip(tokens, embeddings)):
            # Skip special tokens ([CLS], [SEP], etc.)
            if token in ('[CLS]', '[SEP]', '[PAD]'):
                print(f"  Token {j+1}: '{token}', Special token - skipped")
                continue
            
            # Try to get the token's ID in bert_embedding vocabulary
            # This is an approximation since the vocabularies might not match perfectly
            # In practice, you would need a proper mapping between the two vocabularies
            idx = tokenizer.convert_tokens_to_ids(token)
            is_in_id2idx = idx in id2idx and idx != 0
            
            print(f"  Token {j+1}: '{token}', transformer idx: {idx}, in id2idx: {is_in_id2idx}")
            
            # Only keep if the token would be kept in the original process
            if is_in_id2idx:
                filtered_words.append(token)
                filtered_embeddings.append(emb)
        
        all_filtered_words.append(filtered_words)
        all_filtered_embeddings.append(filtered_embeddings)
        
        print(f"Filtered tokens: {len(filtered_words)} out of {len(tokens)}")
    
    return all_words, all_embeddings, all_filtered_words, all_filtered_embeddings

def format_for_collate(filtered_words, filtered_embeddings, id2idx, tokenizer):
    """Format the embeddings as they would be in the collate function"""
    bsz = len(filtered_words)
    
    # Convert words to id2idx indices
    query_words = []
    for sentence_words in filtered_words:
        words = []
        for word in sentence_words:
            idx = tokenizer.convert_tokens_to_ids(word)
            if idx in id2idx and idx != 0:
                words.append(id2idx[idx])
            else:
                # For comparison purposes, we'll just use 0 if not found
                words.append(0)
        query_words.append(words)
    
    # Convert embeddings to numpy arrays
    query_embeddings = [np.array(embs) for embs in filtered_embeddings]
    
    # Calculate query lengths (same as in the original script)
    query_len = []
    for sample in query_embeddings:
        query_len.append(min(len(sample), 10))
    
    # Create query tensors
    max_len = max(query_len) if query_len else 0
    query = np.zeros([bsz, max_len, 768]).astype(np.float32)
    query_idx = np.zeros([bsz, max_len]).astype(np.float32)
    
    for i, sample in enumerate(query_embeddings):
        if len(sample) > 0:  # Only process if we have embeddings
            keep = min(sample.shape[0], query.shape[1])
            query[i, :keep] = sample[:keep]
            query_idx[i, :keep] = query_words[i][:keep]
    
    query_len = np.asarray(query_len)
    
    return query, query_idx, query_len

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts CSV
    try:
        prompts_df = pd.read_csv(args.prompts_csv)
        sample_sentences = prompts_df['prompt'].sample(n=args.num_samples).tolist()
        
        print(f"Selected {args.num_samples} sample sentences:")
        for i, sentence in enumerate(sample_sentences):
            print(f"{i+1}. {sentence}")
    except Exception as e:
        print(f"Error loading prompts CSV: {e}")
        return
    
    # Load id2idx mapping
    id2idx = load_id2idx(args.pickle_path)
    print(f"Loaded id2idx mapping with {len(id2idx)} entries")
    
    # Initialize tokenizer and model from Transformers
    print("Initializing Transformers BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Get BERT embeddings
    all_words, all_embeddings, filtered_words, filtered_embeddings = get_transformers_embeddings(
        sample_sentences, tokenizer, model, id2idx)
    
    # Format for collate function
    query, query_idx, query_len = format_for_collate(
        filtered_words, filtered_embeddings, id2idx, tokenizer)
    
    # Save results
    with open(os.path.join(args.output_dir, 'transformers_test_results.txt'), 'w') as f:
        f.write("Transformers BERT Embedding Test Results\n")
        f.write("====================================\n\n")
        
        for i, sentence in enumerate(sample_sentences):
            f.write(f"Sentence {i+1}: {sentence}\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"Original tokens: {len(all_words[i])}\n")
            f.write(f"Tokens: {all_words[i]}\n\n")
            
            f.write(f"Filtered tokens: {len(filtered_words[i])}\n")
            f.write(f"Filtered tokens: {filtered_words[i]}\n\n")
            
            if len(filtered_embeddings[i]) > 0:
                f.write(f"Embedding shape: {filtered_embeddings[i][0].shape}\n")
                f.write(f"First token embedding (first 10 values): {filtered_embeddings[i][0][:10]}\n\n")
            else:
                f.write("No valid embeddings for this sentence\n\n")
            
            f.write("\n")
        
        f.write("Collate Function Output\n")
        f.write("======================\n\n")
        
        f.write(f"query shape: {query.shape}\n")
        f.write(f"query_idx shape: {query_idx.shape}\n")
        f.write(f"query_len: {query_len.tolist()}\n\n")
        
        for i in range(len(sample_sentences)):
            f.write(f"Sentence {i+1} query_len: {query_len[i]}\n")
            if query_len[i] > 0:
                f.write(f"First token embedding in query (first 10 values): {query[i, 0, :10]}\n")
            f.write("\n")
    
    print(f"Results saved to {os.path.join(args.output_dir, 'transformers_test_results.txt')}")
    
    # Also save the raw data
    torch.save({
        'sentences': sample_sentences,
        'all_words': all_words,
        'filtered_words': filtered_words,
        'query': torch.from_numpy(query).float(),
        'query_idx': torch.from_numpy(query_idx).long(),
        'query_len': torch.from_numpy(query_len).long()
    }, os.path.join(args.output_dir, 'transformers_embeddings_data.pt'))

if __name__ == "__main__":
    main()