#!/usr/bin/env python
"""
This script tests Transformers embeddings for three specific sentences
and compares the results with the bert_embedding results.
"""

import os
import numpy as np
import torch
import pickle
from transformers import BertTokenizer, BertModel
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Test specific sentences with Transformers')
    parser.add_argument('--pickle_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl',
                        help='Path to the pickle file containing id2idx mapping')
    parser.add_argument('--output_dir', type=str,
                        default='specific_transformers_results',
                        help='Directory to save the test results')
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
        
        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        print(f"Tokenized into: {tokens}")
        
        # Convert tokens to IDs and get embeddings
        inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # Get the last hidden state for each token
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Convert IDs back to tokens to see what we got
        token_ids = inputs.input_ids[0].tolist()
        tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
        
        print(f"Total tokens (including special): {len(tokens_with_special)}")
        print(f"Tokens with special: {tokens_with_special}")
        
        # Store original tokens and embeddings (excluding special tokens)
        # [CLS] is at the beginning and [SEP] is at the end
        non_special_tokens = tokens_with_special[1:-1]  # Remove [CLS] and [SEP]
        non_special_embeddings = embeddings[1:-1]       # Remove [CLS] and [SEP] embeddings
        
        all_words.append(non_special_tokens)
        all_embeddings.append(non_special_embeddings)
        
        # Filter words based on id2idx
        filtered_words = []
        filtered_embeddings = []
        
        print("Token details:")
        for j, (token, emb) in enumerate(zip(non_special_tokens, non_special_embeddings)):
            # Try to find an equivalent token in id2idx
            # This is an approximation as the tokenizers might not match perfectly
            idx = tokenizer.convert_tokens_to_ids(token)
            is_in_id2idx = idx in id2idx and idx != 0
            
            print(f"  Token {j+1}: '{token}', transformers idx: {idx}, in id2idx: {is_in_id2idx}")
            
            # Check which tokens would be filtered by the original method
            if is_in_id2idx:
                filtered_words.append(token)
                filtered_embeddings.append(emb)
        
        all_filtered_words.append(filtered_words)
        all_filtered_embeddings.append(filtered_embeddings)
        
        print(f"Filtered tokens: {len(filtered_words)} out of {len(non_special_tokens)}")
    
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
    
    # Calculate query lengths
    query_len = []
    for sample in filtered_embeddings:
        query_len.append(min(len(sample), 10))
    
    # Create query tensors
    max_len = max(query_len) if query_len else 0
    query = np.zeros([bsz, max_len, 768]).astype(np.float32)
    query_idx = np.zeros([bsz, max_len]).astype(np.float32)
    
    for i, sample in enumerate(filtered_embeddings):
        if len(sample) > 0:
            keep = min(len(sample), query.shape[1])
            query[i, :keep] = sample[:keep]
            query_idx[i, :keep] = query_words[i][:keep]
    
    query_len = np.asarray(query_len)
    
    return query, query_idx, query_len

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Specific sentences from your results
    specific_sentences = [
        "The cooing and fluttering sound of a pigeon",
        "The clacking and typing sound of a typewriter",
        "The ringing and percussive sound of a marimba being played"
    ]
    
    # Load id2idx mapping
    id2idx = load_id2idx(args.pickle_path)
    print(f"Loaded id2idx mapping with {len(id2idx)} entries")
    
    # Initialize Transformers
    print("Initializing Transformers BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Time the process
    start_time = time.time()
    
    # Get BERT embeddings
    all_words, all_embeddings, filtered_words, filtered_embeddings = get_transformers_embeddings(
        specific_sentences, tokenizer, model, id2idx)
    
    # Format for collate function
    query, query_idx, query_len = format_for_collate(
        filtered_words, filtered_embeddings, id2idx, tokenizer)
    
    processing_time = time.time() - start_time
    print(f"\nTotal processing time: {processing_time:.4f} seconds")
    
    # Save results
    with open(os.path.join(args.output_dir, 'specific_transformers_results.txt'), 'w') as f:
        f.write("Transformers BERT Embedding Results for Specific Sentences\n")
        f.write("=====================================================\n\n")
        
        for i, sentence in enumerate(specific_sentences):
            f.write(f"Sentence {i+1}: {sentence}\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"Original tokens (excluding special): {len(all_words[i])}\n")
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
        
        for i in range(len(specific_sentences)):
            f.write(f"Sentence {i+1} query_len: {query_len[i]}\n")
            if query_len[i] > 0:
                f.write(f"First token embedding in query (first 10 values): {query[i, 0, :10]}\n")
            f.write("\n")
        
        f.write(f"\nTotal processing time: {processing_time:.4f} seconds")
    
    print(f"Results saved to {os.path.join(args.output_dir, 'specific_transformers_results.txt')}")
    
    # Also save the raw data
    torch.save({
        'sentences': specific_sentences,
        'all_words': all_words,
        'filtered_words': filtered_words,
        'query': torch.from_numpy(query).float(),
        'query_idx': torch.from_numpy(query_idx).long(),
        'query_len': torch.from_numpy(query_len).long()
    }, os.path.join(args.output_dir, 'specific_transformers_data.pt'))

if __name__ == "__main__":
    main()