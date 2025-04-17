#!/usr/bin/env python
"""
This script tests Transformers embeddings for three specific sentences
without filtering by id2idx to show the complete embeddings.
"""

import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Test specific sentences with Transformers without filtering')
    parser.add_argument('--output_dir', type=str,
                        default='embeddings_unfiltered',
                        help='Directory to save the test results')
    return parser.parse_args()

def get_transformers_embeddings(sentences, tokenizer, model):
    """Get BERT embeddings using the Transformers library without filtering"""
    print(f"\nProcessing {len(sentences)} sentences with Transformers (no filtering):")
    
    all_words = []
    all_embeddings = []
    
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
        
        # Store tokens and embeddings (excluding special tokens)
        # [CLS] is at the beginning and [SEP] is at the end
        non_special_tokens = tokens_with_special[1:-1]  # Remove [CLS] and [SEP]
        non_special_embeddings = embeddings[1:-1]       # Remove [CLS] and [SEP] embeddings
        
        all_words.append(non_special_tokens)
        all_embeddings.append(non_special_embeddings)
        
        print(f"Number of tokens (excluding special): {len(non_special_tokens)}")
        print("Token details:")
        for j, (token, emb) in enumerate(zip(non_special_tokens, non_special_embeddings)):
            # Get the token ID from the tokenizer
            idx = tokenizer.convert_tokens_to_ids(token)
            print(f"  Token {j+1}: '{token}', transformers idx: {idx}, embedding shape: {emb.shape}")
    
    return all_words, all_embeddings

def format_for_collate(all_words, all_embeddings, tokenizer):
    """Format the embeddings as they would be in the collate function without filtering"""
    bsz = len(all_words)
    
    # Convert words to tokenizer indices
    query_words = []
    for sentence_words in all_words:
        words = []
        for word in sentence_words:
            idx = tokenizer.convert_tokens_to_ids(word)
            words.append(idx)
        query_words.append(words)
    
    # Calculate query lengths (up to 10 tokens per sentence)
    query_len = []
    for sample in all_embeddings:
        query_len.append(min(len(sample), 10))
    
    # Create query tensors
    query = np.zeros([bsz, 10, 768]).astype(np.float32)  # Fixed to 10 tokens max
    query_idx = np.zeros([bsz, 10]).astype(np.float32)
    
    for i, sample in enumerate(all_embeddings):
        if len(sample) > 0:
            keep = min(len(sample), 10)
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
    
    # Initialize Transformers
    print("Initializing Transformers BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Time the process
    start_time = time.time()
    
    # Get BERT embeddings without filtering
    all_words, all_embeddings = get_transformers_embeddings(
        specific_sentences, tokenizer, model)
    
    # Format for collate function
    query, query_idx, query_len = format_for_collate(
        all_words, all_embeddings, tokenizer)
    
    processing_time = time.time() - start_time
    print(f"\nTotal processing time: {processing_time:.4f} seconds")
    
    # Save results
    with open(os.path.join(args.output_dir, 'unfiltered_embeddings_results.txt'), 'w') as f:
        f.write("Transformers BERT Embedding Results for Specific Sentences (Unfiltered)\n")
        f.write("================================================================\n\n")
        
        for i, sentence in enumerate(specific_sentences):
            f.write(f"Sentence {i+1}: {sentence}\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"Tokens (excluding special): {len(all_words[i])}\n")
            f.write(f"Tokens: {all_words[i]}\n\n")
            
            if len(all_embeddings[i]) > 0:
                f.write(f"Embedding shape for first token: {all_embeddings[i][0].shape}\n")
                f.write(f"Complete embedding shape: ({len(all_embeddings[i])}, {all_embeddings[i][0].shape[0]})\n")
                f.write(f"First token embedding (first 10 values): {all_embeddings[i][0][:10]}\n\n")
            
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
    
    print(f"Results saved to {os.path.join(args.output_dir, 'unfiltered_embeddings_results.txt')}")
    
    # Also save the raw data
    torch.save({
        'sentences': specific_sentences,
        'all_words': all_words,
        'query': torch.from_numpy(query).float(),
        'query_idx': torch.from_numpy(query_idx).long(),
        'query_len': torch.from_numpy(query_len).long()
    }, os.path.join(args.output_dir, 'unfiltered_embeddings_data.pt'))

if __name__ == "__main__":
    main()