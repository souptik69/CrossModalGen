#!/usr/bin/env python
"""
Script to get BERT embeddings for specific sentences using the transformers library
and save them in a format compatible for cross-version comparison.
"""

import os
import numpy as np
import torch
import pickle
import json
from transformers import BertTokenizer, BertModel
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Transformers BERT embeddings for specific sentences')
    parser.add_argument('--pickle_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl',
                        help='Path to the pickle file containing id2idx mapping')
    parser.add_argument('--output_dir', type=str,
                        default='transformers_specific_embeddings',
                        help='Directory to save the embeddings')
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
    
    all_data = []
    
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
        
        # Store original tokens and embeddings (excluding special tokens)
        # [CLS] is at the beginning and [SEP] is at the end
        non_special_tokens = tokens_with_special[1:-1]  # Remove [CLS] and [SEP]
        non_special_embeddings = embeddings[1:-1]       # Remove [CLS] and [SEP] embeddings
        
        # Filter words based on id2idx
        filtered_words = []
        filtered_embeddings = []
        filtered_indices = []
        
        print("Token details:")
        for j, (token, emb) in enumerate(zip(non_special_tokens, non_special_embeddings)):
            # Try to find an equivalent token in id2idx
            idx = tokenizer.convert_tokens_to_ids(token)
            is_in_id2idx = idx in id2idx and idx != 0
            
            print(f"  Token {j+1}: '{token}', transformers idx: {idx}, in id2idx: {is_in_id2idx}")
            
            if is_in_id2idx:
                filtered_words.append(token)
                filtered_embeddings.append(emb)
                filtered_indices.append(id2idx[idx])
        
        print(f"Filtered tokens: {len(filtered_words)} out of {len(non_special_tokens)}")
        
        sentence_data = {
            'sentence': sentence,
            'words': non_special_tokens,
            'filtered_words': filtered_words,
            'embeddings': non_special_embeddings.tolist(),
            'filtered_embeddings': [emb.tolist() for emb in filtered_embeddings],
            'filtered_indices': filtered_indices
        }
        
        all_data.append(sentence_data)
    
    return all_data

def save_embeddings(all_data, output_dir):
    """Save embeddings in multiple formats for cross-version compatibility"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON (most compatible across Python versions)
    with open(os.path.join(output_dir, 'embeddings.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_compatible_data = []
        for item in all_data:
            json_item = {
                'sentence': item['sentence'],
                'words': item['words'],
                'filtered_words': item['filtered_words'],
                'embeddings': item['embeddings'],
                'filtered_embeddings': item['filtered_embeddings'],
                'filtered_indices': item['filtered_indices']
            }
            json_compatible_data.append(json_item)
        json.dump(json_compatible_data, f)
    
    # Save as NumPy arrays (also compatible)
    for i, data in enumerate(all_data):
        sentence_name = f"sentence_{i+1}"
        np.save(os.path.join(output_dir, f"{sentence_name}_embeddings.npy"), 
                np.array(data['embeddings']))
        np.save(os.path.join(output_dir, f"{sentence_name}_filtered_embeddings.npy"), 
                np.array(data['filtered_embeddings']))
    
    # Save individual sentence data as pickle
    for i, data in enumerate(all_data):
        with open(os.path.join(output_dir, f"sentence_{i+1}_data.pkl"), 'wb') as f:
            pickle.dump(data, f, protocol=2)  # Use protocol 2 for better compatibility
    
    # Save full data as pickle with protocol 2 (compatible with Python 2.3+)
    with open(os.path.join(output_dir, 'all_embeddings_py2.pkl'), 'wb') as f:
        pickle.dump(all_data, f, protocol=2)
    
    # Also save with default protocol
    with open(os.path.join(output_dir, 'all_embeddings.pkl'), 'wb') as f:
        pickle.dump(all_data, f)
    
    print(f"Embeddings saved to {output_dir} in multiple formats:")
    print(f"- JSON: embeddings.json")
    print(f"- NumPy: sentence_X_embeddings.npy, sentence_X_filtered_embeddings.npy")
    print(f"- Pickle: sentence_X_data.pkl, all_embeddings_py2.pkl, all_embeddings.pkl")

def main():
    args = parse_args()
    
    # Define specific sentences
    specific_sentences = [
        "The cooing and fluttering sound of a pigeon",
        "The clacking and typing sound of a typewriter",
        "The ringing and percussive sound of a marimba being played"
    ]
    
    print(f"Processing {len(specific_sentences)} specific sentences:")
    for i, sentence in enumerate(specific_sentences):
        print(f"{i+1}. {sentence}")
    
    # Load id2idx mapping
    id2idx = load_id2idx(args.pickle_path)
    print(f"Loaded id2idx mapping with {len(id2idx)} entries")
    
    # Initialize Transformers
    print("Initializing Transformers BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Get BERT embeddings
    start_time = time.time()
    all_data = get_transformers_embeddings(specific_sentences, tokenizer, model, id2idx)
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save embeddings in multiple formats
    save_embeddings(all_data, args.output_dir)
    
    # Save summary information
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write("Transformers BERT Embedding Results for Specific Sentences\n")
        f.write("=====================================================\n\n")
        
        for i, data in enumerate(all_data):
            f.write(f"Sentence {i+1}: {data['sentence']}\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"Original tokens (excluding special): {len(data['words'])}\n")
            f.write(f"Tokens: {data['words']}\n\n")
            
            f.write(f"Filtered tokens: {len(data['filtered_words'])}\n")
            f.write(f"Filtered tokens: {data['filtered_words']}\n\n")
            
            if len(data['filtered_embeddings']) > 0:
                f.write(f"Embedding shape: {np.array(data['filtered_embeddings'][0]).shape}\n")
                f.write(f"First token embedding (first 10 values): {data['filtered_embeddings'][0][:10]}\n\n")
            else:
                f.write("No valid embeddings for this sentence\n\n")
            
            f.write("\n")
        
        f.write("Processing Information\n")
        f.write("=====================\n\n")
        f.write(f"Total processing time: {processing_time:.4f} seconds\n")
    
    print(f"Summary saved to {os.path.join(args.output_dir, 'summary.txt')}")
    print(f"Total processing time: {processing_time:.4f} seconds")

if __name__ == "__main__":
    main()