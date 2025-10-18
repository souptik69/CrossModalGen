import pandas as pd
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Load NEW cnt.pkl
with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)

print("=" * 80)
print("TESTING NEW CNT.PKL WITH transformers LIBRARY")
print("=" * 80)
print(f"\nTotal tokens in NEW cnt.pkl: {len(id2idx)}")

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# CSV files to test
csv_files = [
    ('/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsoundCategories2Prompts.csv', 'VGGSound'),
    ('/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVSBench/data/AVSBenchCategories2Prompts.csv', 'AVSBench')
]

# Output file
output_file = '/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/test_new_cnt_transformers.txt'

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("TESTING NEW CNT.PKL WITH transformers LIBRARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total tokens in NEW cnt.pkl: {len(id2idx)}\n\n")
    
    for csv_path, dataset_name in csv_files:
        print(f"\n{'=' * 80}")
        print(f"Testing {dataset_name} dataset...")
        print(f"{'=' * 80}")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{dataset_name} DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        total_words_kept = 0
        total_words_filtered = 0
        prompts_with_filtered_words = 0
        
        for idx, row in df.iterrows():
            label = row['label']
            prompt = row['prompt']
            
            # Tokenize and get embeddings
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0).numpy()
            
            token_ids = inputs.input_ids[0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            
            # Remove special tokens [CLS] and [SEP]
            non_special_tokens = tokens[1:-1]
            non_special_embeddings = embeddings[1:-1]
            
            words_kept = []
            words_filtered = []
            
            for token in non_special_tokens:
                bert_idx = tokenizer.convert_tokens_to_ids(token)
                
                if bert_idx in id2idx and bert_idx != 0:
                    words_kept.append(token)
                    total_words_kept += 1
                else:
                    words_filtered.append(token)
                    total_words_filtered += 1
            
            if words_filtered:
                prompts_with_filtered_words += 1
            
            # Write to file
            f.write(f"\nLabel: {label}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Total tokens: {len(non_special_tokens)}\n")
            f.write(f"Words KEPT: {len(words_kept)} - {', '.join(words_kept)}\n")
            if words_filtered:
                f.write(f"Words FILTERED: {len(words_filtered)} - {', '.join(words_filtered)}\n")
            else:
                f.write(f"Words FILTERED: 0 - ALL TOKENS KEPT! ✓\n")
            f.write("-" * 80 + "\n")
            
            # Print to console
            status = "✓ ALL KEPT" if not words_filtered else f"✗ {len(words_filtered)} filtered"
            print(f"Label: {label[:40]:<40} | Kept: {len(words_kept):3d} | Filtered: {len(words_filtered):3d} | {status}")
        
        # Summary for this dataset
        total_tokens = total_words_kept + total_words_filtered
        keep_ratio = (total_words_kept / total_tokens * 100) if total_tokens > 0 else 0
        
        summary = f"\n{dataset_name} SUMMARY:\n"
        summary += f"Total prompts: {len(df)}\n"
        summary += f"Prompts with filtered words: {prompts_with_filtered_words}\n"
        summary += f"Total words kept: {total_words_kept}\n"
        summary += f"Total words filtered: {total_words_filtered}\n"
        summary += f"Keep ratio: {keep_ratio:.2f}%\n"
        
        if total_words_filtered == 0:
            summary += "\n🎉 SUCCESS! ALL TOKENS FROM ALL PROMPTS ARE KEPT!\n"
        
        f.write("\n" + summary)
        print(f"\n{summary}")

print(f"\n\nTest complete! Results saved to: {output_file}")