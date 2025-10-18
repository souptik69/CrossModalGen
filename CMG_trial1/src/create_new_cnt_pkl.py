import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel
from collections import OrderedDict

print("=" * 80)
print("CREATING NEW CNT.PKL WITH ALL PROMPT TOKENS")
print("=" * 80)

# Load existing cnt.pkl to see its structure
with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl', 'rb') as fp:
    old_id2idx = pickle.load(fp)

print(f"\nOld cnt.pkl entries: {len(old_id2idx)}")
print(f"Sample old entries: {list(old_id2idx.items())[:10]}")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# CSV files to process
csv_files = [
    '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsoundCategories2Prompts.csv',
    '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVSBench/data/AVSBenchCategories2Prompts.csv'
]

# Collect all unique token IDs from prompts
all_token_ids = set()

print("\nCollecting tokens from CSV files...")

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    dataset_name = 'VGGSound' if 'vggsound' in csv_path else 'AVSBench'
    
    print(f"\nProcessing {dataset_name}...")
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        token_ids = inputs.input_ids[0].tolist()
        
        # Remove special tokens [CLS] and [SEP]
        non_special_token_ids = token_ids[1:-1]
        
        # Add to set
        all_token_ids.update(non_special_token_ids)
    
    print(f"  Total unique tokens so far: {len(all_token_ids)}")

# Create new id2idx mapping
# Start from existing mappings, then add new tokens
new_id2idx = dict(old_id2idx)  # Copy existing mappings

# Find the maximum index in old mapping
if new_id2idx:
    max_idx = max(new_id2idx.values())
else:
    max_idx = -1

# Add new tokens that weren't in the old mapping
new_tokens_added = 0
for token_id in sorted(all_token_ids):
    if token_id not in new_id2idx and token_id != 0:  # Keep excluding token ID 0
        max_idx += 1
        new_id2idx[token_id] = max_idx
        new_tokens_added += 1

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Old cnt.pkl size: {len(old_id2idx)}")
print(f"New tokens added: {new_tokens_added}")
print(f"New cnt.pkl size: {len(new_id2idx)}")
print(f"Total unique token IDs from prompts: {len(all_token_ids)}")

# Save new cnt.pkl
new_pkl_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl'
with open(new_pkl_path, 'wb') as fp:
    pickle.dump(new_id2idx, fp)

print(f"\nNew cnt.pkl saved to: {new_pkl_path}")

# Save a backup of old cnt.pkl
backup_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_old_backup.pkl'
with open(backup_path, 'wb') as fp:
    pickle.dump(old_id2idx, fp)

print(f"Old cnt.pkl backed up to: {backup_path}")

# Save mapping details to text file
details_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new_details.txt'
with open(details_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NEW CNT.PKL DETAILS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Old cnt.pkl size: {len(old_id2idx)}\n")
    f.write(f"New tokens added: {new_tokens_added}\n")
    f.write(f"New cnt.pkl size: {len(new_id2idx)}\n\n")
    
    f.write("Sample token mappings (first 50):\n")
    f.write("-" * 80 + "\n")
    for i, (token_id, idx) in enumerate(sorted(new_id2idx.items())[:50]):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        f.write(f"Token ID {token_id} ('{token}') -> Index {idx}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("NEW TOKENS ADDED\n")
    f.write("=" * 80 + "\n")
    for token_id in sorted(all_token_ids):
        if token_id not in old_id2idx and token_id != 0:
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            f.write(f"Token ID {token_id} ('{token}') -> Index {new_id2idx[token_id]}\n")

print(f"Details saved to: {details_path}")
print("\nDone!")