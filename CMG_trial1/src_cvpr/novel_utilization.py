# import logging
# import os
# import sys
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import pickle
# from collections import Counter
# import torch.nn.functional as F

# # Add paths
# # sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')
# from model.main_model_novel import AV_VQVAE_Encoder, AVT_VQVAE_Encoder
# from transformers import BertTokenizer, BertModel

# # Seeds
# SEED = 43
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# torch.cuda.manual_seed(SEED)

# # Load text processing utilities
# with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl', 'rb') as fp:
#     id2idx = pickle.load(fp)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# def collate_func_AV(samples):
#     return {
#         'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
#         'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
#     }

# def collate_func_AVT(samples):
#     bsz = len(samples)
#     text_prompts = [sample['text_fea'] for sample in samples]
#     query = []
#     query_words = []
    
#     for text in text_prompts:
#         inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
#         with torch.no_grad():
#             outputs = bert_model(**inputs)
#             embeddings = outputs.last_hidden_state.squeeze(0).numpy()
#         token_ids = inputs.input_ids[0].tolist()
#         tokens = tokenizer.convert_ids_to_tokens(token_ids)
#         non_special_tokens = tokens[1:-1]
#         non_special_embeddings = embeddings[1:-1]
#         words = []
#         words_emb = [] 
#         for token, emb in zip(non_special_tokens, non_special_embeddings):
#             idx = tokenizer.convert_tokens_to_ids(token)
#             if idx in id2idx and idx != 0:
#                 words_emb.append(emb)
#                 words.append(id2idx[idx])
#         query.append(np.asarray(words_emb))
#         query_words.append(words)

#     query_len = [10 for _ in query]
#     query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
#     query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
#     for i, sample in enumerate(query):
#         keep = min(sample.shape[0], query1.shape[1])
#         if keep > 0:
#             query1[i, :keep] = sample[:keep]
#             query_idx[i, :keep] = query_words[i][:keep]
    
#     query_len = np.asarray(query_len)
#     query = torch.from_numpy(query1).float()

#     return {
#         'query': query,
#         'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
#         'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
#     }

# def extract_indices_av(encoder, audio_feat, video_feat, device='cuda'):
#     """Extract codebook indices from AV encoder"""
#     encoder.eval()
    
#     with torch.no_grad():
#         audio_feat = audio_feat.to(device).double()
#         video_feat = video_feat.to(device).double()
        
#         # Get semantic features
#         video_semantic_result, _ = encoder.video_semantic_encoder(video_feat)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
#         video_semantic_result = encoder.video_self_att(video_semantic_result)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
#         audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
#         audio_semantic_result = encoder.audio_self_att(audio_semantic_result)
#         audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        
#         B, T, D = audio_semantic_result.size()
        
#         a_flat = audio_semantic_result.reshape(-1, D)
#         v_flat = video_semantic_result.reshape(-1, D)
        
#         # Extract embeddings
#         video_embedding = encoder.Cross_quantizer.embedding[:, :D]
#         audio_embedding = encoder.Cross_quantizer.embedding[:, D:]
        
#         # Compute distances
#         v_distances = torch.addmm(
#             torch.sum(video_embedding ** 2, dim=1) + torch.sum(v_flat ** 2, dim=1, keepdim=True),
#             v_flat, video_embedding.t(), alpha=-2.0, beta=1.0
#         )
        
#         a_distances = torch.addmm(
#             torch.sum(audio_embedding ** 2, dim=1) + torch.sum(a_flat ** 2, dim=1, keepdim=True),
#             a_flat, audio_embedding.t(), alpha=-2.0, beta=1.0
#         )
        
#         v_indices = torch.argmin(v_distances.double(), dim=-1).reshape(B, T)
#         a_indices = torch.argmin(a_distances.double(), dim=-1).reshape(B, T)
        
#         return v_indices, a_indices

# def extract_indices_avt(encoder, audio_feat, video_feat, text_feat, device='cuda'):
#     """Extract codebook indices from AVT encoder"""
#     encoder.eval()
    
#     with torch.no_grad():
#         audio_feat = audio_feat.to(device).double()
#         video_feat = video_feat.to(device).double()
#         text_feat = text_feat.to(device).double()
        
#         # Get semantic features
#         video_semantic_result, _ = encoder.video_semantic_encoder(video_feat)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
#         video_semantic_result = encoder.video_self_att(video_semantic_result)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
#         audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
#         audio_semantic_result = encoder.audio_self_att(audio_semantic_result)
#         audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        
#         text_semantic_result = text_feat.transpose(0, 1).contiguous()
#         text_semantic_result = encoder.text_self_att(text_semantic_result)
#         text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        
#         B, T, D = audio_semantic_result.size()
        
#         a_flat = audio_semantic_result.reshape(-1, D)
#         v_flat = video_semantic_result.reshape(-1, D)
        
#         # Extract embeddings
#         video_embedding = encoder.Cross_quantizer.embedding[:, :D]
#         audio_embedding = encoder.Cross_quantizer.embedding[:, D:2*D]
        
#         # Compute distances
#         v_distances = torch.addmm(
#             torch.sum(video_embedding ** 2, dim=1) + torch.sum(v_flat ** 2, dim=1, keepdim=True),
#             v_flat, video_embedding.t(), alpha=-2.0, beta=1.0
#         )
        
#         a_distances = torch.addmm(
#             torch.sum(audio_embedding ** 2, dim=1) + torch.sum(a_flat ** 2, dim=1, keepdim=True),
#             a_flat, audio_embedding.t(), alpha=-2.0, beta=1.0
#         )
        
#         v_indices = torch.argmin(v_distances.double(), dim=-1).reshape(B, T)
#         a_indices = torch.argmin(a_distances.double(), dim=-1).reshape(B, T)
        
#         return v_indices, a_indices

# def analyze_av_model(checkpoint_path, dataloader, n_embeddings, device, logger):
#     """Analyze AV model codebook utilization"""
#     logger.info(f"\nAnalyzing AV Model")
#     logger.info(f"Checkpoint: {checkpoint_path}")
    
#     # Load model
#     encoder = AV_VQVAE_Encoder(
#         audio_dim=128,
#         video_dim=512,
#         video_output_dim=2048,
#         n_embeddings=n_embeddings,
#         embedding_dim=256
#     )
#     encoder.double().to(device).eval()
    
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     encoder.load_state_dict(checkpoint['Encoder_parameters'])
#     logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
#     # Track indices
#     video_counter = Counter()
#     audio_counter = Counter()
    
#     for batch_data in tqdm(dataloader, desc="Processing AV"):
#         audio_feat = batch_data['audio_fea']
#         video_feat = batch_data['video_fea']
        
#         v_indices, a_indices = extract_indices_av(encoder, audio_feat, video_feat, device)
        
#         video_counter.update(v_indices.cpu().numpy().flatten().tolist())
#         audio_counter.update(a_indices.cpu().numpy().flatten().tolist())
    
#     # Convert to arrays
#     video_counts = np.zeros(n_embeddings)
#     audio_counts = np.zeros(n_embeddings)
    
#     for idx, count in video_counter.items():
#         video_counts[idx] = count
#     for idx, count in audio_counter.items():
#         audio_counts[idx] = count
    
#     logger.info(f"Video: {np.sum(video_counts > 0)}/{n_embeddings} vectors used")
#     logger.info(f"Audio: {np.sum(audio_counts > 0)}/{n_embeddings} vectors used")
    
#     return video_counts, audio_counts

# def analyze_avt_model(checkpoint_path, dataloader, n_embeddings, device, logger):
#     """Analyze AVT model codebook utilization"""
#     logger.info(f"\nAnalyzing AVT Model")
#     logger.info(f"Checkpoint: {checkpoint_path}")
    
#     # Load models
#     encoder = AVT_VQVAE_Encoder(
#         audio_dim=128,
#         video_dim=512,
#         text_dim=256,
#         video_output_dim=2048,
#         n_embeddings=n_embeddings,
#         embedding_dim=256
#     )
#     text_lstm = torch.nn.LSTM(768, 128, num_layers=2, batch_first=True, bidirectional=True)
    
#     encoder.double().to(device).eval()
#     text_lstm.double().to(device).eval()
    
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     encoder.load_state_dict(checkpoint['Encoder_parameters'])
#     text_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
#     logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
#     # Track indices
#     video_counter = Counter()
#     audio_counter = Counter()
    
#     for batch_data in tqdm(dataloader, desc="Processing AVT"):
#         audio_feat = batch_data['audio_fea']
#         video_feat = batch_data['video_fea']
#         query = batch_data['query'].double().to(device)
        
#         batch_dim = query.size()[0]
#         text_hidden = (
#             torch.zeros(4, batch_dim, 128).double().to(device),
#             torch.zeros(4, batch_dim, 128).double().to(device)
#         )
#         text_feat, _ = text_lstm(query, text_hidden)
        
#         v_indices, a_indices = extract_indices_avt(encoder, audio_feat, video_feat, text_feat, device)
        
#         video_counter.update(v_indices.cpu().numpy().flatten().tolist())
#         audio_counter.update(a_indices.cpu().numpy().flatten().tolist())
    
#     # Convert to arrays
#     video_counts = np.zeros(n_embeddings)
#     audio_counts = np.zeros(n_embeddings)
    
#     for idx, count in video_counter.items():
#         video_counts[idx] = count
#     for idx, count in audio_counter.items():
#         audio_counts[idx] = count
    
#     logger.info(f"Video: {np.sum(video_counts > 0)}/{n_embeddings} vectors used")
#     logger.info(f"Audio: {np.sum(audio_counts > 0)}/{n_embeddings} vectors used")
    
#     return video_counts, audio_counts

# def create_visualizations(av_video, av_audio, avt_video, avt_audio, n_embeddings, output_dir):
#     """Create simple comparison plots"""
    
#     plt.style.use('seaborn-v0_8-darkgrid')
    
#     # AV Model Visualization
#     fig = plt.figure(figsize=(16, 10))
    
#     # AV Histogram
#     ax1 = plt.subplot(2, 2, 1)
#     bins = np.linspace(0, max(av_video.max(), av_audio.max()), 50)
#     ax1.hist(av_video[av_video > 0], bins=bins, alpha=0.6, label='Video', color='blue', edgecolor='black')
#     ax1.hist(av_audio[av_audio > 0], bins=bins, alpha=0.6, label='Audio', color='red', edgecolor='black')
#     ax1.set_xlabel('Usage Count', fontsize=12)
#     ax1.set_ylabel('Number of Codebook Vectors', fontsize=12)
#     ax1.set_title('AV Model: Codebook Utilization Histogram', fontsize=14, fontweight='bold')
#     ax1.legend(fontsize=11)
#     ax1.grid(True, alpha=0.3)
    
#     # AV Heatmap
#     rows = int(np.sqrt(n_embeddings))
#     cols = n_embeddings // rows
    
#     ax2 = plt.subplot(2, 2, 2)
#     combined_av = np.stack([av_video[:rows*cols].reshape(rows, cols), 
#                             av_audio[:rows*cols].reshape(rows, cols)])
#     im = ax2.imshow(combined_av.mean(axis=0), cmap='YlOrRd', aspect='auto')
#     ax2.set_title('AV Model: Codebook Utilization Heatmap', fontsize=14, fontweight='bold')
#     ax2.set_xlabel('Codebook Vector Index (grouped)', fontsize=12)
#     ax2.set_ylabel('Codebook Vector Index (grouped)', fontsize=12)
#     plt.colorbar(im, ax=ax2, label='Average Usage Count')
    
#     # AVT Histogram
#     ax3 = plt.subplot(2, 2, 3)
#     bins = np.linspace(0, max(avt_video.max(), avt_audio.max()), 50)
#     ax3.hist(avt_video[avt_video > 0], bins=bins, alpha=0.6, label='Video', color='green', edgecolor='black')
#     ax3.hist(avt_audio[avt_audio > 0], bins=bins, alpha=0.6, label='Audio', color='orange', edgecolor='black')
#     ax3.set_xlabel('Usage Count', fontsize=12)
#     ax3.set_ylabel('Number of Codebook Vectors', fontsize=12)
#     ax3.set_title('AVT Model: Codebook Utilization Histogram', fontsize=14, fontweight='bold')
#     ax3.legend(fontsize=11)
#     ax3.grid(True, alpha=0.3)
    
#     # AVT Heatmap
#     ax4 = plt.subplot(2, 2, 4)
#     combined_avt = np.stack([avt_video[:rows*cols].reshape(rows, cols), 
#                              avt_audio[:rows*cols].reshape(rows, cols)])
#     im = ax4.imshow(combined_avt.mean(axis=0), cmap='YlGnBu', aspect='auto')
#     ax4.set_title('AVT Model: Codebook Utilization Heatmap', fontsize=14, fontweight='bold')
#     ax4.set_xlabel('Codebook Vector Index (grouped)', fontsize=12)
#     ax4.set_ylabel('Codebook Vector Index (grouped)', fontsize=12)
#     plt.colorbar(im, ax=ax4, label='Average Usage Count')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'codebook_utilization_comparison.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"\nVisualization saved to {output_dir}/codebook_utilization_comparison.png")

# def main():
#     # Setup
#     output_dir = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'
#     os.makedirs(output_dir, exist_ok=True)
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         handlers=[
#             logging.FileHandler(os.path.join(output_dir, 'analysis.log')),
#             logging.StreamHandler()
#         ]
#     )
#     logger = logging.getLogger(__name__)
    
#     # Configuration - CHANGE n_embeddings to 800 if needed
#     n_embeddings = 800  # User requested 800
#     batch_size = 64
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     logger.info(f"Using device: {device}")
#     logger.info(f"Number of embeddings: {n_embeddings}")
#     logger.info(f"Output directory: {output_dir}")
    
#     # Checkpoints
#     av_checkpoint = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AV/40k/checkpoint/HierVQ-model-AV-5.pt'
#     avt_checkpoint = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AVT/40k/checkpoint/Ablation-AVT-5.pt'
    
#     # Dataset
#     meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv'
#     audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip'
#     video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip'
    
#     # Import dataset classes
#     sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
#     from dataset.VGG_dataset_novel import VGGSoundDataset_AV_novel, VGGSoundDataset_AVT
    
#     # Create dataloaders
#     logger.info("Creating dataloaders...")
    
#     av_dataset = VGGSoundDataset_AV_novel(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train')
#     av_dataloader = DataLoader(av_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_func_AV)
    
#     avt_dataset = VGGSoundDataset_AVT(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train')
#     avt_dataloader = DataLoader(avt_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_func_AVT)
    
#     logger.info(f"Dataset size: {len(av_dataset)} samples")
    
#     # Analyze models
#     av_video, av_audio = analyze_av_model(av_checkpoint, av_dataloader, n_embeddings, device, logger)
#     avt_video, avt_audio = analyze_avt_model(avt_checkpoint, avt_dataloader, n_embeddings, device, logger)
    
#     # Create visualizations
#     create_visualizations(av_video, av_audio, avt_video, avt_audio, n_embeddings, output_dir)
    
#     # Save statistics
#     with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
#         f.write("CODEBOOK UTILIZATION ANALYSIS\n")
#         f.write("="*80 + "\n\n")
#         f.write(f"Number of codebook vectors: {n_embeddings}\n\n")
        
#         f.write("AV MODEL:\n")
#         f.write(f"  Video - Used: {np.sum(av_video > 0)}/{n_embeddings} ({100*np.sum(av_video > 0)/n_embeddings:.1f}%)\n")
#         f.write(f"  Audio - Used: {np.sum(av_audio > 0)}/{n_embeddings} ({100*np.sum(av_audio > 0)/n_embeddings:.1f}%)\n\n")
        
#         f.write("AVT MODEL:\n")
#         f.write(f"  Video - Used: {np.sum(avt_video > 0)}/{n_embeddings} ({100*np.sum(avt_video > 0)/n_embeddings:.1f}%)\n")
#         f.write(f"  Audio - Used: {np.sum(avt_audio > 0)}/{n_embeddings} ({100*np.sum(avt_audio > 0)/n_embeddings:.1f}%)\n")
    
#     logger.info("\nAnalysis complete!")

# if __name__ == '__main__':
#     main()


import logging
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from collections import Counter
import torch.nn.functional as F

# Add paths
sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')
from model.main_model_novel import AV_VQVAE_Encoder, AVT_VQVAE_Encoder
from transformers import BertTokenizer, BertModel

# Seeds
SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

# Load text processing utilities
with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def collate_func_AV(samples):
    return {
        'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
    }

def collate_func_AVT(samples):
    bsz = len(samples)
    text_prompts = [sample['text_fea'] for sample in samples]
    query = []
    query_words = []
    
    for text in text_prompts:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
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
        query.append(np.asarray(words_emb))
        query_words.append(words)

    query_len = [10 for _ in query]
    query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
    query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
    for i, sample in enumerate(query):
        keep = min(sample.shape[0], query1.shape[1])
        if keep > 0:
            query1[i, :keep] = sample[:keep]
            query_idx[i, :keep] = query_words[i][:keep]
    
    query_len = np.asarray(query_len)
    query = torch.from_numpy(query1).float()

    return {
        'query': query,
        'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
    }

def extract_indices_av(encoder, audio_feat, video_feat, device='cuda'):
    """Extract codebook indices from AV encoder"""
    encoder.eval()
    
    with torch.no_grad():
        audio_feat = audio_feat.to(device).double()
        video_feat = video_feat.to(device).double()
        
        # Get semantic features
        video_semantic_result, _ = encoder.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = encoder.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = encoder.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        
        B, T, D = audio_semantic_result.size()
        
        a_flat = audio_semantic_result.reshape(-1, D)
        v_flat = video_semantic_result.reshape(-1, D)
        
        # Extract embeddings
        video_embedding = encoder.Cross_quantizer.embedding[:, :D]
        audio_embedding = encoder.Cross_quantizer.embedding[:, D:]
        
        # Compute distances
        v_distances = torch.addmm(
            torch.sum(video_embedding ** 2, dim=1) + torch.sum(v_flat ** 2, dim=1, keepdim=True),
            v_flat, video_embedding.t(), alpha=-2.0, beta=1.0
        )
        
        a_distances = torch.addmm(
            torch.sum(audio_embedding ** 2, dim=1) + torch.sum(a_flat ** 2, dim=1, keepdim=True),
            a_flat, audio_embedding.t(), alpha=-2.0, beta=1.0
        )
        
        v_indices = torch.argmin(v_distances.double(), dim=-1).reshape(B, T)
        a_indices = torch.argmin(a_distances.double(), dim=-1).reshape(B, T)
        
        return v_indices, a_indices

def extract_indices_avt(encoder, audio_feat, video_feat, text_feat, device='cuda'):
    """Extract codebook indices from AVT encoder"""
    encoder.eval()
    
    with torch.no_grad():
        audio_feat = audio_feat.to(device).double()
        video_feat = video_feat.to(device).double()
        text_feat = text_feat.to(device).double()
        
        # Get semantic features
        video_semantic_result, _ = encoder.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = encoder.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = encoder.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = encoder.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        
        B, T, D = audio_semantic_result.size()
        
        a_flat = audio_semantic_result.reshape(-1, D)
        v_flat = video_semantic_result.reshape(-1, D)
        
        # Extract embeddings
        video_embedding = encoder.Cross_quantizer.embedding[:, :D]
        audio_embedding = encoder.Cross_quantizer.embedding[:, D:2*D]
        
        # Compute distances
        v_distances = torch.addmm(
            torch.sum(video_embedding ** 2, dim=1) + torch.sum(v_flat ** 2, dim=1, keepdim=True),
            v_flat, video_embedding.t(), alpha=-2.0, beta=1.0
        )
        
        a_distances = torch.addmm(
            torch.sum(audio_embedding ** 2, dim=1) + torch.sum(a_flat ** 2, dim=1, keepdim=True),
            a_flat, audio_embedding.t(), alpha=-2.0, beta=1.0
        )
        
        v_indices = torch.argmin(v_distances.double(), dim=-1).reshape(B, T)
        a_indices = torch.argmin(a_distances.double(), dim=-1).reshape(B, T)
        
        return v_indices, a_indices

def analyze_av_model(checkpoint_path, dataloader, n_embeddings, device, logger):
    """Analyze AV model codebook utilization"""
    logger.info(f"\nAnalyzing AV Model")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    encoder = AV_VQVAE_Encoder(
        audio_dim=128,
        video_dim=512,
        video_output_dim=2048,
        n_embeddings=n_embeddings,
        embedding_dim=256
    )
    encoder.double().to(device).eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Track indices
    video_counter = Counter()
    audio_counter = Counter()
    
    for batch_data in tqdm(dataloader, desc="Processing AV"):
        audio_feat = batch_data['audio_fea']
        video_feat = batch_data['video_fea']
        
        v_indices, a_indices = extract_indices_av(encoder, audio_feat, video_feat, device)
        
        video_counter.update(v_indices.cpu().numpy().flatten().tolist())
        audio_counter.update(a_indices.cpu().numpy().flatten().tolist())
    
    # Convert to arrays
    video_counts = np.zeros(n_embeddings)
    audio_counts = np.zeros(n_embeddings)
    
    for idx, count in video_counter.items():
        video_counts[idx] = count
    for idx, count in audio_counter.items():
        audio_counts[idx] = count
    
    logger.info(f"Video: {np.sum(video_counts > 0)}/{n_embeddings} vectors used ({100*np.sum(video_counts > 0)/n_embeddings:.1f}%)")
    logger.info(f"Audio: {np.sum(audio_counts > 0)}/{n_embeddings} vectors used ({100*np.sum(audio_counts > 0)/n_embeddings:.1f}%)")
    
    return video_counts, audio_counts

def analyze_avt_model(checkpoint_path, dataloader, n_embeddings, device, logger):
    """Analyze AVT model codebook utilization"""
    logger.info(f"\nAnalyzing AVT Model")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load models
    encoder = AVT_VQVAE_Encoder(
        audio_dim=128,
        video_dim=512,
        text_dim=256,
        video_output_dim=2048,
        n_embeddings=n_embeddings,
        embedding_dim=256
    )
    text_lstm = torch.nn.LSTM(768, 128, num_layers=2, batch_first=True, bidirectional=True)
    
    encoder.double().to(device).eval()
    text_lstm.double().to(device).eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    text_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Track indices
    video_counter = Counter()
    audio_counter = Counter()
    
    for batch_data in tqdm(dataloader, desc="Processing AVT"):
        audio_feat = batch_data['audio_fea']
        video_feat = batch_data['video_fea']
        query = batch_data['query'].double().to(device)
        
        batch_dim = query.size()[0]
        text_hidden = (
            torch.zeros(4, batch_dim, 128).double().to(device),
            torch.zeros(4, batch_dim, 128).double().to(device)
        )
        text_feat, _ = text_lstm(query, text_hidden)
        
        v_indices, a_indices = extract_indices_avt(encoder, audio_feat, video_feat, text_feat, device)
        
        video_counter.update(v_indices.cpu().numpy().flatten().tolist())
        audio_counter.update(a_indices.cpu().numpy().flatten().tolist())
    
    # Convert to arrays
    video_counts = np.zeros(n_embeddings)
    audio_counts = np.zeros(n_embeddings)
    
    for idx, count in video_counter.items():
        video_counts[idx] = count
    for idx, count in audio_counter.items():
        audio_counts[idx] = count
    
    logger.info(f"Video: {np.sum(video_counts > 0)}/{n_embeddings} vectors used ({100*np.sum(video_counts > 0)/n_embeddings:.1f}%)")
    logger.info(f"Audio: {np.sum(audio_counts > 0)}/{n_embeddings} vectors used ({100*np.sum(audio_counts > 0)/n_embeddings:.1f}%)")
    
    return video_counts, audio_counts

def create_utilization_bar_chart(video_counts, audio_counts, n_embeddings, model_name, output_path):
    """Create bar chart showing number of vectors utilized - KEY PLOT for showing balance"""
    plt.figure(figsize=(10, 7))
    
    video_used = np.sum(video_counts > 0)
    audio_used = np.sum(audio_counts > 0)
    
    categories = ['Video', 'Audio']
    used = [video_used, audio_used]
    unused = [n_embeddings - video_used, n_embeddings - audio_used]
    
    x = np.arange(len(categories))
    width = 0.5
    
    # Stacked bar chart
    p1 = plt.bar(x, used, width, label='Used Vectors', color='#2ecc71', edgecolor='black', linewidth=1.5)
    p2 = plt.bar(x, unused, width, bottom=used, label='Unused Vectors', color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.6)
    
    plt.ylabel('Number of Codebook Vectors', fontsize=14, fontweight='bold')
    plt.title(f'{model_name}: Codebook Vector Utilization', fontsize=16, fontweight='bold')
    plt.xticks(x, categories, fontsize=13)
    plt.legend(fontsize=12, loc='upper right')
    plt.ylim(0, n_embeddings * 1.1)
    
    # Add value labels
    for i, (u, un) in enumerate(zip(used, unused)):
        plt.text(i, u/2, f'{u}\n({100*u/n_embeddings:.1f}%)', 
                ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        plt.text(i, u + un/2, f'{un}\n({100*un/n_embeddings:.1f}%)', 
                ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(output_path)}")

def create_usage_intensity_heatmap(video_counts, audio_counts, n_embeddings, model_name, output_path):
    """Create interpretable heatmap showing usage patterns"""
    plt.figure(figsize=(14, 6))
    
    # Create matrix: rows are [Video, Audio], columns are codebook vectors
    data = np.vstack([video_counts, audio_counts])
    
    # Normalize for better visualization
    data_normalized = data / (data.max() + 1e-10)
    
    # Create heatmap
    ax = sns.heatmap(data_normalized, 
                     cmap='YlOrRd', 
                     cbar_kws={'label': 'Normalized Usage Intensity'},
                     xticklabels=False,  # Too many to show
                     yticklabels=['Video', 'Audio'],
                     linewidths=0)
    
    plt.xlabel('Codebook Vector Index', fontsize=13, fontweight='bold')
    plt.ylabel('Modality', fontsize=13, fontweight='bold')
    plt.title(f'{model_name}: Codebook Usage Intensity Across Vectors', fontsize=15, fontweight='bold')
    
    # Add x-axis markers
    xticks_pos = np.linspace(0, n_embeddings, 11)
    xticks_labels = [f'{int(x)}' for x in xticks_pos]
    plt.xticks(xticks_pos, xticks_labels, fontsize=10)
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(output_path)}")

def create_cumulative_usage_plot(video_counts, audio_counts, n_embeddings, model_name, output_path):
    """Create cumulative usage plot - shows if modalities have similar spread"""
    plt.figure(figsize=(10, 7))
    
    # Sort vectors by usage
    video_sorted = np.sort(video_counts[video_counts > 0])[::-1]
    audio_sorted = np.sort(audio_counts[audio_counts > 0])[::-1]
    
    # Cumulative sum
    video_cumsum = np.cumsum(video_sorted)
    audio_cumsum = np.cumsum(audio_sorted)
    
    # Normalize to percentage
    video_cumsum_pct = (video_cumsum / video_cumsum[-1]) * 100
    audio_cumsum_pct = (audio_cumsum / audio_cumsum[-1]) * 100
    
    plt.plot(range(len(video_cumsum_pct)), video_cumsum_pct, 
             label='Video', linewidth=3, color='#3498db', marker='o', markersize=4, markevery=max(1, len(video_cumsum_pct)//20))
    plt.plot(range(len(audio_cumsum_pct)), audio_cumsum_pct, 
             label='Audio', linewidth=3, color='#e74c3c', marker='s', markersize=4, markevery=max(1, len(audio_cumsum_pct)//20))
    
    plt.xlabel('Number of Codebook Vectors (Sorted by Usage)', fontsize=13, fontweight='bold')
    plt.ylabel('Cumulative Usage (%)', fontsize=13, fontweight='bold')
    plt.title(f'{model_name}: Cumulative Codebook Usage Distribution', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines
    plt.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    plt.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(output_path)}")

def create_distribution_comparison(video_counts, audio_counts, model_name, output_path):
    """Create usage frequency distribution - demonstrates similar usage patterns"""
    plt.figure(figsize=(12, 7))
    
    # Get only non-zero counts
    video_nonzero = video_counts[video_counts > 0]
    audio_nonzero = audio_counts[audio_counts > 0]
    
    # Create bins based on data
    max_count = max(video_nonzero.max(), audio_nonzero.max())
    bins = np.logspace(np.log10(1), np.log10(max_count + 1), 40)
    
    plt.hist(video_nonzero, bins=bins, alpha=0.6, label=f'Video (n={len(video_nonzero)})', 
             color='#3498db', edgecolor='black', linewidth=1.2)
    plt.hist(audio_nonzero, bins=bins, alpha=0.6, label=f'Audio (n={len(audio_nonzero)})', 
             color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    plt.xscale('log')
    plt.xlabel('Usage Count (log scale)', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Codebook Vectors', fontsize=13, fontweight='bold')
    plt.title(f'{model_name}: Distribution of Vector Usage Frequency', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    
    # Add text box with statistics
    textstr = f'Video: {len(video_nonzero)} vectors, Mean={video_nonzero.mean():.0f}\n'
    textstr += f'Audio: {len(audio_nonzero)} vectors, Mean={audio_nonzero.mean():.0f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {os.path.basename(output_path)}")

def main():
    # Setup
    output_dir = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'analysis.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    n_embeddings = 800  # Match your checkpoint - CHANGE if needed
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Number of embeddings: {n_embeddings}")
    logger.info(f"Output directory: {output_dir}")
    
    # Checkpoints
    av_checkpoint = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AV/40k/checkpoint/HierVQ-model-AV-5.pt'
    avt_checkpoint = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AVT/40k/checkpoint/Ablation-AVT-5.pt'
    
    # Dataset
    meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv'
    audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip'
    video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip'
    
    # Import dataset classes
    sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
    from dataset.VGG_dataset_novel import VGGSoundDataset_AV_novel, VGGSoundDataset_AVT
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    
    av_dataset = VGGSoundDataset_AV_novel(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train')
    av_dataloader = DataLoader(av_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_func_AV)
    
    avt_dataset = VGGSoundDataset_AVT(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train')
    avt_dataloader = DataLoader(avt_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_func_AVT)
    
    logger.info(f"Dataset size: {len(av_dataset)} samples")
    
    # Analyze models
    av_video, av_audio = analyze_av_model(av_checkpoint, av_dataloader, n_embeddings, device, logger)
    avt_video, avt_audio = analyze_avt_model(avt_checkpoint, avt_dataloader, n_embeddings, device, logger)
    
    print("\n" + "="*80)
    print("Creating 4 separate plots for AV model...")
    print("="*80)
    create_utilization_bar_chart(av_video, av_audio, n_embeddings, 'AV Model', 
                                 os.path.join(output_dir, 'AV_utilization_bar.png'))
    create_usage_intensity_heatmap(av_video, av_audio, n_embeddings, 'AV Model',
                                   os.path.join(output_dir, 'AV_intensity_heatmap.png'))
    create_cumulative_usage_plot(av_video, av_audio, n_embeddings, 'AV Model',
                                os.path.join(output_dir, 'AV_cumulative_usage.png'))
    create_distribution_comparison(av_video, av_audio, 'AV Model',
                                   os.path.join(output_dir, 'AV_distribution.png'))
    
    print("\n" + "="*80)
    print("Creating 4 separate plots for AVT model...")
    print("="*80)
    create_utilization_bar_chart(avt_video, avt_audio, n_embeddings, 'AVT Model',
                                 os.path.join(output_dir, 'AVT_utilization_bar.png'))
    create_usage_intensity_heatmap(avt_video, avt_audio, n_embeddings, 'AVT Model',
                                   os.path.join(output_dir, 'AVT_intensity_heatmap.png'))
    create_cumulative_usage_plot(avt_video, avt_audio, n_embeddings, 'AVT Model',
                                os.path.join(output_dir, 'AVT_cumulative_usage.png'))
    create_distribution_comparison(avt_video, avt_audio, 'AVT Model',
                                   os.path.join(output_dir, 'AVT_distribution.png'))
    
    # Save detailed statistics
    with open(os.path.join(output_dir, 'statistics_detailed.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("CODEBOOK UTILIZATION ANALYSIS - DETAILED STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total codebook vectors: {n_embeddings}\n\n")
        
        f.write("AV MODEL:\n")
        f.write("-"*80 + "\n")
        av_v_used = np.sum(av_video > 0)
        av_a_used = np.sum(av_audio > 0)
        f.write(f"Video:\n")
        f.write(f"  Vectors used: {av_v_used}/{n_embeddings} ({100*av_v_used/n_embeddings:.1f}%)\n")
        f.write(f"  Mean usage (non-zero): {av_video[av_video>0].mean():.0f}\n")
        f.write(f"  Max usage: {av_video.max():.0f}\n")
        f.write(f"Audio:\n")
        f.write(f"  Vectors used: {av_a_used}/{n_embeddings} ({100*av_a_used/n_embeddings:.1f}%)\n")
        f.write(f"  Mean usage (non-zero): {av_audio[av_audio>0].mean():.0f}\n")
        f.write(f"  Max usage: {av_audio.max():.0f}\n")
        f.write(f"Difference: Audio uses {av_a_used-av_v_used:+d} vectors relative to Video ")
        f.write(f"({100*(av_a_used-av_v_used)/n_embeddings:+.1f}%)\n\n")
        
        f.write("AVT MODEL:\n")
        f.write("-"*80 + "\n")
        avt_v_used = np.sum(avt_video > 0)
        avt_a_used = np.sum(avt_audio > 0)
        f.write(f"Video:\n")
        f.write(f"  Vectors used: {avt_v_used}/{n_embeddings} ({100*avt_v_used/n_embeddings:.1f}%)\n")
        f.write(f"  Mean usage (non-zero): {avt_video[avt_video>0].mean():.0f}\n")
        f.write(f"  Max usage: {avt_video.max():.0f}\n")
        f.write(f"Audio:\n")
        f.write(f"  Vectors used: {avt_a_used}/{n_embeddings} ({100*avt_a_used/n_embeddings:.1f}%)\n")
        f.write(f"  Mean usage (non-zero): {avt_audio[avt_audio>0].mean():.0f}\n")
        f.write(f"  Max usage: {avt_audio.max():.0f}\n")
        f.write(f"Difference: Audio uses {avt_a_used-avt_v_used:+d} vectors relative to Video ")
        f.write(f"({100*(avt_a_used-avt_v_used)/n_embeddings:+.1f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        
        av_diff_pct = abs(100*(av_a_used - av_v_used)/n_embeddings)
        avt_diff_pct = abs(100*(avt_a_used - avt_v_used)/n_embeddings)
        
        if av_diff_pct < 10:
            f.write(f"✓ AV Model: Audio and Video use similar number of vectors (diff={av_diff_pct:.1f}%)\n")
        else:
            f.write(f"  AV Model: Audio and Video differ in usage by {av_diff_pct:.1f}%\n")
            
        if avt_diff_pct < 10:
            f.write(f"✓ AVT Model: Audio and Video use similar number of vectors (diff={avt_diff_pct:.1f}%)\n")
        else:
            f.write(f"  AVT Model: Audio and Video differ in usage by {avt_diff_pct:.1f}%\n")
            
        f.write(f"\n")
        f.write(f"CONCLUSION: ")
        if av_diff_pct < 10 and avt_diff_pct < 10:
            f.write(f"Both models show balanced codebook utilization.\n")
            f.write(f"Audio is NOT underrepresented in either model.\n")
        else:
            f.write(f"Check individual plots for detailed analysis.\n")
    
    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)
    print(f"\nGenerated 8 visualizations (4 per model):")
    print("  1. Utilization Bar Chart - Shows # of vectors used")
    print("  2. Intensity Heatmap - Shows which vectors are used")
    print("  3. Cumulative Usage - Shows usage concentration")
    print("  4. Distribution Comparison - Shows usage frequency patterns")
    print(f"\nAll files saved to: {output_dir}")
    print("="*80)
    
    logger.info("\nAll visualizations and statistics saved!")

if __name__ == '__main__':
    main()