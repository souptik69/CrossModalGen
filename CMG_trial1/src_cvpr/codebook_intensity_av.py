import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')
from model.main_model_novel import AV_VQVAE_Encoder

# Configuration
N_EMBEDDINGS = 800
BATCH_SIZE = 128
OUTPUT_DIR = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'
# CHECKPOINT = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/400Codebook/checkpoint/HierVQ-model-AV-5.pt'
CHECKPOINT = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AV/40k/checkpoint/HierVQ-model-AV-5.pt'
def collate_func_AV(samples):
    return {
        'audio_fea': torch.from_numpy(np.asarray([s['audio_fea'] for s in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([s['video_fea'] for s in samples])).float()
    }

def extract_indices(encoder, audio_feat, video_feat, device):
    encoder.eval()
    with torch.no_grad():
        audio_feat = audio_feat.to(device).double()
        video_feat = video_feat.to(device).double()
        
        # Video processing
        video_semantic, _ = encoder.video_semantic_encoder(video_feat)
        video_semantic = video_semantic.transpose(0, 1).contiguous()
        video_semantic = encoder.video_self_att(video_semantic).transpose(0, 1).contiguous()
        
        # Audio processing
        audio_semantic = audio_feat.transpose(0, 1).contiguous()
        audio_semantic = encoder.audio_self_att(audio_semantic).transpose(0, 1).contiguous()
        
        B, T, D = audio_semantic.size()
        
        # Extract codebook embeddings
        video_embedding = encoder.Cross_quantizer.embedding[:, :D]
        audio_embedding = encoder.Cross_quantizer.embedding[:, D:]
        
        # Compute distances
        v_flat = video_semantic.reshape(-1, D)
        a_flat = audio_semantic.reshape(-1, D)
        
        v_distances = torch.addmm(
            torch.sum(video_embedding**2, dim=1) + torch.sum(v_flat**2, dim=1, keepdim=True),
            v_flat, video_embedding.t(), alpha=-2.0, beta=1.0
        )
        a_distances = torch.addmm(
            torch.sum(audio_embedding**2, dim=1) + torch.sum(a_flat**2, dim=1, keepdim=True),
            a_flat, audio_embedding.t(), alpha=-2.0, beta=1.0
        )
        
        v_indices = torch.argmin(v_distances, dim=-1).reshape(B, T)
        a_indices = torch.argmin(a_distances, dim=-1).reshape(B, T)
        
        return v_indices, a_indices

def analyze_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("Loading CoDAAR AV model...")
    encoder = AV_VQVAE_Encoder(
        audio_dim=128, video_dim=512, video_output_dim=2048,
        n_embeddings=N_EMBEDDINGS, embedding_dim=256
    )
    encoder.double().to(device).eval()
    encoder.load_state_dict(torch.load(CHECKPOINT, map_location=device)['Encoder_parameters'])
    
    # Load dataset
    sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
    from dataset.VGG_dataset_novel import VGGSoundDataset_AV_novel
    
    dataset = VGGSoundDataset_AV_novel(
        meta_csv_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv',
        audio_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip',
        video_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip',
        split='train'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=8, collate_fn=collate_func_AV)
    
    # Collect usage statistics
    print("Analyzing codebook usage...")
    video_counter = Counter()
    audio_counter = Counter()
    
    for batch_data in tqdm(dataloader, desc="Processing"):
        v_idx, a_idx = extract_indices(encoder, batch_data['audio_fea'], 
                                       batch_data['video_fea'], device)
        video_counter.update(v_idx.cpu().numpy().flatten().tolist())
        audio_counter.update(a_idx.cpu().numpy().flatten().tolist())
    
    # Convert to arrays
    video_counts = np.zeros(N_EMBEDDINGS)
    audio_counts = np.zeros(N_EMBEDDINGS)
    for idx, count in video_counter.items():
        video_counts[idx] = count
    for idx, count in audio_counter.items():
        audio_counts[idx] = count
    
    print(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)")
    print(f"Audio: {np.sum(audio_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(audio_counts>0)/N_EMBEDDINGS:.1f}%)")
    
    # Create heatmap
    print("Creating intensity heatmap...")
    plt.figure(figsize=(14, 6))
    
    data = np.vstack([video_counts, audio_counts])
    data_normalized = data / (data.max() + 1e-10)
    
    ax = sns.heatmap(data_normalized, cmap='YlOrRd',
                     cbar_kws={'label': 'Normalized Usage Intensity'},
                     xticklabels=False, yticklabels=['Video', 'Audio'],
                     linewidths=0)
    # ax.axhline(y=1, color='black', linewidth=2, linestyle='-')

    plt.xlabel('Codebook Vector Index', fontsize=15, fontweight='bold')
    plt.ylabel('Modality', fontsize=15, fontweight='bold')
    plt.title('CoDAAR AV Model: Codebook Usage Intensity Across Vectors', 
             fontsize=20, fontweight='bold')
    
    # Add x-axis ticks
    xticks_pos = np.linspace(0, N_EMBEDDINGS, 11)
    plt.xticks(xticks_pos, [f'{int(x)}' for x in xticks_pos], fontsize=10)
    plt.yticks(fontsize=15, rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'CoDAAR_AV_intensity_heatmap_800.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved: {output_path}")

if __name__ == '__main__':
    torch.manual_seed(43)
    np.random.seed(43)
    analyze_and_plot()