import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

# sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')

# Import both model classes
from model.main_model_novel import AV_VQVAE_Encoder as CoDAAR_Encoder
from model.main_model_2 import AV_VQVAE_Encoder as MICU_Encoder

# Configuration
BATCH_SIZE = 500  # Use CoDAAR's batch size for consistency
OUTPUT_DIR = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'

# Model-specific configs
CODAAR_CONFIG = {
    'n_embeddings': 400,
    'checkpoint': '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AV/400Codebook/checkpoint/HierVQ-model-AV-5.pt',
    'name': 'CoDAAR'
}

MICU_CONFIG = {
    'n_embeddings': 400,
    'checkpoint': '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AV/MICU/40k/checkpoint/MICU-step2400.pt',
    'name': 'MICU'
}

def collate_func_AV(samples):
    """Simple collate function for audio-video"""
    return {
        'audio_fea': torch.from_numpy(np.asarray([s['audio_fea'] for s in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([s['video_fea'] for s in samples])).float()
    }

def extract_indices_codaar(encoder, audio_feat, video_feat, device):
    """Extract indices from CoDAAR (SPLIT codebook architecture)"""
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
        
        # SPLIT CODEBOOK: First half for video, second half for audio
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

def extract_indices_micu(encoder, audio_feat, video_feat, device):
    """Extract indices from MICU (SHARED codebook architecture)"""
    encoder.eval()
    with torch.no_grad():
        audio_feat = audio_feat.to(device).double()
        video_feat = video_feat.to(device).double()
        
        # Video processing
        video_semantic = encoder.video_semantic_encoder(video_feat)
        video_semantic = video_semantic.transpose(0, 1).contiguous()
        video_semantic = encoder.video_self_att(video_semantic)
        video_semantic = video_semantic.transpose(0, 1).contiguous()
        
        # Audio processing
        audio_semantic = audio_feat.transpose(0, 1).contiguous()
        audio_semantic = encoder.audio_self_att(audio_semantic)
        audio_semantic = audio_semantic.transpose(0, 1).contiguous()
        
        B, T, D = audio_semantic.size()
        
        # SHARED CODEBOOK: Both modalities use same embedding
        embedding = encoder.Cross_quantizer.embedding  # [400, 256]
        
        # Compute distances
        v_flat = video_semantic.reshape(-1, D)
        a_flat = audio_semantic.reshape(-1, D)
        
        v_distances = torch.addmm(
            torch.sum(embedding**2, dim=1) + torch.sum(v_flat**2, dim=1, keepdim=True),
            v_flat, embedding.t(), alpha=-2.0, beta=1.0
        )
        a_distances = torch.addmm(
            torch.sum(embedding**2, dim=1) + torch.sum(a_flat**2, dim=1, keepdim=True),
            a_flat, embedding.t(), alpha=-2.0, beta=1.0
        )
        
        v_indices = torch.argmin(v_distances, dim=-1).reshape(B, T)
        a_indices = torch.argmin(a_distances, dim=-1).reshape(B, T)
        
        return v_indices, a_indices

def analyze_model(model_config, extract_fn, encoder_class, dataloader, device):
    """Analyze a single model and return usage statistics"""
    print(f"\n{'='*70}")
    print(f"Analyzing {model_config['name']} Model")
    print(f"{'='*70}")
    
    # Load model
    print(f"Loading {model_config['name']} encoder...")
    if model_config['name'] == 'CoDAAR':
        encoder = encoder_class(
            audio_dim=128, video_dim=512, video_output_dim=2048,
            n_embeddings=model_config['n_embeddings'], embedding_dim=256
        )
    else:  # MICU
        encoder = encoder_class(
            audio_dim=128, video_dim=512,
            audio_output_dim=256, video_output_dim=2048,
            n_embeddings=model_config['n_embeddings'], embedding_dim=256
        )
    
    encoder.double().to(device).eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_config['checkpoint'], map_location=device)
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    print(f"✓ Loaded checkpoint")
    
    # Collect usage statistics
    print(f"Processing dataset...")
    video_counter = Counter()
    audio_counter = Counter()
    
    for batch_data in tqdm(dataloader, desc=f"{model_config['name']} Processing"):
        v_idx, a_idx = extract_fn(encoder, batch_data['audio_fea'], 
                                   batch_data['video_fea'], device)
        video_counter.update(v_idx.cpu().numpy().flatten().tolist())
        audio_counter.update(a_idx.cpu().numpy().flatten().tolist())
    
    # Convert to arrays
    video_counts = np.zeros(model_config['n_embeddings'])
    audio_counts = np.zeros(model_config['n_embeddings'])
    
    for idx, count in video_counter.items():
        video_counts[idx] = count
    for idx, count in audio_counter.items():
        audio_counts[idx] = count
    
    n_emb = model_config['n_embeddings']
    print(f"\nResults:")
    print(f"  Video: {np.sum(video_counts > 0)}/{n_emb} vectors used ({100*np.sum(video_counts>0)/n_emb:.1f}%)")
    print(f"  Audio: {np.sum(audio_counts > 0)}/{n_emb} vectors used ({100*np.sum(audio_counts>0)/n_emb:.1f}%)")
    
    return video_counts, audio_counts

def create_comparison_plot(codaar_video, codaar_audio, micu_video, micu_audio):
    """Create a 4-row comparison heatmap with proper spacing"""
    print("\n" + "="*70)
    print("Creating Comparison Heatmap")
    print("="*70)
    
    # Create figure with 4 subplots (one per row)
    fig, axes = plt.subplots(4, 1, figsize=(18, 12))
    
    # Normalize each dataset
    codaar_video_norm = codaar_video / (codaar_video.max() + 1e-10)
    codaar_audio_norm = codaar_audio / (codaar_audio.max() + 1e-10)
    micu_video_norm = micu_video / (micu_video.max() + 1e-10)
    micu_audio_norm = micu_audio / (micu_audio.max() + 1e-10)
    
    # Row 0: CoDAAR Video
    sns.heatmap(codaar_video_norm.reshape(1, -1), ax=axes[0], cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Intensity'}, 
                xticklabels=False, yticklabels=['Video'], linewidths=0)
    # axes[0].set_ylabel('CoDAAR', fontsize=18, fontweight='bold', rotation=90)
    axes[0].set_ylabel('', fontsize=18)
    axes[0].tick_params(axis='y', labelsize=16)
    
    # Row 1: CoDAAR Audio
    sns.heatmap(codaar_audio_norm.reshape(1, -1), ax=axes[1], cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Intensity'},
                xticklabels=False, yticklabels=['Audio'], linewidths=0)
    axes[1].set_ylabel('', fontsize=18)
    axes[1].tick_params(axis='y', labelsize=16)
    axes[1].set_xlabel('Codebook Vector Index', fontsize=16, fontweight='bold')
    
    # Add x-axis ticks for CoDAAR
    xticks_pos_codaar = np.linspace(0, 400, 11)
    axes[1].set_xticks(xticks_pos_codaar)
    axes[1].set_xticklabels([f'{int(x)}' for x in xticks_pos_codaar], fontsize=12)
    
    # Row 2: MICU Video
    sns.heatmap(micu_video_norm.reshape(1, -1), ax=axes[2], cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Intensity'},
                xticklabels=False, yticklabels=['Video'], linewidths=0)
    # axes[2].set_ylabel('MICU', fontsize=18, fontweight='bold', rotation=90)
    axes[2].set_ylabel('', fontsize=18)
    axes[2].tick_params(axis='y', labelsize=16)
    
    # Row 3: MICU Audio
    sns.heatmap(micu_audio_norm.reshape(1, -1), ax=axes[3], cmap='YlOrRd',
                cbar_kws={'label': 'Normalized Intensity'},
                xticklabels=False, yticklabels=['Audio'], linewidths=0)
    axes[3].set_ylabel('', fontsize=18)
    axes[3].tick_params(axis='y', labelsize=16)
    axes[3].set_xlabel('Codebook Vector Index', fontsize=16, fontweight='bold')
    
    # Add x-axis ticks for MICU
    xticks_pos_micu = np.linspace(0, 400, 11)
    axes[3].set_xticks(xticks_pos_micu)
    axes[3].set_xticklabels([f'{int(x)}' for x in xticks_pos_micu], fontsize=12)
    
    # Overall title
    fig.suptitle('CoDAAR vs MICU: Codebook Usage Intensity Comparison',
                 fontsize=22, fontweight='bold', y=0.995)
    
    # # Adjust spacing
    # plt.subplots_adjust(hspace=0.35, top=0.96, bottom=0.06)
    
    # # Save
    # output_path = os.path.join(OUTPUT_DIR, 'Comparison_CoDAAR_vs_MICU_heatmap.png')
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()
    
    # Initial adjustment with very small gap between rows of same model
    plt.subplots_adjust(hspace=0.08, top=0.96, bottom=0.06, left=0.10, right=0.98)
    
    # Get current positions
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    pos3 = axes[3].get_position()
    
    # Add extra gap between CoDAAR and MICU sections
    gap = 0.08  # Extra gap between models
    
    # Keep CoDAAR rows as is, move MICU rows down
    axes[2].set_position([pos2.x0, pos2.y0 - gap, pos2.width, pos2.height])
    axes[3].set_position([pos3.x0, pos3.y0 - gap, pos3.width, pos3.height])
    
    # Recalculate positions after adjustment
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    pos3 = axes[3].get_position()
    
    # Calculate middle y-position for each model section
    codaar_y_middle = (pos0.y0 + pos1.y1) / 2
    micu_y_middle = (pos2.y0 + pos3.y1) / 2
    
    # Add centered model labels on the left side
    fig.text(0.02, codaar_y_middle, 'CoDAAR', fontsize=20, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.02, micu_y_middle, 'MICU', fontsize=20, fontweight='bold',
             rotation=90, va='center', ha='center')
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'Comparison_CoDAAR_vs_MICU_heatmap_final_new.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison heatmap saved: {output_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("CODEBOOK USAGE COMPARISON: CoDAAR vs MICU")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load dataset (use CoDAAR's dataset class for consistency)
    print("\nLoading dataset...")
    sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
    from dataset.VGG_dataset_novel import VGGSoundDataset_AV_novel
    from dataset.VGGSOUND_dataset import VGGSoundDataset_AV_1
    
    codaar_dataset = VGGSoundDataset_AV_novel(
        meta_csv_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv',
        audio_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip',
        video_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip',
        split='train'
    )

    micu_dataset = VGGSoundDataset_AV_1(
        meta_csv_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv',
        audio_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip',
        video_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip',
        split='train'
    )

    codaar_dataloader = DataLoader(codaar_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=8, collate_fn=collate_func_AV)

    micu_dataloader = DataLoader(micu_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, collate_fn=collate_func_AV)

    print(f"✓ Dataset loaded: {len(codaar_dataset)} samples")
    
    # Analyze CoDAAR
    codaar_video, codaar_audio = analyze_model(
        CODAAR_CONFIG, extract_indices_codaar, CoDAAR_Encoder, codaar_dataloader, device
    )
    
    # Analyze MICU
    micu_video, micu_audio = analyze_model(
        MICU_CONFIG, extract_indices_micu, MICU_Encoder, micu_dataloader, device
    )
    
    # Create comparison plot
    create_comparison_plot(codaar_video, codaar_audio, micu_video, micu_audio)
    
    # Save statistics
    print("\nSaving statistics...")
    with open(os.path.join(OUTPUT_DIR, 'Comparison_statistics.txt'), 'w') as f:
        f.write("CODEBOOK USAGE COMPARISON: CoDAAR vs MICU\n")
        f.write("="*70 + "\n\n")
        
        f.write("CoDAAR Model (SPLIT Codebook Architecture)\n")
        f.write("-"*70 + "\n")
        f.write(f"Total vectors: 400 \n")
        f.write(f"Video: {np.sum(codaar_video > 0)}/400 used ({100*np.sum(codaar_video>0)/400:.1f}%)\n")
        f.write(f"Audio: {np.sum(codaar_audio > 0)}/400 used ({100*np.sum(codaar_audio>0)/400:.1f}%)\n\n")
        
        f.write("MICU Model (SHARED Codebook Architecture)\n")
        f.write("-"*70 + "\n")
        f.write(f"Total vectors: 400 (shared across modalities)\n")
        f.write(f"Video: {np.sum(micu_video > 0)}/400 used ({100*np.sum(micu_video>0)/400:.1f}%)\n")
        f.write(f"Audio: {np.sum(micu_audio > 0)}/400 used ({100*np.sum(micu_audio>0)/400:.1f}%)\n\n")
        
        f.write("KEY DIFFERENCES:\n")
        f.write("-"*70 + "\n")
        f.write("CoDAAR: Each modality has dedicated 400-vector space (no overlap)\n")
        f.write("MICU: Both modalities compete for same 400 vectors (overlap expected)\n")
    
    print("✓ Statistics saved")
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    torch.manual_seed(43)
    np.random.seed(43)
    main()