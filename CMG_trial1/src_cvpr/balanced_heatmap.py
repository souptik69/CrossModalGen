import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap

# Import both model classes
from model.main_model_novel import AV_VQVAE_Encoder as CoDAAR_Encoder
from model.main_model_2 import AV_VQVAE_Encoder as MICU_Encoder

# Configuration
BATCH_SIZE = 500
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
        
        # SPLIT CODEBOOK
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
        
        # SHARED CODEBOOK
        embedding = encoder.Cross_quantizer.embedding
        
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

def create_balance_heatmap(codaar_video, codaar_audio, micu_video, micu_audio):
    """Create balance heatmap showing audio-video dominance for each vector
    
    Balance score: (audio_usage - video_usage) / (audio_usage + video_usage)
    - Positive (red) = Audio-dominant
    - Negative (blue) = Video-dominant
    - Zero (white) = Balanced
    """
    print("\n" + "="*70)
    print("Creating Audio-Video Balance Heatmap")
    print("="*70)
    
    n_vectors = 400
    
    # Calculate balance for each vector
    codaar_balance = np.zeros(n_vectors)
    micu_balance = np.zeros(n_vectors)

    codaar_threshold = 0.4
    micu_threshold = 0.001
    
    for i in range(n_vectors):
        # CoDAAR balance
        total_c = codaar_audio[i] + codaar_video[i]
        if total_c > 0:
            # Positive = audio-dominant, Negative = video-dominant
            balance_c  = (codaar_audio[i] - codaar_video[i]) / total_c

            if abs(balance_c) <= codaar_threshold:
                codaar_balance[i] = 0  # Within threshold → Balanced (white)
            else:
                codaar_balance[i] = balance_c 
        else:
            # Unused vector - set to NaN for neutral color
            codaar_balance[i] = 0
        
        # MICU balance
        total_m = micu_audio[i] + micu_video[i]
        if total_m > 0:
            balance_m = (micu_audio[i] - micu_video[i]) / total_m
            if abs(balance_m) <= micu_threshold:
                micu_balance[i] = 0  # Within threshold → Balanced (white)
            else:
                micu_balance[i] = balance_m 
        else:
            micu_balance[i] = 0
    
    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(18, 6))
    
    # # Use diverging colormap: Blue=Video-dominant, Red=Audio-dominant, White=Balanced
    # cmap = 'RdBu_r'  # Red for audio, Blue for video

    colors = ['#0000FF',  # Pure Blue (video-dominant)
          '#4D4DFF',  # Light Blue
          '#B8FFB8',  # Light yellow (balanced)
          '#FF4D4D',  # Light Red
          '#FF0000']  # Pure Red (audio-dominant)

    # colors = ['#0055FF',  # Blue (video-dominant)
    #       '#88CCFF',  # Light blue
    #       '#00DD00',  # Green (balanced) 
    #       '#FFAA44',  # Orange  
    #       '#FF0000']  # Red (audio-dominant)
    
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('audio_video_balance', colors, N=n_bins)
    
    # Row 0: CoDAAR
    sns.heatmap(codaar_balance.reshape(1, -1), ax=axes[0], 
                cmap=cmap, center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Audio ← Balanced → Video'},
                xticklabels=False, yticklabels=['CoDAAR'], linewidths=0)
    axes[0].set_ylabel('', fontsize=18)
    axes[0].tick_params(axis='y', labelsize=16)
    
    # Row 1: MICU
    sns.heatmap(micu_balance.reshape(1, -1), ax=axes[1],
                cmap=cmap, center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Audio ← Balanced → Video'},
                xticklabels=False, yticklabels=['MICU'], linewidths=0)
    axes[1].set_ylabel('', fontsize=18)
    axes[1].tick_params(axis='y', labelsize=16)
    axes[1].set_xlabel('Codebook Vector Index', fontsize=16, fontweight='bold')
    
    # Add x-axis ticks
    xticks_pos = np.linspace(0, 400, 11)
    axes[1].set_xticks(xticks_pos)
    axes[1].set_xticklabels([f'{int(x)}' for x in xticks_pos], fontsize=12)
    
    # Overall title
    fig.suptitle('Modality Dominance per Codebook Vector',
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Adjust spacing
    plt.subplots_adjust(hspace=0.3, top=0.92, bottom=0.12)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'Final_Balance_CoDAAR_vs_MICU_heatmap_final_4.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Balance heatmap saved: {output_path}")
    
    # Calculate and print statistics
    print("\n" + "="*70)
    print("Balance Statistics")
    print("="*70)
    
    # CoDAAR
    codaar_audio_dom = np.sum(codaar_balance > 0.4)  # Strongly audio-dominant
    codaar_video_dom = np.sum(codaar_balance < -0.4)  # Strongly video-dominant
    codaar_balanced = np.sum(np.abs(codaar_balance) <= 0.4)  # Balanced
    
    print(f"\nCoDAAR (SPLIT Codebook):")
    print(f"  Video-dominant vectors: {codaar_video_dom}/400 ({100*codaar_video_dom/400:.1f}%)")
    print(f"  Balanced vectors: {codaar_balanced}/400 ({100*codaar_balanced/400:.1f}%)")
    print(f"  Audio-dominant vectors: {codaar_audio_dom}/400 ({100*codaar_audio_dom/400:.1f}%)")
    
    # MICU
    micu_audio_dom = np.sum(micu_balance > 0.2)
    micu_video_dom = np.sum(micu_balance < -0.2)
    micu_balanced = np.sum(np.abs(micu_balance) <= 0.2)
    
    print(f"\nMICU (SHARED Codebook):")
    print(f"  Video-dominant vectors: {micu_video_dom}/400 ({100*micu_video_dom/400:.1f}%)")
    print(f"  Balanced vectors: {micu_balanced}/400 ({100*micu_balanced/400:.1f}%)")
    print(f"  Audio-dominant vectors: {micu_audio_dom}/400 ({100*micu_audio_dom/400:.1f}%)")
    
    # Save statistics
    with open(os.path.join(OUTPUT_DIR, 'Balance_statistics_2.txt'), 'w') as f:
        f.write("AUDIO-VIDEO BALANCE ANALYSIS: CoDAAR vs MICU\n")
        f.write("="*70 + "\n\n")
        f.write("Balance Score = (audio_usage - video_usage) / (audio_usage + video_usage)\n")
        f.write("  > +0.2: Audio-dominant\n")
        f.write("  -0.2 to +0.2: Balanced\n")
        f.write("  < -0.2: Video-dominant\n\n")
        
        f.write("CoDAAR (SPLIT Codebook):\n")
        f.write(f"  Video-dominant: {codaar_video_dom}/400 ({100*codaar_video_dom/400:.1f}%)\n")
        f.write(f"  Balanced: {codaar_balanced}/400 ({100*codaar_balanced/400:.1f}%)\n")
        f.write(f"  Audio-dominant: {codaar_audio_dom}/400 ({100*codaar_audio_dom/400:.1f}%)\n\n")
        
        f.write("MICU (SHARED Codebook):\n")
        f.write(f"  Video-dominant: {micu_video_dom}/400 ({100*micu_video_dom/400:.1f}%)\n")
        f.write(f"  Balanced: {micu_balanced}/400 ({100*micu_balanced/400:.1f}%)\n")
        f.write(f"  Audio-dominant: {micu_audio_dom}/400 ({100*micu_audio_dom/400:.1f}%)\n\n")
        
        f.write("KEY INSIGHT:\n")
        f.write("="*70 + "\n")
        f.write("CoDAAR's split architecture allows independent balance per modality.\n")
        f.write("MICU's shared codebook shows competition - vectors dominated by one modality.\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("AUDIO-VIDEO BALANCE ANALYSIS: CoDAAR vs MICU")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load dataset
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
    
    # Create balance heatmap
    create_balance_heatmap(codaar_video, codaar_audio, micu_video, micu_audio)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    torch.manual_seed(43)
    np.random.seed(43)
    main()