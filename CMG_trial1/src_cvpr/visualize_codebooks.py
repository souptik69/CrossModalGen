#!/usr/bin/env python3
"""
Codebook t-SNE Visualization Script
====================================
Analyzes and visualizes codebook vector usage patterns across modalities after one epoch.

This script:
1. Loads pretrained AV or AVT model checkpoints
2. Processes VGGSound 40k dataset for one epoch
3. Tracks which codebook indices are used by which modalities (audio, video, text)
4. Classifies indices as: inactive, unimodal (a/v/t), or cross-modal (av/vt/at/avt)
5. Creates t-SNE visualization with color coding based on usage patterns
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from collections import defaultdict
import pickle
import pandas as pd

# Add source directory to path for imports
SRCDIR = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr'
sys.path.insert(0, SRCDIR)

# Import model architectures
from model.main_model_novel import AVT_VQVAE_Encoder
from transformers import BertTokenizer, BertModel

# Set random seeds for reproducibility
import random
SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CodebookUsageTracker:
    """
    Tracks which codebook indices are used by which modalities during inference.
    
    Attributes:
        n_embeddings: Total number of codebook vectors
        usage_video: Set of indices used for video quantization
        usage_audio: Set of indices used for audio quantization
        usage_text: Set of indices used for text quantization (AVT only)
    """
    
    def __init__(self, n_embeddings=400):
        self.n_embeddings = n_embeddings
        self.usage_video = set()
        self.usage_audio = set()
        self.usage_text = set()
        
    def update(self, v_indices, a_indices, t_indices=None):
        """Update tracking with new batch of indices."""
        self.usage_video.update(v_indices.cpu().numpy().flatten().tolist())
        self.usage_audio.update(a_indices.cpu().numpy().flatten().tolist())
        if t_indices is not None:
            self.usage_text.update(t_indices.cpu().numpy().flatten().tolist())
    
    def classify_indices(self, model_type='AVT'):
        """
        Classify each codebook index based on modality usage.
        
        Returns:
            dict: Mapping of index to category label
        """
        classifications = {}
        
        for idx in range(self.n_embeddings):
            in_v = idx in self.usage_video
            in_a = idx in self.usage_audio
            in_t = idx in self.usage_text if model_type == 'AVT' else False
            
            # Count how many modalities use this index
            modality_count = sum([in_v, in_a, in_t])
            
            if modality_count == 0:
                classifications[idx] = 'inactive'
            elif modality_count == 1:
                # Unimodal
                if in_v:
                    classifications[idx] = 'v_only'
                elif in_a:
                    classifications[idx] = 'a_only'
                else:  # in_t
                    classifications[idx] = 't_only'
            else:
                # Cross-modal
                modalities = []
                if in_v:
                    modalities.append('v')
                if in_a:
                    modalities.append('a')
                if in_t:
                    modalities.append('t')
                classifications[idx] = '/'.join(sorted(modalities))
        
        return classifications
    
    def get_statistics(self):
        """Get usage statistics for logging."""
        stats = {
            'video_unique': len(self.usage_video),
            'audio_unique': len(self.usage_audio),
            'text_unique': len(self.usage_text),
            'total_active': len(self.usage_video | self.usage_audio | self.usage_text)
        }
        return stats


def load_vggsound_40k_dataset(model_type='AVT'):
    """
    Load VGGSound 40k dataset with proper preprocessing.
    
    Args:
        model_type: 'AV' or 'AVT'
    
    Returns:
        DataLoader for the dataset
    """
    # Load label to prompt mapping for text
    if model_type == 'AVT':
        with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl', 'rb') as fp:
            id2idx = pickle.load(fp)
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        
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
            
            query_len = [10 for _ in range(bsz)]  # max_num_words:10
            query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
            query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
            
            for i, sample in enumerate(query):
                keep = min(sample.shape[0], query1.shape[1])
                if keep > 0:
                    query1[i, :keep] = sample[:keep]
                    query_idx[i, :keep] = query_words[i][:keep]
            
            query_len = np.asarray(query_len)
            query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
            query_idx = torch.from_numpy(query_idx).long()
            
            return {
                'query': query,
                'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
                'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
            }
        
        collate_fn = collate_func_AVT
    else:
        def collate_func_AV(samples):
            return {
                'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
                'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
            }
        collate_fn = collate_func_AV
    
    # Import dataset class
    sys.path.insert(0, '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
    if model_type == 'AVT':
        from VGG_dataset_novel import VGGSoundDataset_AVT as VGGDataset
    else:
        from VGG_dataset_novel import VGGSoundDataset_AV_novel as VGGDataset
    
    # Dataset paths
    meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv'
    audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip'
    video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip'

    dataset = VGGDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Smaller batch size for memory efficiency
        shuffle=False,  # Don't shuffle to ensure reproducibility
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def extract_quantization_indices(encoder, audio_feature, video_feature, text_feature=None, model_type='AVT'):
    """
    Extract quantization indices using individual VQ encoders directly.
    
    This function extracts quantization indices by calling the individual 
    Audio_VQ_Encoder, Video_VQ_Encoder, and Text_VQ_Encoder directly,
    which is the proper way to use the pretrained model for inference.
    
    Args:
        encoder: The encoder model with VQ encoders
        audio_feature: Audio features [B, T, 128]
        video_feature: Video features [B, T, 7, 7, 512]
        text_feature: Text features [B, 10, 768] (AVT only)
        model_type: 'AV' or 'AVT'
    
    Returns:
        v_indices, a_indices, t_indices (if AVT, else None)
    """
    with torch.no_grad():
        # Extract video quantization indices
        # out_vq_video: full quantized vector [B, T, 3*D], video_vq: segment quantized [B, T, D]
        # The indices are stored in the encoder after forward pass
        out_vq_video, video_vq = encoder.Video_VQ_Encoder(video_feature)
        
        # Extract audio quantization indices  
        out_vq_audio, audio_vq = encoder.Audio_VQ_Encoder(audio_feature)
        
        # For AVT model, also extract text quantization
        if model_type == 'AVT' and text_feature is not None:
            out_vq_text, text_vq = encoder.Text_VQ_Encoder(text_feature)
        
        # Now extract the actual indices used during quantization
        # The indices are computed inside the VQ encoder's quantizer
        # We need to recompute them from the semantic features
        
        # Get semantic features for each modality
        audio_encoder_result, audio_semantic_result = encoder.audio_encoder(audio_feature)
        video_spatial, video_semantic_result = encoder.video_semantic_encoder(video_feature)
        
        B, T, D = video_semantic_result.shape
        
        # Get the codebook embeddings for each segment
        video_embedding = encoder.Cross_quantizer.embedding[:, :D]  # [M, D] - video segment
        audio_embedding = encoder.Cross_quantizer.embedding[:, D:2*D]  # [M, D] - audio segment
        
        # Flatten semantic features for distance computation
        v_flat = video_semantic_result.reshape(-1, D)  # [B*T, D]
        a_flat = audio_semantic_result.reshape(-1, D)  # [B*T, D]
        
        # Compute distances to codebook (same as in VQ_EMA)
        # Distance formula: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        v_distances = (
            torch.sum(v_flat**2, dim=1, keepdim=True) + 
            torch.sum(video_embedding**2, dim=1) - 
            2 * torch.matmul(v_flat, video_embedding.t())
        )
        v_indices = torch.argmin(v_distances, dim=1).reshape(B, T)
        
        a_distances = (
            torch.sum(a_flat**2, dim=1, keepdim=True) + 
            torch.sum(audio_embedding**2, dim=1) - 
            2 * torch.matmul(a_flat, audio_embedding.t())
        )
        a_indices = torch.argmin(a_distances, dim=1).reshape(B, T)
        
        # For AVT model, process text features
        if model_type == 'AVT' and text_feature is not None:
            text_encoder_result, text_semantic_result = encoder.text_encoder(text_feature)
            text_embedding = encoder.Cross_quantizer.embedding[:, 2*D:]  # [M, D] - text segment
            
            t_flat = text_semantic_result.reshape(-1, D)  # [B*T, D]
            
            t_distances = (
                torch.sum(t_flat**2, dim=1, keepdim=True) + 
                torch.sum(text_embedding**2, dim=1) - 
                2 * torch.matmul(t_flat, text_embedding.t())
            )
            t_indices = torch.argmin(t_distances, dim=1).reshape(B, T)
            
            return v_indices, a_indices, t_indices
        else:
            return v_indices, a_indices, None


def create_tsne_visualization(embeddings, classifications, model_type='AVT', output_path='tsne_plot.png'):
    """
    Create t-SNE visualization of codebook vectors colored by usage pattern.
    
    Args:
        embeddings: Codebook embeddings [400, 768]
        classifications: Dict mapping index to category
        model_type: 'AV' or 'AVT'
        output_path: Where to save the plot
    """
    print("Computing t-SNE projection...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Define color scheme
    if model_type == 'AVT':
        color_map = {
            'inactive': '#000000',      # Black
            'v_only': '#FF6B6B',        # Red
            'a_only': '#4ECDC4',        # Cyan
            't_only': '#95E1D3',        # Light cyan
            'a/v': '#FFA07A',           # Light salmon (av)
            'a/t': '#98D8C8',           # Cyan-green (at)
            'v/t': '#FFB6C1',           # Light pink (vt)
            'a/v/t': '#9B59B6',         # Purple (avt)
        }
        label_names = {
            'inactive': 'Inactive',
            'v_only': 'Video only',
            'a_only': 'Audio only',
            't_only': 'Text only',
            'a/v': 'Audio-Video',
            'a/t': 'Audio-Text',
            'v/t': 'Video-Text',
            'a/v/t': 'Audio-Video-Text',
        }
    else:
        color_map = {
            'inactive': '#000000',      # Black
            'v_only': '#FF6B6B',        # Red
            'a_only': '#4ECDC4',        # Cyan
            'a/v': '#FFA07A',           # Light salmon (av)
        }
        label_names = {
            'inactive': 'Inactive',
            'v_only': 'Video only',
            'a_only': 'Audio only',
            'a/v': 'Audio-Video',
        }
    
    # Count instances of each category
    category_counts = defaultdict(int)
    for cat in classifications.values():
        category_counts[cat] += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each category
    for category, color in color_map.items():
        indices = [i for i, cat in classifications.items() if cat == category]
        if len(indices) > 0:
            count = len(indices)
            percentage = (count / len(classifications)) * 100
            label = f"{label_names[category]}: {percentage:.1f}% ({count})"
            ax.scatter(
                embeddings_2d[indices, 0],
                embeddings_2d[indices, 1],
                c=color,
                label=label,
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(
        f'Codebook Vector Usage Patterns ({model_type} Model)\n'
        f'VGGSound 40k Dataset - One Epoch Analysis',
        fontsize=14,
        fontweight='bold'
    )
    
    # Legend with better positioning
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=10,
        framealpha=0.9,
        title='Modality Usage'
    )
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE visualization to: {output_path}")
    
    return fig, category_counts


def main():
    parser = argparse.ArgumentParser(description='Codebook t-SNE Visualization')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['AV', 'AVT'], required=True,
                        help='Model type: AV or AVT')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    # Model architecture parameters
    parser.add_argument('--audio_dim', type=int, default=128)
    parser.add_argument('--video_dim', type=int, default=512)
    parser.add_argument('--text_dim', type=int, default=256)
    parser.add_argument('--video_output_dim', type=int, default=2048)
    parser.add_argument('--n_embeddings', type=int, default=400)
    parser.add_argument('--embedding_dim', type=int, default=256)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"Codebook t-SNE Visualization - {args.model_type} Model")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Model architecture: {args.audio_dim}D audio, {args.video_dim}D video, {args.text_dim}D text")
    print(f"Codebook: {args.n_embeddings} vectors × {args.embedding_dim}D")
    print("="*70)
    
    # Load model
    print("\nLoading pretrained model...")
    encoder = AVT_VQVAE_Encoder(
        audio_dim=args.audio_dim,
        video_dim=args.video_dim,
        text_dim=args.text_dim,
        video_output_dim=args.video_output_dim,
        n_embeddings=args.n_embeddings,
        embedding_dim=args.embedding_dim,
        model_type=args.model_type
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if 'Encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['Encoder'])
    else:
        encoder.load_state_dict(checkpoint)
    encoder.eval()
    print("Model loaded successfully!")
    
    # Load dataset
    print("\nLoading VGGSound 40k dataset...")
    dataloader = load_vggsound_40k_dataset(model_type=args.model_type)
    print(f"Dataset loaded: {len(dataloader)} batches")
    
    # Initialize usage tracker
    tracker = CodebookUsageTracker(n_embeddings=args.n_embeddings)
    
    # Process one epoch
    print("\nProcessing one epoch to track codebook usage...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            audio_feature = batch['audio_fea'].to(device)
            video_feature = batch['video_fea'].to(device)
            text_feature = batch.get('query', None)
            
            if text_feature is not None:
                text_feature = text_feature.to(device)
            
            # Extract indices
            if args.model_type == 'AVT':
                v_indices, a_indices, t_indices = extract_quantization_indices(
                    encoder, audio_feature, video_feature, text_feature, args.model_type
                )
            else:
                v_indices, a_indices, _ = extract_quantization_indices(
                    encoder, audio_feature, video_feature, None, args.model_type
                )
                t_indices = None
            
            # Update tracker
            tracker.update(v_indices, a_indices, t_indices)
    
    # Get statistics
    stats = tracker.get_statistics()
    print("\n" + "="*70)
    print("Codebook Usage Statistics:")
    print("-"*70)
    print(f"Video modality:   {stats['video_unique']:3d} / {args.n_embeddings} unique indices")
    print(f"Audio modality:   {stats['audio_unique']:3d} / {args.n_embeddings} unique indices")
    if args.model_type == 'AVT':
        print(f"Text modality:    {stats['text_unique']:3d} / {args.n_embeddings} unique indices")
    print(f"Total active:     {stats['total_active']:3d} / {args.n_embeddings} indices")
    print(f"Inactive:         {args.n_embeddings - stats['total_active']:3d} / {args.n_embeddings} indices")
    print("="*70)
    
    # Classify indices
    print("\nClassifying codebook indices by usage pattern...")
    classifications = tracker.classify_indices(model_type=args.model_type)
    
    # Extract embeddings
    print("Extracting codebook embeddings...")
    with torch.no_grad():
        embeddings = encoder.Cross_quantizer.embedding.cpu().numpy()  # [400, 768]
    
    # Create visualization
    output_path = os.path.join(args.output_dir, f'codebook_tsne_{args.model_type}.png')
    fig, category_counts = create_tsne_visualization(
        embeddings, classifications, args.model_type, output_path
    )
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, f'codebook_stats_{args.model_type}.txt')
    with open(stats_path, 'w') as f:
        f.write("Codebook Usage Statistics\n")
        f.write("="*70 + "\n")
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Total Codebook Size: {args.n_embeddings}\n\n")
        
        f.write("Modality-specific Usage:\n")
        f.write(f"  Video: {stats['video_unique']} indices\n")
        f.write(f"  Audio: {stats['audio_unique']} indices\n")
        if args.model_type == 'AVT':
            f.write(f"  Text:  {stats['text_unique']} indices\n")
        f.write(f"\nTotal Active: {stats['total_active']} indices\n")
        f.write(f"Inactive:     {args.n_embeddings - stats['total_active']} indices\n\n")
        
        f.write("Category Breakdown:\n")
        for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            percentage = (count / args.n_embeddings) * 100
            f.write(f"  {category:15s}: {count:3d} ({percentage:5.1f}%)\n")
    
    print(f"\nStatistics saved to: {stats_path}")
    
    # Save classifications for further analysis
    classifications_path = os.path.join(args.output_dir, f'codebook_classifications_{args.model_type}.pkl')
    with open(classifications_path, 'wb') as f:
        pickle.dump({
            'classifications': classifications,
            'tracker': {
                'video': list(tracker.usage_video),
                'audio': list(tracker.usage_audio),
                'text': list(tracker.usage_text) if args.model_type == 'AVT' else []
            }
        }, f)
    print(f"Classifications saved to: {classifications_path}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()