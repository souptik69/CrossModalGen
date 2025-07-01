from model.main_model_novel import AVT_VQVAE_Encoder
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def visualize_codebook_embeddings(checkpoint_path, output_dir, top_k=25):
    """
    Visualize the top-k most used codebook vectors using PCA
    
    Args:
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save the visualization
        top_k: Number of top vectors to visualize
    """
    
    # Model configuration (matching your setup)
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    n_embeddings = 400
    embedding_dim = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize and load model
    print("Loading model...")
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim, n_embeddings, embedding_dim)
    Encoder.double()
    Encoder.to(device)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoints = torch.load(checkpoint_path, map_location=device)
    Encoder.load_state_dict(checkpoints['Encoder_parameters'])
    
    # Extract quantizer and embeddings
    quantizer = Encoder.Cross_quantizer
    embeddings = quantizer.embedding.detach().cpu().numpy()  # [400, 768]
    ema_counts = quantizer.ema_count.detach().cpu().numpy()  # [400]
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"EMA counts shape: {ema_counts.shape}")
    print(f"Total usage across all vectors: {ema_counts.sum():.2f}")
    
    # Find top-k most used vectors
    top_indices = np.argsort(ema_counts)[-top_k:][::-1]  # Descending order
    top_embeddings = embeddings[top_indices]  # [top_k, 768]
    top_counts = ema_counts[top_indices]
    
    print(f"\nTop {top_k} most used vectors:")
    for i, (idx, count) in enumerate(zip(top_indices, top_counts)):
        print(f"  Rank {i+1}: Vector {idx}, Usage: {count:.2f}")
    
    # Extract modality segments
    video_segments = top_embeddings[:, :embedding_dim]      # [top_k, 256]
    audio_segments = top_embeddings[:, embedding_dim:2*embedding_dim]  # [top_k, 256]
    text_segments = top_embeddings[:, 2*embedding_dim:]     # [top_k, 256]
    
    print(f"\nSegment shapes:")
    print(f"  Video: {video_segments.shape}")
    print(f"  Audio: {audio_segments.shape}")
    print(f"  Text: {text_segments.shape}")
    
    # Apply PCA to each modality
    print("\nApplying PCA...")
    pca_video = PCA(n_components=2)
    pca_audio = PCA(n_components=2)
    pca_text = PCA(n_components=2)
    
    video_2d = pca_video.fit_transform(video_segments)
    audio_2d = pca_audio.fit_transform(audio_segments)
    text_2d = pca_text.fit_transform(text_segments)
    
    print(f"PCA explained variance ratios:")
    print(f"  Video: {pca_video.explained_variance_ratio_}")
    print(f"  Audio: {pca_audio.explained_variance_ratio_}")
    print(f"  Text: {pca_text.explained_variance_ratio_}")

    

   # PCA for full 768-dimensional vectors
    print("\nApplying PCA to full 768-dimensional vectors...")
    pca_full = PCA(n_components=2)
    full_2d = pca_full.fit_transform(top_embeddings)  # Full 768-dim vectors

    print(f"Full vectors PCA explained variance ratios: {pca_full.explained_variance_ratio_}")

    # Create separate plot for full vectors
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a gradient colormap for usage intensity (rank-based)
    usage_colors = plt.cm.plasma(np.linspace(0, 1, top_k))

    # Plot full vectors with usage-based coloring
    scatter_full = ax.scatter(full_2d[:, 0], full_2d[:, 1], 
                            c=usage_colors, s=150, marker='D', 
                            alpha=0.8, edgecolors='black', linewidth=1.0)

    # Add vector indices as labels
    for i, (x, y) in enumerate(full_2d):
        ax.annotate(f'V{top_indices[i]}', (x, y), 
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, alpha=0.9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_title(f'Full 768D Codebook Vectors - Top {top_k} Most Used', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 (Explained Variance: {pca_full.explained_variance_ratio_[0]:.3f})', fontsize=12)
    ax.set_ylabel(f'PC2 (Explained Variance: {pca_full.explained_variance_ratio_[1]:.3f})', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add colorbar for usage ranking - Fixed version
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=1, vmax=top_k))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Usage Rank (1=Most Used, {}=Least Used)'.format(top_k), rotation=270, labelpad=20)

    plt.tight_layout()

    # Save full vectors plot
    full_output_path = os.path.join(output_dir, f'codebook_full_vectors_top{top_k}.png')
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    print(f"Full vectors visualization saved to: {full_output_path}")

    plt.show()

    # Optional: Analyze clustering in full vector space
    from sklearn.cluster import KMeans

    # Apply K-means clustering to see if there are natural groupings
    n_clusters = min(5, top_k//3)  # Reasonable number of clusters
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(top_embeddings)
        
        # Plot with cluster coloring
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        scatter_clusters = ax2.scatter(full_2d[:, 0], full_2d[:, 1], 
                                    c=cluster_labels, s=150, marker='D', 
                                    alpha=0.8, edgecolors='black', linewidth=1.0,
                                    cmap='tab10')
        
        # Add vector indices
        for i, (x, y) in enumerate(full_2d):
            ax2.annotate(f'V{top_indices[i]}', (x, y), 
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=10, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax2.set_title(f'Full 768D Vectors - K-means Clustering ({n_clusters} clusters)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 (Explained Variance: {pca_full.explained_variance_ratio_[0]:.3f})', fontsize=12)
        ax2.set_ylabel(f'PC2 (Explained Variance: {pca_full.explained_variance_ratio_[1]:.3f})', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for clusters - Fixed version
        cbar2 = plt.colorbar(scatter_clusters, ax=ax2, shrink=0.8)
        cbar2.set_label('Cluster ID', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Save clustered plot
        cluster_output_path = os.path.join(output_dir, f'codebook_full_vectors_clustered_top{top_k}.png')
        plt.savefig(cluster_output_path, dpi=300, bbox_inches='tight')
        print(f"Clustered full vectors visualization saved to: {cluster_output_path}")
        
        plt.show()
        
        print(f"\nCluster analysis for full 768D vectors:")
        for cluster_id in range(n_clusters):
            cluster_vectors = top_indices[cluster_labels == cluster_id]
            cluster_usage = top_counts[cluster_labels == cluster_id]
            print(f"  Cluster {cluster_id}: Vectors {cluster_vectors.tolist()}, Avg Usage: {cluster_usage.mean():.2f}")






    
    # Create visualization
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_k} Codebook Vectors - PCA Visualization', fontsize=16, fontweight='bold')
    
    # Color map based on usage (darker = more used)
    colors = plt.cm.viridis(np.linspace(0.3, 1.0, top_k))
    
    # Individual modality plots
    modalities = [
        (video_2d, 'Video Segments', axes[0, 0], 'o'),
        (audio_2d, 'Audio Segments', axes[0, 1], 's'),
        (text_2d, 'Text Segments', axes[1, 0], '^')
    ]
    
    for data_2d, title, ax, marker in modalities:
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                           c=colors, s=100, marker=marker, 
                           alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add vector indices as labels
        for i, (x, y) in enumerate(data_2d):
            ax.annotate(f'{top_indices[i]}', (x, y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
    
    # Combined plot
    ax_combined = axes[1, 1]
    
    # Plot all modalities together
    scatter1 = ax_combined.scatter(video_2d[:, 0], video_2d[:, 1], 
                                  c='red', s=80, marker='o', 
                                  alpha=0.7, label='Video', edgecolors='black', linewidth=0.5)
    scatter2 = ax_combined.scatter(audio_2d[:, 0], audio_2d[:, 1], 
                                  c='blue', s=80, marker='s', 
                                  alpha=0.7, label='Audio', edgecolors='black', linewidth=0.5)
    scatter3 = ax_combined.scatter(text_2d[:, 0], text_2d[:, 1], 
                                  c='green', s=80, marker='^', 
                                  alpha=0.7, label='Text', edgecolors='black', linewidth=0.5)
    
    ax_combined.set_title('All Modalities Combined', fontsize=12, fontweight='bold')
    ax_combined.set_xlabel('PC1')
    ax_combined.set_ylabel('PC2')
    ax_combined.legend()
    ax_combined.grid(True, alpha=0.3)
    
    # Add colorbar for usage intensity
    # cbar = plt.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6, aspect=30)
    # cbar.set_label('Usage Rank (Dark = More Used)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'codebook_pca_top{top_k}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    
    
    # Save detailed statistics
    stats_path = os.path.join(output_dir, f'codebook_stats_top{top_k}.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Codebook Analysis - Top {top_k} Vectors\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Total vectors: {n_embeddings}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n")
        f.write(f"Total usage: {ema_counts.sum():.2f}\n\n")
        
        f.write("PCA Explained Variance Ratios:\n")
        f.write(f"  Video: PC1={pca_video.explained_variance_ratio_[0]:.3f}, PC2={pca_video.explained_variance_ratio_[1]:.3f}\n")
        f.write(f"  Audio: PC1={pca_audio.explained_variance_ratio_[0]:.3f}, PC2={pca_audio.explained_variance_ratio_[1]:.3f}\n")
        f.write(f"  Text:  PC1={pca_text.explained_variance_ratio_[0]:.3f}, PC2={pca_text.explained_variance_ratio_[1]:.3f}\n\n")
        
        f.write(f"Top {top_k} Most Used Vectors:\n")
        f.write("-" * 30 + "\n")
        for i, (idx, count) in enumerate(zip(top_indices, top_counts)):
            f.write(f"Rank {i+1:2d}: Vector {idx:3d}, Usage: {count:8.2f}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    plt.show()
    
    return top_indices, top_counts, (video_2d, audio_2d, text_2d)


def main():
    parser = argparse.ArgumentParser(description='Visualize VQ-VAE codebook embeddings')
    parser.add_argument('--checkpoint', type=str, 
                       default="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVT_model/Best_Text_CPC_noNoise/40k/checkpoint/DCID-model-5.pt",
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, 
                       default="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/visualizations",
                       help='Output directory for saving plots')
    parser.add_argument('--top_k', type=int, default=25,
                       help='Number of top vectors to visualize')
    
    args = parser.parse_args()
    
    print("Starting codebook visualization...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top K vectors: {args.top_k}")
    
    visualize_codebook_embeddings(args.checkpoint, args.output_dir, args.top_k)
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
