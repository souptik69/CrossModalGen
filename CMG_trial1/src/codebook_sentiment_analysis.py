import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import sys
from datetime import datetime
import pandas as pd

# Add your project paths
sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/dataset")
sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src")

from dataset.MOSEI_MOSI import get_mosei_supervised_dataloaders, get_mosi_dataloaders
from model.main_model_mosei import AVT_VQVAE_Encoder

def print_analysis_progress(message):
    """Helper function for logging analysis progress"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def discretize_sentiment_labels(labels, num_bins=7):
    """
    Discretize continuous sentiment labels into bins for sampling.
    
    Args:
        labels: Tensor of continuous sentiment labels
        num_bins: Number of discrete bins to create
    
    Returns:
        discrete_labels: Discretized labels
        bin_edges: The edges used for binning
        label_mapping: Mapping from discrete labels to sentiment ranges
    """
    # Convert to numpy if tensor
    if isinstance(labels, torch.Tensor):
        labels_np = labels.numpy()
    else:
        labels_np = labels
    
    # Create bins from min to max label values
    min_label = np.min(labels_np)
    max_label = np.max(labels_np)
    bin_edges = np.linspace(min_label, max_label, num_bins + 1)
    
    # Discretize labels
    discrete_labels = np.digitize(labels_np, bin_edges[1:-1])  # Exclude the last edge for digitize
    
    # Create label mapping for interpretation
    label_mapping = {}
    for i in range(num_bins):
        label_mapping[i] = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
    
    return discrete_labels, bin_edges, label_mapping

def sample_by_sentiment_labels(dataloader, dataset_name, samples_per_label=2, num_bins=7):
    """
    Sample data points from different sentiment label bins.
    
    Args:
        dataloader: PyTorch DataLoader
        dataset_name: Name of the dataset (for identification)
        samples_per_label: Number of samples to collect per sentiment bin
        num_bins: Number of sentiment bins to create
    
    Returns:
        sampled_data: Dictionary containing sampled data organized by sentiment bins
    """
    print_analysis_progress(f"Sampling data from {dataset_name} dataset...")
    
    # First pass: collect all labels to determine bins
    all_labels = []
    all_samples = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        audio_feature = batch_data['audio_fea']
        video_feature = batch_data['video_fea'] 
        text_feature = batch_data['text_fea']
        labels = batch_data['labels']
        video_ids = batch_data['video_ids']
        attention_mask = batch_data['attention_mask']
        
        batch_size = audio_feature.shape[0]
        for i in range(batch_size):
            sample = {
                'audio': audio_feature[i].numpy(),
                'video': video_feature[i].numpy(), 
                'text': text_feature[i].numpy(),
                'label': labels[i].numpy(),
                'video_id': video_ids[i],
                'attention_mask': attention_mask[i],
                'global_idx': len(all_samples)  # Unique global index
            }
            all_samples.append(sample)
            all_labels.append(labels[i].item())
    
    print_analysis_progress(f"Collected {len(all_samples)} total samples from {dataset_name}")
    
    # Discretize labels
    all_labels = np.array(all_labels)
    discrete_labels, bin_edges, label_mapping = discretize_sentiment_labels(all_labels, num_bins)
    
    print_analysis_progress(f"Created {num_bins} sentiment bins:")
    for bin_id, range_str in label_mapping.items():
        count = np.sum(discrete_labels == bin_id)
        print_analysis_progress(f"  Bin {bin_id}: {range_str} - {count} samples")
    
    # Sample from each bin
    sampled_data = {
        'samples': {},
        'bin_info': {
            'bin_edges': bin_edges,
            'label_mapping': label_mapping,
            'dataset_name': dataset_name
        }
    }
    
    for bin_id in range(num_bins):
        bin_indices = np.where(discrete_labels == bin_id)[0]
        if len(bin_indices) >= samples_per_label:
            # Randomly sample from this bin
            selected_indices = np.random.choice(bin_indices, samples_per_label, replace=False)
        else:
            # Take all available samples if not enough
            selected_indices = bin_indices
            print_analysis_progress(f"Warning: Only {len(bin_indices)} samples available for bin {bin_id}, taking all")
        
        sampled_data['samples'][bin_id] = []
        for idx in selected_indices:
            sample = all_samples[idx].copy()
            sample['bin_id'] = bin_id
            sample['bin_range'] = label_mapping[bin_id]
            sampled_data['samples'][bin_id].append(sample)
    
    return sampled_data

def print_sample_diagnostics(sample, sample_idx, bin_id, bin_range, dataset_name):
    """
    Print comprehensive diagnostics for a single sample.
    
    Args:
        sample: Dictionary containing audio, video, text features
        sample_idx: Unique identifier for the sample
        bin_id: Sentiment bin ID
        bin_range: Sentiment range string
        dataset_name: Name of the dataset
    """
    print_analysis_progress(f"\n{'='*80}")
    print_analysis_progress(f"SAMPLE DIAGNOSTICS - {dataset_name}")
    print_analysis_progress(f"Sample Index: {sample_idx}")
    print_analysis_progress(f"Video ID: {sample['video_id']}")
    print_analysis_progress(f"Sentiment Bin: {bin_id} ({bin_range})")
    print_analysis_progress(f"Actual Label: {sample['label'].item():.4f}")
    print_analysis_progress(f"{'='*80}")
    
    # Analyze each modality
    modalities = [
        ('AUDIO', sample['audio'], 74),
        ('VIDEO', sample['video'], 35), 
        ('TEXT', sample['text'], 300)
    ]
    
    for mod_name, features, expected_dim in modalities:
        print_analysis_progress(f"\n--- {mod_name} FEATURE ANALYSIS ---")
        print_analysis_progress(f"Shape: {features.shape}")
        print_analysis_progress(f"Expected dimensions: [sequence_length, {expected_dim}]")
        print_analysis_progress(f"Data type: {features.dtype}")
        print_analysis_progress(f"Min value: {np.min(features):.6f}")
        print_analysis_progress(f"Max value: {np.max(features):.6f}")
        print_analysis_progress(f"Mean: {np.mean(features):.6f}")
        print_analysis_progress(f"Std: {np.std(features):.6f}")
        
        # Check for NaN or Inf values
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))
        print_analysis_progress(f"NaN values: {nan_count}")
        print_analysis_progress(f"Inf values: {inf_count}")
        
        # Find the actual sequence length (non-zero frames)
        seq_len = features.shape[0]
        non_zero_frames = []
        for t in range(seq_len):
            if not np.allclose(features[t], 0, atol=1e-7):
                non_zero_frames.append(t)
        
        actual_seq_len = len(non_zero_frames)
        print_analysis_progress(f"Actual sequence length (non-zero frames): {actual_seq_len}/{seq_len}")
        
        if actual_seq_len > 0:
            # Show first timestep
            first_idx = non_zero_frames[0]
            print_analysis_progress(f"First timestep (t={first_idx}) values (first 10): {features[first_idx, :10]}")
            
            # Show middle timestep if available
            if actual_seq_len > 1:
                mid_idx = non_zero_frames[actual_seq_len // 2]
                print_analysis_progress(f"Middle timestep (t={mid_idx}) values (first 10): {features[mid_idx, :10]}")
            
            # Show last timestep if different from first
            if actual_seq_len > 2:
                last_idx = non_zero_frames[-1]
                print_analysis_progress(f"Last timestep (t={last_idx}) values (first 10): {features[last_idx, :10]}")
        
        print_analysis_progress(f"--- END {mod_name} ANALYSIS ---")

def load_pretrained_encoder(checkpoint_path, model_config):
    """
    Load a pretrained encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_config: Dictionary containing model configuration
    
    Returns:
        encoder: Loaded encoder model
    """
    print_analysis_progress(f"Loading pretrained encoder from: {checkpoint_path}")
    
    # Initialize encoder with the same configuration used during training
    encoder = AVT_VQVAE_Encoder(
        audio_dim=model_config['audio_dim'],
        video_dim=model_config['video_dim'],
        text_dim=model_config['text_dim'],
        n_embeddings=model_config['n_embeddings'],
        embedding_dim=model_config['embedding_dim']
    )
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    encoder = encoder.double().to(device)
    encoder.eval()  # Set to evaluation mode
    
    print_analysis_progress(f"Encoder loaded successfully on {device}")
    return encoder

def analyze_sample_quantization(encoder, sample, sample_idx, bin_id, bin_range, dataset_name):
    """
    Analyze how a single sample gets quantized by the VQ-VAE encoder.
    
    Args:
        encoder: Pretrained encoder model
        sample: Sample data dictionary
        sample_idx: Unique sample identifier
        bin_id: Sentiment bin ID
        bin_range: Sentiment range string
        dataset_name: Dataset name
    
    Returns:
        analysis_results: Dictionary containing detailed quantization analysis
    """
    print_analysis_progress(f"\n{'='*100}")
    print_analysis_progress(f"QUANTIZATION ANALYSIS - {dataset_name} Sample {sample_idx}")
    print_analysis_progress(f"Sentiment Bin: {bin_id} ({bin_range}), Label: {sample['label'].item():.4f}")
    print_analysis_progress(f"{'='*100}")
    
    device = next(encoder.parameters()).device
    
    # Prepare input tensors (add batch dimension)
    audio_feat = torch.from_numpy(sample['audio']).unsqueeze(0).double().to(device)  # [1, seq_len, 74]
    video_feat = torch.from_numpy(sample['video']).unsqueeze(0).double().to(device)  # [1, seq_len, 35]  
    text_feat = torch.from_numpy(sample['text']).unsqueeze(0).double().to(device)   # [1, seq_len, 300]
    # attention_mask = torch.from_numpy(sample['attention_mask']).unsqueeze(0).double().to(device)
    attention_mask = sample['attention_mask'].unsqueeze(0).to(device)


    with torch.no_grad():
        # STEP 1: Get semantic representations through full encoder forward pass
        # This captures the output of temporal attention modules before quantization
        print_analysis_progress("  Step 1: Extracting semantic representations from temporal attention modules...")
        original_training_state = encoder.training
        encoder.train()
        (audio_semantic_result, audio_encoder_result, video_semantic_result, video_encoder_result,
         text_semantic_result, text_encoder_result, _, _, _, _,
         _, _, video_embedding_loss, audio_embedding_loss, text_embedding_loss,
         video_perplexity, audio_perplexity, text_perplexity, equal_num, cmcm_loss, _) = encoder(
            audio_feat, video_feat, text_feat, epoch=0, attention_mask=attention_mask)
        
        # STEP 2: Get quantized representations using individual VQ encoders
        # This shows exactly how semantic vectors get mapped to codebook indices
        encoder.train(original_training_state)
        print_analysis_progress("  Step 2: Performing quantization using individual VQ encoders...")

        encoder.eval()
        
        # Use individual VQ encoder methods for cleaner quantization analysis
        out_vq_audio, audio_vq = encoder.Audio_VQ_Encoder(audio_feat, attention_mask=attention_mask)
        out_vq_video, video_vq = encoder.Video_VQ_Encoder(video_feat, attention_mask=attention_mask)
        out_vq_text, text_vq = encoder.Text_VQ_Encoder(text_feat, attention_mask=attention_mask)
    
    # Remove batch dimension for analysis
    print_analysis_progress("  Step 3: Processing results for analysis...")
    
    # Semantic representations (output of temporal attention, before quantization)
    audio_semantic = audio_semantic_result.squeeze(0).cpu().numpy()  # [seq_len, 256]
    video_semantic = video_semantic_result.squeeze(0).cpu().numpy()  # [seq_len, 256]
    text_semantic = text_semantic_result.squeeze(0).cpu().numpy()    # [seq_len, 256]
    
    # Quantized representations (256-dim segments from codebook)
    audio_quantized = audio_vq.squeeze(0).cpu().numpy()  # [seq_len, 256]
    video_quantized = video_vq.squeeze(0).cpu().numpy()  # [seq_len, 256]
    text_quantized = text_vq.squeeze(0).cpu().numpy()   # [seq_len, 256]
    
    # Full codebook vectors (complete 768-dim vectors)
    out_vq_audio_full = out_vq_audio.squeeze(0).cpu().numpy()  # [seq_len, 768]
    out_vq_video_full = out_vq_video.squeeze(0).cpu().numpy()  # [seq_len, 768]
    out_vq_text_full = out_vq_text.squeeze(0).cpu().numpy()   # [seq_len, 768]
    
    print_analysis_progress("  Semantic and quantized representations extracted successfully!")
    
    print_analysis_progress("\n--- SEMANTIC REPRESENTATIONS (BEFORE QUANTIZATION) ---")
    print_analysis_progress(f"Audio semantic shape: {audio_semantic.shape}")
    print_analysis_progress(f"Video semantic shape: {video_semantic.shape}")
    print_analysis_progress(f"Text semantic shape: {text_semantic.shape}")
    
    # Show semantic representation statistics
    for name, semantic in [("Audio", audio_semantic), ("Video", video_semantic), ("Text", text_semantic)]:
        print_analysis_progress(f"\n{name} Semantic Statistics:")
        print_analysis_progress(f"  Min: {np.min(semantic):.6f}, Max: {np.max(semantic):.6f}")
        print_analysis_progress(f"  Mean: {np.mean(semantic):.6f}, Std: {np.std(semantic):.6f}")
        
        # Find non-zero timesteps
        non_zero_timesteps = []
        seq_len = semantic.shape[0]
        for t in range(seq_len):
            if not np.allclose(semantic[t], 0, atol=1e-7):
                non_zero_timesteps.append(t)
        
        print_analysis_progress(f"  Non-zero timesteps: {len(non_zero_timesteps)}/{seq_len}")
        if len(non_zero_timesteps) > 0:
            print_analysis_progress(f"  First non-zero timestep (t={non_zero_timesteps[0]}) first 5 dims: {semantic[non_zero_timesteps[0], :5]}")
            if len(non_zero_timesteps) > 1:
                mid_idx = len(non_zero_timesteps) // 2
                t_mid = non_zero_timesteps[mid_idx]
                print_analysis_progress(f"  Middle non-zero timestep (t={t_mid}) first 5 dims: {semantic[t_mid, :5]}")
            if len(non_zero_timesteps) > 2:
                t_last = non_zero_timesteps[-1]
                print_analysis_progress(f"  Last non-zero timestep (t={t_last}) first 5 dims: {semantic[t_last, :5]}")
    
    print_analysis_progress("\n--- QUANTIZATION INDEX ANALYSIS ---")
    
    # Get the quantizer to find which codebook indices were used
    quantizer = encoder.Cross_quantizer
    codebook_embedding = quantizer.embedding.detach().cpu().numpy()  # [n_embeddings, 768]
    
    # Analyze overall codebook usage patterns from training
    print_analysis_progress("\n--- OVERALL CODEBOOK USAGE PATTERNS ---")
    ema_counts = quantizer.ema_count.detach().cpu().numpy()  # [n_embeddings]
    
    # Find the most and least used codebook vectors
    max_used_idx = np.argmax(ema_counts)
    max_used_count = ema_counts[max_used_idx]
    min_used_idx = np.argmin(ema_counts)
    min_used_count = ema_counts[min_used_idx]
    
    # Calculate usage statistics
    total_usage = np.sum(ema_counts)
    mean_usage = np.mean(ema_counts)
    std_usage = np.std(ema_counts)
    median_usage = np.median(ema_counts)
    
    print_analysis_progress(f"Codebook Usage Statistics:")
    print_analysis_progress(f"  Total codebook vectors: {len(ema_counts)}")
    print_analysis_progress(f"  Total usage across all vectors: {total_usage:.2f}")
    print_analysis_progress(f"  Mean usage per vector: {mean_usage:.2f}")
    print_analysis_progress(f"  Median usage per vector: {median_usage:.2f}")
    print_analysis_progress(f"  Standard deviation of usage: {std_usage:.2f}")
    print_analysis_progress(f"  Usage coefficient of variation: {(std_usage/mean_usage)*100:.1f}%")
    
    print_analysis_progress(f"\nMost Used Codebook Vector:")
    print_analysis_progress(f"  Index: {max_used_idx}")
    print_analysis_progress(f"  Usage count: {max_used_count:.2f}")
    print_analysis_progress(f"  Percentage of total usage: {(max_used_count/total_usage)*100:.2f}%")
    print_analysis_progress(f"  Vector values (first 10 dims): {codebook_embedding[max_used_idx, :10]}")
    print_analysis_progress(f"  Video segment (dims 0-4): {codebook_embedding[max_used_idx, :5]}")
    print_analysis_progress(f"  Audio segment (dims 256-260): {codebook_embedding[max_used_idx, 30:35]}")
    print_analysis_progress(f"  Text segment (dims 512-516): {codebook_embedding[max_used_idx, 60:65]}")
    
    print_analysis_progress(f"\nLeast Used Codebook Vector:")
    print_analysis_progress(f"  Index: {min_used_idx}")
    print_analysis_progress(f"  Usage count: {min_used_count:.2f}")
    print_analysis_progress(f"  Percentage of total usage: {(min_used_count/total_usage)*100:.4f}%")
    print_analysis_progress(f"  Vector values (first 10 dims): {codebook_embedding[min_used_idx, :10]}")
    
    # Show top 5 most used vectors
    top_indices = np.argsort(ema_counts)[-5:][::-1]  # Top 5 in descending order
    print_analysis_progress(f"\nTop 5 Most Used Codebook Vectors:")
    for i, idx in enumerate(top_indices):
        usage_pct = (ema_counts[idx]/total_usage)*100
        print_analysis_progress(f"  Rank {i+1}: Vector {idx} (usage: {ema_counts[idx]:.2f}, {usage_pct:.2f}%)")
    
    # Show bottom 5 least used vectors
    bottom_indices = np.argsort(ema_counts)[:5]  # Bottom 5 in ascending order
    print_analysis_progress(f"\nTop 5 Least Used Codebook Vectors:")
    for i, idx in enumerate(bottom_indices):
        usage_pct = (ema_counts[idx]/total_usage)*100
        print_analysis_progress(f"  Rank {i+1}: Vector {idx} (usage: {ema_counts[idx]:.2f}, {usage_pct:.4f}%)")
    
    # Analyze usage distribution
    dead_vectors = np.sum(ema_counts < 0.01)  # Vectors with very low usage
    highly_used_vectors = np.sum(ema_counts > mean_usage + 2*std_usage)  # Vectors with very high usage
    
    print_analysis_progress(f"\nUsage Distribution Analysis:")
    print_analysis_progress(f"  Dead vectors (usage < 0.01): {dead_vectors} ({(dead_vectors/len(ema_counts))*100:.1f}%)")
    print_analysis_progress(f"  Highly used vectors (usage > mean + 2σ): {highly_used_vectors} ({(highly_used_vectors/len(ema_counts))*100:.1f}%)")
    
    # Determine usage pattern
    if std_usage/mean_usage > 1.0:
        print_analysis_progress(f"  OBSERVATION: HIGH usage variance - some vectors heavily used, others rarely used")
        print_analysis_progress(f"  → This suggests the model has found a concentrated set of preferred representations")
    elif std_usage/mean_usage > 0.5:
        print_analysis_progress(f"  OBSERVATION: MODERATE usage variance - somewhat uneven distribution")
        print_analysis_progress(f"  → This suggests partial specialization in the codebook")
    else:
        print_analysis_progress(f"  OBSERVATION: LOW usage variance - relatively even distribution")
        print_analysis_progress(f"  → This suggests the model uses the full codebook capacity fairly evenly")
    
    # Find quantization indices by computing distances for each modality
    analysis_results = {
        'sample_info': {
            'sample_idx': sample_idx,
            'video_id': sample['video_id'],
            'bin_id': bin_id,
            'bin_range': bin_range,
            'label': sample['label'].item(),
            'dataset_name': dataset_name
        },
        'semantic_representations': {
            'audio': audio_semantic,
            'video': video_semantic,
            'text': text_semantic
        },
        'quantized_representations': {
            'audio': audio_quantized,
            'video': video_quantized, 
            'text': text_quantized
        },
        'full_quantized_vectors': {
            'audio': out_vq_audio_full,
            'video': out_vq_video_full,
            'text': out_vq_text_full
        },
        'quantization_indices': {},
        'codebook_matches': {}
    }
    
    # For each modality, find the quantization indices
    modalities = [
        ('audio', audio_semantic, audio_quantized, out_vq_audio_full, slice(30, 60)),  # Audio segment: dims 256-511
        ('video', video_semantic, video_quantized, out_vq_video_full, slice(0, 30)),    # Video segment: dims 0-255
        ('text', text_semantic, text_quantized, out_vq_text_full, slice(60, 90))      # Text segment: dims 512-767
    ]
    
    for mod_name, semantic, quantized, full_vq, codebook_slice in modalities:
        print_analysis_progress(f"\n{mod_name.upper()} Quantization Analysis:")
        
        # Get the relevant segment of the codebook for this modality
        modality_codebook = codebook_embedding[:, codebook_slice]  # [n_embeddings, 256]
        
        seq_len = semantic.shape[0]
        content_quantization_indices = []  # For non-zero (content) timesteps
        padding_quantization_indices = []  # For zero (padding) timesteps
        
        # Classify timesteps as content vs padding based on semantic representations
        content_timesteps = []
        padding_timesteps = []
        
        for t in range(seq_len):
            if not np.allclose(semantic[t], 0, atol=1e-7):
                content_timesteps.append(t)
            else:
                padding_timesteps.append(t)
        
        print_analysis_progress(f"  Found {len(content_timesteps)} content timesteps and {len(padding_timesteps)} padding timesteps")
        
        # Analyze content timesteps (non-zero semantic representations)
        if len(content_timesteps) > 0:
            print_analysis_progress(f"\n  --- CONTENT TIMESTEPS ANALYSIS ---")
            for t in content_timesteps:
                # Find closest codebook vector by comparing the quantized result with codebook segments
                distances = np.sum((modality_codebook - quantized[t])**2, axis=1)
                closest_idx = np.argmin(distances)
                content_quantization_indices.append((t, closest_idx, distances[closest_idx]))
                
                print_analysis_progress(f"  Timestep {t} (CONTENT): Quantized by codebook vector {closest_idx} (distance: {distances[closest_idx]:.6f})")
                print_analysis_progress(f"    Semantic vector first 5 dims: {semantic[t, :5]}")
                print_analysis_progress(f"    Quantized vector first 5 dims: {quantized[t, :5]}")
                print_analysis_progress(f"    Codebook vector first 5 dims: {modality_codebook[closest_idx, :5]}")
                print_analysis_progress(f"    Full codebook vector first 5 dims: {codebook_embedding[closest_idx, :5]}")
        
        # Analyze padding timesteps (zero semantic representations)  
        if len(padding_timesteps) > 0:
            print_analysis_progress(f"\n  --- PADDING TIMESTEPS ANALYSIS ---")
            
            # Collect all padding quantization indices to find patterns
            padding_indices_used = []
            
            for t in padding_timesteps:
                # Even though semantic is zero, the quantization process still assigns a codebook vector
                distances = np.sum((modality_codebook - quantized[t])**2, axis=1)
                closest_idx = np.argmin(distances)
                padding_quantization_indices.append((t, closest_idx, distances[closest_idx]))
                padding_indices_used.append(closest_idx)
                
                # Only show detailed info for first few padding timesteps to avoid spam
                if len(padding_quantization_indices) <= 3:
                    print_analysis_progress(f"  Timestep {t} (PADDING): Quantized by codebook vector {closest_idx} (distance: {distances[closest_idx]:.6f})")
                    print_analysis_progress(f"    Semantic vector (should be ~0): {semantic[t, :5]}")
                    print_analysis_progress(f"    Quantized vector first 5 dims: {quantized[t, :5]}")
                    print_analysis_progress(f"    Codebook vector first 5 dims: {modality_codebook[closest_idx, :5]}")
                    print_analysis_progress(f"    Full codebook vector first 5 dims: {codebook_embedding[closest_idx, :5]}")
            
            # Analyze padding patterns
            from collections import Counter
            padding_counter = Counter(padding_indices_used)
            print_analysis_progress(f"\n  PADDING QUANTIZATION PATTERNS:")
            print_analysis_progress(f"    Total padding timesteps: {len(padding_timesteps)}")
            print_analysis_progress(f"    Unique codebook vectors used for padding: {len(padding_counter)}")
            print_analysis_progress(f"    Most frequent padding indices: {padding_counter.most_common(5)}")
            
            # Calculate max used and average used codebook indices
            if len(padding_counter) > 0:
                # Max used codebook index (most frequent)
                max_used_idx, max_count = padding_counter.most_common(1)[0]
                print_analysis_progress(f"    Max used codebook index: {max_used_idx} (used {max_count} times)")
                
                # Average used codebook index (weighted by frequency)
                total_usage = sum(padding_counter.values())
                weighted_sum = sum(index * count for index, count in padding_counter.items())
                avg_used_idx = weighted_sum / total_usage
                print_analysis_progress(f"    Average used codebook index: {avg_used_idx:.2f} (frequency-weighted)")
                
                # Additional distribution statistics
                all_indices = list(padding_counter.keys())
                min_idx = min(all_indices)
                max_idx = max(all_indices)
                idx_range = max_idx - min_idx
                print_analysis_progress(f"    Codebook index range for padding: [{min_idx}, {max_idx}] (span: {idx_range})")
            
            # Check if padding uses consistent vs diverse indices
            if len(padding_counter) == 1:
                single_idx = list(padding_counter.keys())[0]
                print_analysis_progress(f"    OBSERVATION: All padding uses the SAME codebook vector {single_idx}")
                print_analysis_progress(f"    → This suggests the model learned a dedicated 'padding' representation")
            elif len(padding_counter) < len(padding_timesteps) * 0.5:
                print_analysis_progress(f"    OBSERVATION: Padding uses a LIMITED set of codebook vectors")
                print_analysis_progress(f"    → This suggests some specialization for padding representations")
            else:
                print_analysis_progress(f"    OBSERVATION: Padding uses DIVERSE codebook vectors")
                print_analysis_progress(f"    → This suggests padding may not have specialized representations")
        
        # Store results separately for content and padding
        analysis_results['quantization_indices'][mod_name] = {
            'content': content_quantization_indices,
            'padding': padding_quantization_indices,
            'content_timesteps': content_timesteps,
            'padding_timesteps': padding_timesteps
        }
    
    print_analysis_progress(f"\n{'='*100}")
    print_analysis_progress(f"COMPLETED QUANTIZATION ANALYSIS FOR SAMPLE {sample_idx}")
    print_analysis_progress(f"{'='*100}")
    
    return analysis_results

# def create_comprehensive_analysis_report(sampled_datasets, checkpoint_path, model_config, output_dir):
#     """
#     Create a comprehensive analysis report for all sampled data.
    
#     Args:
#         sampled_datasets: Dictionary of sampled datasets (MOSEI and MOSI)
#         checkpoint_path: Path to pretrained model checkpoint
#         model_config: Model configuration dictionary
#         output_dir: Directory to save analysis results
#     """
#     print_analysis_progress(f"\n{'='*120}")
#     print_analysis_progress("COMPREHENSIVE CODEBOOK-SENTIMENT ANALYSIS")
#     print_analysis_progress(f"{'='*120}")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load pretrained encoder
#     encoder = load_pretrained_encoder(checkpoint_path, model_config)
    
#     all_analyses = {}
    
#     for dataset_name, dataset_samples in sampled_datasets.items():
#         print_analysis_progress(f"\n{'#'*80}")
#         print_analysis_progress(f"ANALYZING {dataset_name.upper()} DATASET")
#         print_analysis_progress(f"{'#'*80}")
        
#         bin_info = dataset_samples['bin_info']
#         samples = dataset_samples['samples']
        
#         dataset_analyses = {}
        
#         for bin_id, bin_samples in samples.items():
#             print_analysis_progress(f"\n--- Analyzing Sentiment Bin {bin_id}: {bin_info['label_mapping'][bin_id]} ---")
            
#             bin_analyses = []
            
#             for i, sample in enumerate(bin_samples):
#                 print_analysis_progress(f"\nProcessing sample {i+1}/{len(bin_samples)} from bin {bin_id}...")
                
#                 # Print sample diagnostics
#                 sample_idx = sample['global_idx']
#                 print_sample_diagnostics(sample, sample_idx, bin_id, bin_info['label_mapping'][bin_id], dataset_name)
                
#                 # Analyze quantization
#                 analysis = analyze_sample_quantization(
#                     encoder, sample, sample_idx, bin_id, 
#                     bin_info['label_mapping'][bin_id], dataset_name
#                 )
                
#                 bin_analyses.append(analysis)
            
#             dataset_analyses[bin_id] = bin_analyses
        
#         all_analyses[dataset_name] = dataset_analyses
    
#     # Save comprehensive analysis
#     analysis_summary_path = os.path.join(output_dir, 'codebook_sentiment_analysis_summary.txt')
#     print_analysis_progress(f"\nSaving comprehensive analysis summary to: {analysis_summary_path}")
    
#     with open(analysis_summary_path, 'w') as f:
#         f.write("COMPREHENSIVE CODEBOOK-SENTIMENT ANALYSIS SUMMARY\n")
#         f.write("="*80 + "\n\n")
#         f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Checkpoint path: {checkpoint_path}\n")
#         f.write(f"Model config: {model_config}\n\n")
        
#         for dataset_name, dataset_analyses in all_analyses.items():
#             f.write(f"\n{dataset_name.upper()} DATASET ANALYSIS\n")
#             f.write("-" * 50 + "\n")
            
#             for bin_id, bin_analyses in dataset_analyses.items():
#                 f.write(f"\nSentiment Bin {bin_id}:\n")
                
#                 # Collect all quantization indices used in this bin
#                 audio_indices = []
#                 video_indices = []
#                 text_indices = []
                
#                 for analysis in bin_analyses:
#                     audio_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['audio']])
#                     video_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['video']])
#                     text_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['text']])
                
#                 # Count frequency of codebook usage
#                 audio_counter = Counter(audio_indices)
#                 video_counter = Counter(video_indices)  
#                 text_counter = Counter(text_indices)
                
#                 f.write(f"  Samples analyzed: {len(bin_analyses)}\n")
#                 f.write(f"  Most used audio codebook vectors: {audio_counter.most_common(5)}\n")
#                 f.write(f"  Most used video codebook vectors: {video_counter.most_common(5)}\n")
#                 f.write(f"  Most used text codebook vectors: {text_counter.most_common(5)}\n\n")
    
#     print_analysis_progress(f"Analysis complete! Results saved to {output_dir}")
#     return all_analyses

def create_comprehensive_analysis_report(sampled_datasets, checkpoint_path, model_config, output_dir):
    """
    Create a comprehensive analysis report for all sampled data.
    
    Args:
        sampled_datasets: Dictionary of sampled datasets (MOSEI and MOSI)
        checkpoint_path: Path to pretrained model checkpoint
        model_config: Model configuration dictionary
        output_dir: Directory to save analysis results
    """
    print_analysis_progress(f"\n{'='*120}")
    print_analysis_progress("COMPREHENSIVE CODEBOOK-SENTIMENT ANALYSIS")
    print_analysis_progress(f"{'='*120}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(checkpoint_path, model_config)
    
    all_analyses = {}
    
    for dataset_name, dataset_samples in sampled_datasets.items():
        print_analysis_progress(f"\n{'#'*80}")
        print_analysis_progress(f"ANALYZING {dataset_name.upper()} DATASET")
        print_analysis_progress(f"{'#'*80}")
        
        bin_info = dataset_samples['bin_info']
        samples = dataset_samples['samples']
        
        dataset_analyses = {}
        
        for bin_id, bin_samples in samples.items():
            print_analysis_progress(f"\n--- Analyzing Sentiment Bin {bin_id}: {bin_info['label_mapping'][bin_id]} ---")
            
            bin_analyses = []
            
            for i, sample in enumerate(bin_samples):
                print_analysis_progress(f"\nProcessing sample {i+1}/{len(bin_samples)} from bin {bin_id}...")
                
                # Print sample diagnostics
                sample_idx = sample['global_idx']
                print_sample_diagnostics(sample, sample_idx, bin_id, bin_info['label_mapping'][bin_id], dataset_name)
                
                # Analyze quantization
                analysis = analyze_sample_quantization(
                    encoder, sample, sample_idx, bin_id, 
                    bin_info['label_mapping'][bin_id], dataset_name
                )
                
                bin_analyses.append(analysis)
            
            dataset_analyses[bin_id] = bin_analyses
        
        all_analyses[dataset_name] = dataset_analyses
    
    # Save comprehensive analysis
    analysis_summary_path = os.path.join(output_dir, 'codebook_sentiment_analysis_summary.txt')
    print_analysis_progress(f"\nSaving comprehensive analysis summary to: {analysis_summary_path}")
    
    with open(analysis_summary_path, 'w') as f:
        f.write("COMPREHENSIVE CODEBOOK-SENTIMENT ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint path: {checkpoint_path}\n")
        f.write(f"Model config: {model_config}\n\n")
        
        for dataset_name, dataset_analyses in all_analyses.items():
            f.write(f"\n{dataset_name.upper()} DATASET ANALYSIS\n")
            f.write("-" * 50 + "\n")
            
            for bin_id, bin_analyses in dataset_analyses.items():
                f.write(f"\nSentiment Bin {bin_id}:\n")
                
                # Collect all quantization indices used in this bin, separating content and padding
                audio_content_indices = []
                video_content_indices = []
                text_content_indices = []
                audio_padding_indices = []
                video_padding_indices = []
                text_padding_indices = []
                
                for analysis in bin_analyses:
                    # Extract content indices (actual meaningful timesteps)
                    audio_content_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['audio']['content']])
                    video_content_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['video']['content']])
                    text_content_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['text']['content']])
                    
                    # Extract padding indices (zero timesteps)
                    audio_padding_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['audio']['padding']])
                    video_padding_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['video']['padding']])
                    text_padding_indices.extend([idx for t, idx, dist in analysis['quantization_indices']['text']['padding']])
                
                # Count frequency of codebook usage for content timesteps
                audio_content_counter = Counter(audio_content_indices)
                video_content_counter = Counter(video_content_indices)  
                text_content_counter = Counter(text_content_indices)
                
                # Count frequency of codebook usage for padding timesteps
                audio_padding_counter = Counter(audio_padding_indices)
                video_padding_counter = Counter(video_padding_indices)
                text_padding_counter = Counter(text_padding_indices)
                
                f.write(f"  Samples analyzed: {len(bin_analyses)}\n")
                f.write(f"  \n")
                f.write(f"  CONTENT TIMESTEPS (meaningful data):\n")
                f.write(f"    Most used audio codebook vectors: {audio_content_counter.most_common(5)}\n")
                f.write(f"    Most used video codebook vectors: {video_content_counter.most_common(5)}\n")
                f.write(f"    Most used text codebook vectors: {text_content_counter.most_common(5)}\n")
                f.write(f"    Total content timesteps - Audio: {len(audio_content_indices)}, Video: {len(video_content_indices)}, Text: {len(text_content_indices)}\n")
                f.write(f"  \n")
                f.write(f"  PADDING TIMESTEPS (zero-padded frames):\n")
                if len(audio_padding_indices) > 0:
                    f.write(f"    Most used audio padding vectors: {audio_padding_counter.most_common(5)}\n")
                    f.write(f"    Audio padding diversity: {len(audio_padding_counter)} unique vectors for {len(audio_padding_indices)} padding timesteps\n")
                else:
                    f.write(f"    No audio padding timesteps found\n")
                    
                if len(video_padding_indices) > 0:
                    f.write(f"    Most used video padding vectors: {video_padding_counter.most_common(5)}\n")
                    f.write(f"    Video padding diversity: {len(video_padding_counter)} unique vectors for {len(video_padding_indices)} padding timesteps\n")
                else:
                    f.write(f"    No video padding timesteps found\n")
                    
                if len(text_padding_indices) > 0:
                    f.write(f"    Most used text padding vectors: {text_padding_counter.most_common(5)}\n")
                    f.write(f"    Text padding diversity: {len(text_padding_counter)} unique vectors for {len(text_padding_indices)} padding timesteps\n")
                else:
                    f.write(f"    No text padding timesteps found\n")
                
                # Analysis of content vs padding patterns
                f.write(f"  \n")
                f.write(f"  CONTENT vs PADDING ANALYSIS:\n")
                for mod_name, content_counter, padding_counter in [
                    ('Audio', audio_content_counter, audio_padding_counter),
                    ('Video', video_content_counter, video_padding_counter),
                    ('Text', text_content_counter, text_padding_counter)
                ]:
                    if len(content_counter) > 0 and len(padding_counter) > 0:
                        # Check if there's overlap between content and padding vectors
                        content_vectors = set(content_counter.keys())
                        padding_vectors = set(padding_counter.keys())
                        overlap = content_vectors.intersection(padding_vectors)
                        
                        f.write(f"    {mod_name}: {len(overlap)} vectors used for both content and padding\n")
                        if len(overlap) == 0:
                            f.write(f"      → Perfect separation: content and padding use completely different vectors\n")
                        elif len(overlap) < min(len(content_vectors), len(padding_vectors)) * 0.3:
                            f.write(f"      → Good separation: minimal overlap between content and padding representations\n")
                        else:
                            f.write(f"      → Poor separation: significant overlap between content and padding vectors\n")
                    elif len(padding_counter) == 0:
                        f.write(f"    {mod_name}: No padding timesteps found (sequences may be full-length)\n")
                    else:
                        f.write(f"    {mod_name}: No content timesteps found (unusual - check data)\n")
                
                f.write(f"\n")
    
    print_analysis_progress(f"Analysis complete! Results saved to {output_dir}")
    return all_analyses






def create_codebook_sentiment_visualizations(sampled_datasets, checkpoint_path, model_config, output_dir):
    """
    Create visualizations showing how codebook indices are distributed across sentiment bins.
    
    Args:
        sampled_datasets: Dictionary of sampled datasets (MOSEI and MOSI)
        checkpoint_path: Path to pretrained model checkpoint
        model_config: Model configuration dictionary
        output_dir: Directory to save visualization results
    """
    print_analysis_progress(f"\n{'='*120}")
    print_analysis_progress("CREATING CODEBOOK-SENTIMENT VISUALIZATIONS")
    print_analysis_progress(f"{'='*120}")
    
    # Create visualization subdirectory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(checkpoint_path, model_config)
    
    # Set up plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Collect all quantization data
    all_data = {}
    
    for dataset_name, dataset_samples in sampled_datasets.items():
        print_analysis_progress(f"\nProcessing {dataset_name} for visualization...")
        
        bin_info = dataset_samples['bin_info']
        samples = dataset_samples['samples']
        
        dataset_data = {
            'content_indices': {'audio': {}, 'video': {}, 'text': {}},
            'padding_indices': {'audio': {}, 'video': {}, 'text': {}},
            'sentiment_labels': {},
            'bin_info': bin_info
        }
        
        for bin_id, bin_samples in samples.items():
            # Initialize storage for this bin
            for modality in ['audio', 'video', 'text']:
                dataset_data['content_indices'][modality][bin_id] = []
                dataset_data['padding_indices'][modality][bin_id] = []
            dataset_data['sentiment_labels'][bin_id] = []
            
            for sample in bin_samples:
                # Analyze quantization for this sample
                analysis = analyze_sample_quantization(
                    encoder, sample, sample['global_idx'], bin_id, 
                    bin_info['label_mapping'][bin_id], dataset_name
                )
                
                # Extract indices for each modality
                for modality in ['audio', 'video', 'text']:
                    content_indices = [idx for t, idx, dist in analysis['quantization_indices'][modality]['content']]
                    padding_indices = [idx for t, idx, dist in analysis['quantization_indices'][modality]['padding']]
                    
                    dataset_data['content_indices'][modality][bin_id].extend(content_indices)
                    dataset_data['padding_indices'][modality][bin_id].extend(padding_indices)
                
                dataset_data['sentiment_labels'][bin_id].append(sample['label'].item())
        
        all_data[dataset_name] = dataset_data
    
    # Create visualizations
    print_analysis_progress("Creating visualizations...")
    
    # 1. Codebook Usage Heatmaps by Sentiment Bin
    create_codebook_heatmaps(all_data, viz_dir)
    
    # 2. Top Codebook Indices Bar Plots
    create_top_indices_barplots(all_data, viz_dir)
    
    # 3. Codebook Diversity Analysis
    create_diversity_analysis(all_data, viz_dir)
    
    # 4. Content vs Padding Comparison
    # create_content_padding_comparison(all_data, viz_dir)
    
    # 5. Sentiment-Codebook Correlation Analysis
    create_sentiment_correlation_analysis(all_data, viz_dir)
    
    print_analysis_progress(f"Visualizations saved to: {viz_dir}")


def create_codebook_heatmaps(all_data, viz_dir):
    """Create heatmaps showing codebook usage across sentiment bins."""
    print_analysis_progress("Creating codebook usage heatmaps...")
    
    for dataset_name, dataset_data in all_data.items():
        bin_info = dataset_data['bin_info']
        n_bins = len(bin_info['label_mapping'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name} - Codebook Usage Heatmaps', fontsize=16, fontweight='bold')
        
        modalities = ['audio', 'video', 'text']
        data_types = ['content_indices', 'padding_indices']
        
        for i, data_type in enumerate(data_types):
            for j, modality in enumerate(modalities):
                ax = axes[i, j]
                
                # Collect usage data for heatmap
                codebook_usage = defaultdict(lambda: defaultdict(int))
                all_indices = set()
                
                for bin_id in range(n_bins):
                    indices = dataset_data[data_type][modality].get(bin_id, [])
                    counter = Counter(indices)
                    
                    for idx, count in counter.items():
                        codebook_usage[bin_id][idx] = count
                        all_indices.add(idx)
                
                if all_indices:
                    # Create matrix for heatmap
                    sorted_indices = sorted(all_indices)
                    matrix = np.zeros((n_bins, len(sorted_indices)))
                    
                    for bin_idx, bin_id in enumerate(range(n_bins)):
                        for idx_pos, codebook_idx in enumerate(sorted_indices):
                            matrix[bin_idx, idx_pos] = codebook_usage[bin_id][codebook_idx]
                    
                    # Plot heatmap
                    sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=True, 
                               xticklabels=[f'{idx}' for idx in sorted_indices[::max(1, len(sorted_indices)//10)]], 
                               yticklabels=[f'Bin {i}' for i in range(n_bins)])
                    
                    title_suffix = "Content" if data_type == 'content_indices' else "Padding"
                    ax.set_title(f'{modality.title()} - {title_suffix}')
                    ax.set_xlabel('Codebook Index')
                    ax.set_ylabel('Sentiment Bin')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{modality.title()} - {"Content" if data_type == "content_indices" else "Padding"}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{dataset_name}_codebook_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_top_indices_barplots(all_data, viz_dir):
    """Create bar plots showing top codebook indices for each sentiment bin."""
    print_analysis_progress("Creating top indices bar plots...")
    
    for dataset_name, dataset_data in all_data.items():
        bin_info = dataset_data['bin_info']
        n_bins = len(bin_info['label_mapping'])
        
        modalities = ['audio', 'video', 'text']
        
        for modality in modalities:
            fig, axes = plt.subplots(2, (n_bins + 1) // 2, figsize=(4 * ((n_bins + 1) // 2), 8))
            if n_bins == 1:
                axes = [axes]
            elif len(axes.shape) == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'{dataset_name} - Top {modality.title()} Codebook Indices by Sentiment Bin', 
                        fontsize=14, fontweight='bold')
            
            for bin_id in range(n_bins):
                row = bin_id // ((n_bins + 1) // 2)
                col = bin_id % ((n_bins + 1) // 2)
                ax = axes[row, col] if len(axes.shape) > 1 else axes[col]
                
                # Get content indices for this bin
                content_indices = dataset_data['content_indices'][modality].get(bin_id, [])
                
                if content_indices:
                    counter = Counter(content_indices)
                    top_indices = counter.most_common(10)
                    
                    indices, counts = zip(*top_indices)
                    
                    bars = ax.bar(range(len(indices)), counts, alpha=0.7)
                    ax.set_xlabel('Codebook Index')
                    ax.set_ylabel('Usage Count')
                    ax.set_title(f'Bin {bin_id}: {bin_info["label_mapping"][bin_id]}')
                    ax.set_xticks(range(len(indices)))
                    ax.set_xticklabels([str(idx) for idx in indices], rotation=45)
                    
                    # Add value labels on bars
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{count}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No Content Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Bin {bin_id}: {bin_info["label_mapping"][bin_id]}')
            
            # Hide empty subplots
            for bin_id in range(n_bins, len(axes.flat)):
                axes.flat[bin_id].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{dataset_name}_{modality}_top_indices.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()


def create_diversity_analysis(all_data, viz_dir):
    """Create plots analyzing codebook diversity across sentiment bins."""
    print_analysis_progress("Creating diversity analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Codebook Diversity Analysis Across Datasets', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of unique indices per bin
    ax1 = axes[0, 0]
    
    for dataset_name, dataset_data in all_data.items():
        bin_info = dataset_data['bin_info']
        n_bins = len(bin_info['label_mapping'])
        
        modalities = ['audio', 'video', 'text']
        bin_ids = list(range(n_bins))
        
        for modality in modalities:
            unique_counts = []
            for bin_id in bin_ids:
                content_indices = dataset_data['content_indices'][modality].get(bin_id, [])
                unique_counts.append(len(set(content_indices)))
            
            ax1.plot(bin_ids, unique_counts, marker='o', label=f'{dataset_name}-{modality}', alpha=0.7)
    
    ax1.set_xlabel('Sentiment Bin')
    ax1.set_ylabel('Number of Unique Codebook Indices')
    ax1.set_title('Codebook Diversity by Sentiment Bin')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total usage per bin
    ax2 = axes[0, 1]
    
    for dataset_name, dataset_data in all_data.items():
        bin_info = dataset_data['bin_info']
        n_bins = len(bin_info['label_mapping'])
        
        for modality in modalities:
            total_counts = []
            for bin_id in range(n_bins):
                content_indices = dataset_data['content_indices'][modality].get(bin_id, [])
                total_counts.append(len(content_indices))
            
            ax2.plot(range(n_bins), total_counts, marker='s', label=f'{dataset_name}-{modality}', alpha=0.7)
    
    ax2.set_xlabel('Sentiment Bin')
    ax2.set_ylabel('Total Codebook Usage')
    ax2.set_title('Total Codebook Usage by Sentiment Bin')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Content vs Padding ratio
    ax3 = axes[1, 0]
    
    for dataset_name, dataset_data in all_data.items():
        bin_info = dataset_data['bin_info']
        n_bins = len(bin_info['label_mapping'])
        
        for modality in modalities:
            ratios = []
            for bin_id in range(n_bins):
                content_count = len(dataset_data['content_indices'][modality].get(bin_id, []))
                padding_count = len(dataset_data['padding_indices'][modality].get(bin_id, []))
                total = content_count + padding_count
                ratio = content_count / total if total > 0 else 0
                ratios.append(ratio)
            
            ax3.plot(range(n_bins), ratios, marker='^', label=f'{dataset_name}-{modality}', alpha=0.7)
    
    ax3.set_xlabel('Sentiment Bin')
    ax3.set_ylabel('Content/(Content+Padding) Ratio')
    ax3.set_title('Content vs Padding Ratio by Sentiment Bin')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Average sentiment per bin
    ax4 = axes[1, 1]
    
    for dataset_name, dataset_data in all_data.items():
        bin_info = dataset_data['bin_info']
        n_bins = len(bin_info['label_mapping'])
        
        avg_sentiments = []
        bin_labels = []
        
        for bin_id in range(n_bins):
            sentiments = dataset_data['sentiment_labels'].get(bin_id, [])
            if sentiments:
                avg_sentiments.append(np.mean(sentiments))
                bin_labels.append(f'Bin {bin_id}')
        
        if avg_sentiments:
            bars = ax4.bar([f'{dataset_name}\n{label}' for label in bin_labels], avg_sentiments, 
                          alpha=0.7, label=dataset_name)
            
            # Add value labels on bars
            for bar, val in zip(bars, avg_sentiments):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_ylabel('Average Sentiment Score')
    ax4.set_title('Average Sentiment Score by Bin')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'diversity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_content_padding_comparison(all_data, viz_dir):
    """Create visualizations comparing content vs padding codebook usage."""
    print_analysis_progress("Creating content vs padding comparison plots...")
    
    for dataset_name, dataset_data in all_data.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{dataset_name} - Content vs Padding Codebook Usage', fontsize=14, fontweight='bold')
        
        modalities = ['audio', 'video', 'text']
        
        for j, modality in enumerate(modalities):
            ax = axes[j]
            
            # Collect all content and padding indices across all bins
            all_content_indices = []
            all_padding_indices = []
            
            bin_info = dataset_data['bin_info']
            n_bins = len(bin_info['label_mapping'])
            
            for bin_id in range(n_bins):
                all_content_indices.extend(dataset_data['content_indices'][modality].get(bin_id, []))
                all_padding_indices.extend(dataset_data['padding_indices'][modality].get(bin_id, []))
            
            # Create histograms
            if all_content_indices:
                content_counter = Counter(all_content_indices)
                content_indices, content_counts = zip(*content_counter.most_common(20))
                
                x_pos = np.arange(len(content_indices))
                ax.bar(x_pos - 0.2, content_counts, 0.4, label='Content', alpha=0.7)
            
            if all_padding_indices:
                padding_counter = Counter(all_padding_indices)
                padding_indices, padding_counts = zip(*padding_counter.most_common(20))
                
                # Align with content indices for comparison
                padding_counts_aligned = []
                for idx in content_indices if all_content_indices else padding_indices:
                    padding_counts_aligned.append(padding_counter.get(idx, 0))
                
                x_pos = np.arange(len(content_indices) if all_content_indices else len(padding_indices))
                ax.bar(x_pos + 0.2, padding_counts_aligned if all_content_indices else padding_counts, 
                      0.4, label='Padding', alpha=0.7)
            
            ax.set_xlabel('Codebook Index')
            ax.set_ylabel('Usage Count')
            ax.set_title(f'{modality.title()} Modality')
            ax.legend()
            
            # Set x-tick labels
            if all_content_indices:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(idx) for idx in content_indices], rotation=45)
            elif all_padding_indices:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(idx) for idx in padding_indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{dataset_name}_content_padding_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def create_sentiment_correlation_analysis(all_data, viz_dir):
    """Create scatter plots showing correlation between sentiment and codebook usage patterns."""
    print_analysis_progress("Creating sentiment correlation analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sentiment vs Codebook Usage Correlation Analysis', fontsize=16, fontweight='bold')
    
    modalities = ['audio', 'video', 'text']
    
    for j, modality in enumerate(modalities):
        # Top subplot: Unique indices vs sentiment
        ax1 = axes[0, j]
        # Bottom subplot: Total usage vs sentiment
        ax2 = axes[1, j]
        
        for dataset_name, dataset_data in all_data.items():
            bin_info = dataset_data['bin_info']
            n_bins = len(bin_info['label_mapping'])
            
            sentiments = []
            unique_counts = []
            total_counts = []
            
            for bin_id in range(n_bins):
                # Get average sentiment for this bin
                bin_sentiments = dataset_data['sentiment_labels'].get(bin_id, [])
                if bin_sentiments:
                    avg_sentiment = np.mean(bin_sentiments)
                    sentiments.append(avg_sentiment)
                    
                    # Get codebook usage statistics
                    content_indices = dataset_data['content_indices'][modality].get(bin_id, [])
                    unique_counts.append(len(set(content_indices)))
                    total_counts.append(len(content_indices))
            
            if sentiments:
                # Plot unique indices vs sentiment
                ax1.scatter(sentiments, unique_counts, alpha=0.7, s=100, label=dataset_name)
                
                # Plot total usage vs sentiment
                ax2.scatter(sentiments, total_counts, alpha=0.7, s=100, label=dataset_name)
        
        ax1.set_xlabel('Average Sentiment Score')
        ax1.set_ylabel('Number of Unique Codebook Indices')
        ax1.set_title(f'{modality.title()} - Diversity vs Sentiment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Average Sentiment Score')
        ax2.set_ylabel('Total Codebook Usage')
        ax2.set_title(f'{modality.title()} - Usage vs Sentiment')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sentiment_correlation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the comprehensive codebook-sentiment analysis.
    """
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Comprehensive Codebook-Sentiment Analysis for VQ-VAE Models')
    
    # Model and checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save analysis results')
    
    # Analysis configuration arguments
    parser.add_argument('--samples_per_label', type=int, default=2,
                        help='Number of samples to collect per sentiment bin (default: 2)')
    parser.add_argument('--num_sentiment_bins', type=int, default=7,
                        help='Number of sentiment bins to create (default: 7)')
    
    # Data loading arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading (default: 32)')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length (default: 50)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Model configuration arguments
    parser.add_argument('--audio_dim', type=int, default=74,
                        help='Audio feature dimension (default: 74)')
    parser.add_argument('--video_dim', type=int, default=35,
                        help='Video feature dimension (default: 35)')
    parser.add_argument('--text_dim', type=int, default=300,
                        help='Text feature dimension (default: 300)')
    parser.add_argument('--n_embeddings', type=int, default=256,
                        help='Number of codebook embeddings (default: 256)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension (default: 256)')
    
    parser.add_argument('--skip_analysis', action='store_true', default=False,
                        help='Skip detailed text analysis and only create visualizations')
    parser.add_argument('--create_visualizations', action='store_true', default=True,
                        help='Create visualization plots (default: True)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create model configuration dictionary
    MODEL_CONFIG = {
        'audio_dim': args.audio_dim,
        'video_dim': args.video_dim,
        'text_dim': args.text_dim,
        'n_embeddings': args.n_embeddings,
        'embedding_dim': args.embedding_dim
    }
    
    print_analysis_progress("Starting comprehensive codebook-sentiment analysis...")
    print_analysis_progress(f"Configuration:")
    print_analysis_progress(f"  Checkpoint: {args.checkpoint_path}")
    print_analysis_progress(f"  Output directory: {args.output_dir}")
    print_analysis_progress(f"  Samples per label: {args.samples_per_label}")
    print_analysis_progress(f"  Sentiment bins: {args.num_sentiment_bins}")
    print_analysis_progress(f"  Model config: {MODEL_CONFIG}")
    print_analysis_progress(f"  Create visualizations: {args.create_visualizations}")
    print_analysis_progress(f"  Skip detailed analysis: {args.skip_analysis}")
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print_analysis_progress(f"ERROR: Checkpoint file not found at {args.checkpoint_path}")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print_analysis_progress(f"Output directory created: {args.output_dir}")
    
    # Load datasets
    print_analysis_progress("Loading MOSEI dataset...")
    train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(
        batch_size=args.batch_size, max_seq_len=args.max_seq_len, num_workers=args.num_workers
    )
    
    print_analysis_progress("Loading MOSI dataset...")
    mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(
        batch_size=args.batch_size, max_seq_len=args.max_seq_len, num_workers=args.num_workers
    )
    
    # Sample data from different sentiment bins
    print_analysis_progress("\nSampling data by sentiment labels...")
    
    mosei_samples = sample_by_sentiment_labels(
        test_loader, "MOSEI", args.samples_per_label, args.num_sentiment_bins
    )
    
    mosi_samples = sample_by_sentiment_labels(
        mosi_test, "MOSI", args.samples_per_label, args.num_sentiment_bins
    )
    
    sampled_datasets = {
        'MOSEI': mosei_samples,
        'MOSI': mosi_samples
    }
    
    # Create comprehensive analysis
    # all_analyses = create_comprehensive_analysis_report(
    #     sampled_datasets, args.checkpoint_path, MODEL_CONFIG, args.output_dir
    # )

    # print_analysis_progress("\nReplacing padding timesteps with -inf...")
    # sampled_datasets_with_inf, padding_stats = process_sampled_datasets_with_inf_padding(
    #     sampled_datasets, 
    #     padding_value=-float('inf'), 
    #     tolerance=1e-7, 
    #     verbose=True
    # )
    
    # # Use the modified datasets for analysis
    # all_analyses = create_comprehensive_analysis_report(
    #     sampled_datasets_with_inf, args.checkpoint_path, MODEL_CONFIG, args.output_dir
    # )
    
    # print_analysis_progress("Analysis pipeline completed successfully!")
    # print_analysis_progress(f"Check results in: {args.output_dir}")
    if not args.skip_analysis:
        print_analysis_progress("\nCreating comprehensive text analysis...")
        all_analyses = create_comprehensive_analysis_report(
            sampled_datasets, args.checkpoint_path, MODEL_CONFIG, args.output_dir
        )
    
    # Create visualizations
    if args.create_visualizations:
        print_analysis_progress("\nCreating visualizations...")
        create_codebook_sentiment_visualizations(
            sampled_datasets, args.checkpoint_path, MODEL_CONFIG, args.output_dir
        )
    
    print_analysis_progress("Analysis pipeline completed successfully!")
    print_analysis_progress(f"Check results in: {args.output_dir}")
    if args.create_visualizations:
        print_analysis_progress(f"Check visualizations in: {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()