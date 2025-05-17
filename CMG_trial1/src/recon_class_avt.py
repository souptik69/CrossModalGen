import logging
import os
import time
import random
import json
from tqdm import tqdm
import sys
import torch
from itertools import chain
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from configs.opts import parser
from model.main_model_2 import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from transformers import BertTokenizer, BertModel

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize BERT model and tokenizer for text processing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def generate_category_list():
    """Load category names from file"""
    file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/100kcategories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

def collate_func_AVT(samples):
    """Collate function for processing batches"""
    bsz = len(samples)
    
    # Get text prompts from samples
    text_prompts = [sample['text_fea'] for sample in samples]
    
    # Process using Transformers
    query = []
    query_words = []
    
    for text in text_prompts:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Get the last hidden state for each token
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Get token IDs and convert back to tokens
        token_ids = inputs.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Remove special tokens [CLS] and [SEP]
        non_special_tokens = tokens[1:-1]
        non_special_embeddings = embeddings[1:-1]
        
        # Filter tokens
        words = []
        words_emb = []
        
        for token, emb in zip(non_special_tokens, non_special_embeddings):
            # Get the token ID from the tokenizer
            idx = tokenizer.convert_tokens_to_ids(token)
            
            if idx != 0:
                words_emb.append(emb)
                words.append(idx)
        
        query.append(np.asarray(words_emb))
        query_words.append(words)

    query_len = []
    for i, sample in enumerate(query):
        query_len.append(10)  # max_num_words:10
    
    query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
    query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
    for i, sample in enumerate(query):
        keep = min(sample.shape[0], query1.shape[1])
        query1[i, :keep] = sample[:keep]
        query_idx[i, :keep] = query_words[i][:keep]
    
    query_len = np.asarray(query_len)
    query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
    query_idx = torch.from_numpy(query_idx).long()

    return {
        'query': query,
        'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float(),
        'avel_label': torch.from_numpy(np.asarray([sample['avel_label'] for sample in samples])).float()
    }

def create_recon_loss_plots(loss_df, class_names, modality, output_dir, k=30):
    """Create and save bar plots for best and worst reconstructed classes"""
    # Filter for classes with sufficient samples (e.g., more than 5)
    filtered_df = loss_df[loss_df['sample_count'] > 5].copy()
    
    # Sort by reconstruction loss
    df_sorted = filtered_df.sort_values(f'{modality.lower()}_recon_loss')
    
    # Plot best reconstructed classes (lowest loss)
    plt.figure(figsize=(14, 10))
    top_k = min(k, len(df_sorted))
    best_classes = df_sorted.head(top_k)
    sns.barplot(x=f'{modality.lower()}_recon_loss', y='category', data=best_classes, palette="YlGn")
    plt.title(f'Top {top_k} Best Reconstructed Classes - {modality} Modality')
    plt.xlabel('Reconstruction Loss (Lower is Better)')
    # Add sample count annotations
    for i, row in enumerate(best_classes.itertuples()):
        plt.text(0.01, i, f"n={int(row.sample_count)}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{modality.lower()}_best_recon_classes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot worst reconstructed classes (highest loss)
    plt.figure(figsize=(14, 10))
    worst_classes = df_sorted.tail(top_k).iloc[::-1]  # Reverse to show highest on top
    sns.barplot(x=f'{modality.lower()}_recon_loss', y='category', data=worst_classes, palette="YlOrRd_r")
    plt.title(f'Top {top_k} Worst Reconstructed Classes - {modality} Modality')
    plt.xlabel('Reconstruction Loss (Higher is Worse)')
    # Add sample count annotations
    for i, row in enumerate(worst_classes.itertuples()):
        plt.text(0.01, i, f"n={int(row.sample_count)}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{modality.lower()}_worst_recon_classes.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_compare_modalities_plot(loss_df, output_dir, num_classes=20):
    """Create plot comparing reconstruction loss across modalities for top classes"""
    # Calculate the average reconstruction loss across all modalities for each class
    loss_df['avg_recon_loss'] = (loss_df['audio_recon_loss'] + loss_df['video_recon_loss'] + loss_df['text_recon_loss']) / 3
    
    # Filter for classes with sufficient samples (e.g., more than 5)
    filtered_df = loss_df[loss_df['sample_count'] > 5].copy()
    
    # Sort by average reconstruction loss and get the top N classes
    top_classes = filtered_df.sort_values('avg_recon_loss').head(num_classes)
    
    # Melt the dataframe for easier plotting
    plot_df = pd.melt(
        top_classes,
        id_vars=['category', 'sample_count'],
        value_vars=['audio_recon_loss', 'video_recon_loss', 'text_recon_loss'],
        var_name='modality',
        value_name='recon_loss'
    )
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    sns.barplot(x='recon_loss', y='category', hue='modality', data=plot_df, palette='viridis')
    plt.title(f'Reconstruction Loss Comparison Across Modalities (Top {num_classes} Classes)')
    plt.xlabel('Reconstruction Loss (Lower is Better)')
    plt.legend(title='Modality')
    
    # Add sample count annotations
    unique_categories = plot_df['category'].unique()
    for i, category in enumerate(unique_categories):
        count = plot_df[plot_df['category'] == category]['sample_count'].iloc[0]
        plt.text(0.01, i*3, f"n={int(count)}", va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'modality_comparison_recon_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Get configurations
    global args, logger
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'statistics'), exist_ok=True)
    
    # Set up logger
    logger = Prepare_logger(args, eval=True)
    logger.info(f'\nOutput will be saved to {args.output_dir}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load category list
    category_list = generate_category_list()
    category_list.append("background")  # Add background class
    logger.info(f'Loaded {len(category_list)} categories (including background)')
    
    # Load dataset
    from dataset.vgg_test_dataset import VGGSoundDataset_AVT_test as AVEDataset
    
    meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv'
    audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/audio80k_features_new'
    video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/video80k_features_keras'
    avc_label_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/100klabels'
    
    logger.info('Loading test dataset...')
    test_dataset = AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffle for consistent results
        num_workers=8,
        pin_memory=False,
        collate_fn=collate_func_AVT
    )
    logger.info(f'Test dataset loaded with {len(test_dataset)} samples')
    
    # Initialize model
    logger.info('Initializing model...')
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    text_output_dim = 256
    audio_output_dim = 256
    n_embeddings = 400
    embedding_dim = 256
    num_classes = len(category_list)
    
    # Text LSTM for processing BERT embeddings
    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Text_ar_lstm.double().to(device)
    
    # Create encoder and decoder
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)
    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim)
    
    Encoder.double().to(device)
    Decoder.double().to(device)
    
    # Load pretrained model
    path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVT_100k_classloss_foreground/checkpoint/DCID-model-5.pt"
    logger.info(f'Loading pretrained model from {path_checkpoints}')
    
    checkpoint = torch.load(path_checkpoints)
    Encoder.load_state_dict(checkpoint['Encoder_parameters'])
    
    # If we have the Text_ar_lstm in the checkpoint
    if 'Text_ar_lstm_parameters' in checkpoint:
        Text_ar_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
    
    # If we have the Decoder in the checkpoint (optional)
    if 'Decoder_parameters' in checkpoint:
        Decoder.load_state_dict(checkpoint['Decoder_parameters'])
    
    # Set models to evaluation mode
    Encoder.eval()
    Text_ar_lstm.eval()
    Decoder.eval()
    
    # Initialize containers for reconstruction losses by class
    class_audio_recon_losses = {i: [] for i in range(num_classes)}
    class_video_recon_losses = {i: [] for i in range(num_classes)}
    class_text_recon_losses = {i: [] for i in range(num_classes)}
    class_sample_counts = {i: 0 for i in range(num_classes)}
    
    logger.info('Starting reconstruction loss analysis...')
    
    # Process all batches
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Analyzing Reconstruction Loss")):
            # Get data from batch
            query = batch_data['query'].double().to(device)
            audio_feature = batch_data['audio_fea'].double().to(device)
            video_feature = batch_data['video_fea'].double().to(device)
            labels = batch_data['avel_label'].to(device)
            
            # Extract labels
            labels = labels.double().cuda()
            # Get maximum class per frame, excluding background
            labels_BCE, labels_evn = labels.max(-1)  # Shape: [batch, 10]
            
            # For each sample in the batch, find the most common foreground class
            batch_foreground_classes = []
            for i in range(labels_evn.size(0)):
                frame_classes = labels_evn[i]
                bg_index = labels.size(2) - 1  # Background is the last class
                fg_classes = [idx.item() for idx in frame_classes if idx.item() != bg_index]
                
                if fg_classes:
                    from collections import Counter
                    most_common_fg = Counter(fg_classes).most_common(1)[0][0]
                    batch_foreground_classes.append(most_common_fg)
                else:
                    batch_foreground_classes.append(bg_index)
            
            labels_event = torch.tensor(batch_foreground_classes, device=labels.device)
            
            # Process text through LSTM
            batch_size = query.size()[0]
            hidden_dim = 128
            num_layers = 2
            text_hidden = (
                torch.zeros(2*num_layers, batch_size, hidden_dim).double().to(device),
                torch.zeros(2*num_layers, batch_size, hidden_dim).double().to(device)
            )
            text_feature, _ = Text_ar_lstm(query, text_hidden)
            
            # Forward pass through encoder
            audio_semantic_result, video_semantic_result, text_semantic_result, \
            audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
            audio_vq, video_vq, text_vq, _, _, _, _, _ = Encoder(audio_feature, video_feature, text_feature, 0)
            
            # Forward pass through decoder (for reconstruction)
            audio_recon_loss, video_recon_loss, text_recon_loss, _, _, _ = Decoder(
                audio_feature, video_feature, text_feature,
                audio_encoder_result, video_encoder_result, text_encoder_result,
                audio_vq, video_vq, text_vq
            )
            
            # Store reconstruction losses by class
            # for i in range(batch_size):
            #     class_idx = labels_event[i].item()
            #     class_sample_counts[class_idx] += 1
            for i in range(batch_size):
                class_idx = labels_event[i].item()
                
                # Check for background class (last index)
                if class_idx == len(category_list) - 1:
                    category_name = "background"
                else:
                    category_name = category_list[class_idx]
                
                class_sample_counts[class_idx] += 1
                
                # Use item() to get the scalar values from tensors
                class_audio_recon_losses[class_idx].append(audio_recon_loss.item())
                class_video_recon_losses[class_idx].append(video_recon_loss.item())
                class_text_recon_losses[class_idx].append(text_recon_loss.item())
    
    logger.info('Reconstruction loss analysis complete. Computing statistics...')
    
    # Calculate average reconstruction loss per class
    class_stats = []
    for i in range(num_classes):
        if class_sample_counts[i] > 0:
            if i == len(category_list) - 1:
                category_name = "background"
            else:
                category_name = category_list[i]

            avg_audio_recon = np.mean(class_audio_recon_losses[i]) if class_audio_recon_losses[i] else np.nan
            avg_video_recon = np.mean(class_video_recon_losses[i]) if class_video_recon_losses[i] else np.nan
            avg_text_recon = np.mean(class_text_recon_losses[i]) if class_text_recon_losses[i] else np.nan
            
            std_audio_recon = np.std(class_audio_recon_losses[i]) if len(class_audio_recon_losses[i]) > 1 else np.nan
            std_video_recon = np.std(class_video_recon_losses[i]) if len(class_video_recon_losses[i]) > 1 else np.nan
            std_text_recon = np.std(class_text_recon_losses[i]) if len(class_text_recon_losses[i]) > 1 else np.nan
            
            stat = {
                'category': category_list[i],
                'sample_count': class_sample_counts[i],
                'audio_recon_loss': avg_audio_recon,
                'video_recon_loss': avg_video_recon,
                'text_recon_loss': avg_text_recon,
                'audio_recon_std': std_audio_recon,
                'video_recon_std': std_video_recon,
                'text_recon_std': std_text_recon
            }
            class_stats.append(stat)
    
    # Create DataFrame for class statistics
    class_stats_df = pd.DataFrame(class_stats)
    
    # Save reconstruction loss statistics to CSV
    class_stats_df.to_csv(os.path.join(args.output_dir, 'statistics', 'class_reconstruction_stats.csv'), index=False)
    
    # Create visualizations
    logger.info('Generating reconstruction loss visualizations...')
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    
    # Create reconstruction loss plots for each modality
    create_recon_loss_plots(class_stats_df, category_list, 'Audio', vis_dir)
    create_recon_loss_plots(class_stats_df, category_list, 'Video', vis_dir)
    create_recon_loss_plots(class_stats_df, category_list, 'Text', vis_dir)
    
    # Create plot comparing modalities
    create_compare_modalities_plot(class_stats_df, vis_dir)
    
    # Create histogram of reconstruction loss distribution for each modality
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(class_stats_df['audio_recon_loss'].dropna(), kde=True)
    plt.title('Audio Reconstruction Loss Distribution')
    plt.xlabel('Loss')
    
    plt.subplot(1, 3, 2)
    sns.histplot(class_stats_df['video_recon_loss'].dropna(), kde=True)
    plt.title('Video Reconstruction Loss Distribution')
    plt.xlabel('Loss')
    
    plt.subplot(1, 3, 3)
    sns.histplot(class_stats_df['text_recon_loss'].dropna(), kde=True)
    plt.title('Text Reconstruction Loss Distribution')
    plt.xlabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'recon_loss_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of audio vs video reconstruction loss
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        class_stats_df['audio_recon_loss'], 
        class_stats_df['video_recon_loss'],
        c=class_stats_df['sample_count'],
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add colorbar for sample count
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sample Count')
    
    # Add labels for points with extreme values
    threshold_high = class_stats_df['audio_recon_loss'].quantile(0.9) 
    threshold_low = class_stats_df['audio_recon_loss'].quantile(0.1)
    
    for _, row in class_stats_df.iterrows():
        if (row['audio_recon_loss'] > threshold_high or 
            row['audio_recon_loss'] < threshold_low or
            row['video_recon_loss'] > class_stats_df['video_recon_loss'].quantile(0.9) or
            row['video_recon_loss'] < class_stats_df['video_recon_loss'].quantile(0.1)):
            plt.annotate(
                row['category'],
                (row['audio_recon_loss'], row['video_recon_loss']),
                fontsize=8
            )
    
    plt.title('Audio vs Video Reconstruction Loss by Class')
    plt.xlabel('Audio Reconstruction Loss')
    plt.ylabel('Video Reconstruction Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'audio_vs_video_recon_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate correlation between modality reconstruction losses
    audio_video_corr = class_stats_df['audio_recon_loss'].corr(class_stats_df['video_recon_loss'])
    audio_text_corr = class_stats_df['audio_recon_loss'].corr(class_stats_df['text_recon_loss'])
    video_text_corr = class_stats_df['video_recon_loss'].corr(class_stats_df['text_recon_loss'])
    
    # Create correlation matrix for visualization
    corr_matrix = np.array([
        [1.0, audio_video_corr, audio_text_corr],
        [audio_video_corr, 1.0, video_text_corr],
        [audio_text_corr, video_text_corr, 1.0]
    ])
    
    # Plot correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=['Audio', 'Video', 'Text'],
                yticklabels=['Audio', 'Video', 'Text'])
    plt.title('Correlation Between Modality Reconstruction Losses')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'recon_loss_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary of reconstruction loss results
    summary = {
        'num_classes_analyzed': len(class_stats_df),
        'total_samples': int(class_stats_df['sample_count'].sum()),
        
        'avg_audio_recon_loss': float(class_stats_df['audio_recon_loss'].mean()),
        'std_audio_recon_loss': float(class_stats_df['audio_recon_loss'].std()),
        'min_audio_recon_loss': float(class_stats_df['audio_recon_loss'].min()),
        'max_audio_recon_loss': float(class_stats_df['audio_recon_loss'].max()),
        'best_audio_recon_class': class_stats_df.loc[class_stats_df['audio_recon_loss'].idxmin()]['category'],
        'worst_audio_recon_class': class_stats_df.loc[class_stats_df['audio_recon_loss'].idxmax()]['category'],
        
        'avg_video_recon_loss': float(class_stats_df['video_recon_loss'].mean()),
        'std_video_recon_loss': float(class_stats_df['video_recon_loss'].std()),
        'min_video_recon_loss': float(class_stats_df['video_recon_loss'].min()),
        'max_video_recon_loss': float(class_stats_df['video_recon_loss'].max()),
        'best_video_recon_class': class_stats_df.loc[class_stats_df['video_recon_loss'].idxmin()]['category'],
        'worst_video_recon_class': class_stats_df.loc[class_stats_df['video_recon_loss'].idxmax()]['category'],
        
        'avg_text_recon_loss': float(class_stats_df['text_recon_loss'].mean()),
        'std_text_recon_loss': float(class_stats_df['text_recon_loss'].std()),
        'min_text_recon_loss': float(class_stats_df['text_recon_loss'].min()),
        'max_text_recon_loss': float(class_stats_df['text_recon_loss'].max()),
        'best_text_recon_class': class_stats_df.loc[class_stats_df['text_recon_loss'].idxmin()]['category'],
        'worst_text_recon_class': class_stats_df.loc[class_stats_df['text_recon_loss'].idxmax()]['category'],
        
        'audio_video_corr': float(audio_video_corr),
        'audio_text_corr': float(audio_text_corr),
        'video_text_corr': float(video_text_corr)
    }
    
    with open(os.path.join(args.output_dir, 'recon_loss_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    logger.info('Reconstruction loss analysis complete! Results saved to {}'.format(args.output_dir))
    logger.info(f'Summary:')
    logger.info(f'Average Audio Reconstruction Loss: {summary["avg_audio_recon_loss"]:.6f}')
    logger.info(f'Average Video Reconstruction Loss: {summary["avg_video_recon_loss"]:.6f}')
    logger.info(f'Average Text Reconstruction Loss: {summary["avg_text_recon_loss"]:.6f}')
    logger.info(f'Best Audio Reconstruction Class: {summary["best_audio_recon_class"]} (Loss: {summary["min_audio_recon_loss"]:.6f})')
    logger.info(f'Best Video Reconstruction Class: {summary["best_video_recon_class"]} (Loss: {summary["min_video_recon_loss"]:.6f})')
    logger.info(f'Best Text Reconstruction Class: {summary["best_text_recon_class"]} (Loss: {summary["min_text_recon_loss"]:.6f})')
    logger.info(f'Worst Audio Reconstruction Class: {summary["worst_audio_recon_class"]} (Loss: {summary["max_audio_recon_loss"]:.6f})')
    logger.info(f'Worst Video Reconstruction Class: {summary["worst_video_recon_class"]} (Loss: {summary["max_video_recon_loss"]:.6f})')
    logger.info(f'Worst Text Reconstruction Class: {summary["worst_text_recon_class"]} (Loss: {summary["max_text_recon_loss"]:.6f})')

if __name__ == '__main__':
    main()