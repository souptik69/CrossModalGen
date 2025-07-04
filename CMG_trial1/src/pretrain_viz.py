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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser
from model.main_model_novel import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
from model.CPC import Cross_CPC, Cross_CPC_AVT
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pickle
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
torch.autograd.set_detect_anomaly(True)

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def load_checkpoint_and_find_top_vectors(checkpoint_path, top_k=25):
    """
    Load checkpoint and find the top k most used vectors to track during training
    """
    # Model configuration (matching setup)
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    n_embeddings = 400
    embedding_dim = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim, n_embeddings, embedding_dim)
    Encoder.double()
    Encoder.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoints = torch.load(checkpoint_path, map_location=device)
    Encoder.load_state_dict(checkpoints['Encoder_parameters'])
    
    # Extract quantizer and get top vectors
    quantizer = Encoder.Cross_quantizer
    ema_counts = quantizer.ema_count.detach().cpu().numpy()
    
    # Find top-k most used vectors
    top_indices = np.argsort(ema_counts)[-top_k:][::-1]  # Descending order
    top_counts = ema_counts[top_indices]
    
    print(f"Top {top_k} most used vectors from checkpoint:")
    for i, (idx, count) in enumerate(zip(top_indices, top_counts)):
        print(f"  Rank {i+1}: Vector {idx}, Usage: {count:.2f}")
    
    return top_indices

def create_combined_modality_plot(quantizer, top_indices, output_dir, iteration, embedding_dim=256, top_k=25):
    """
    Create the combined modality plot for the specified top vectors
    This shows how video, audio, and text segments of the same vectors cluster together
    """
    # Extract embeddings for top vectors
    embeddings = quantizer.embedding.detach().cpu().numpy()
    ema_counts = quantizer.ema_count.detach().cpu().numpy()
    top_indices_1 = np.argsort(ema_counts)[-top_k:][::-1]
    top_embeddings = embeddings[top_indices_1]

    # top_embeddings = embeddings[top_indices]

    
    # Extract modality segments from the 768-dimensional embeddings
    # Each vector is split into: video (0:256), audio (256:512), text (512:768)
    video_segments = top_embeddings[:, :embedding_dim]
    audio_segments = top_embeddings[:, embedding_dim:2*embedding_dim]
    text_segments = top_embeddings[:, 2*embedding_dim:]
    
    # Apply PCA to each modality to reduce to 2D for visualization
    pca_video = PCA(n_components=2)
    pca_audio = PCA(n_components=2)
    pca_text = PCA(n_components=2)
    
    video_2d = pca_video.fit_transform(video_segments)
    audio_2d = pca_audio.fit_transform(audio_segments)
    text_2d = pca_text.fit_transform(text_segments)
    
    # Create the combined plot
    plt.figure(figsize=(12, 10))
    
    # Plot all modalities together with different colors and markers
    scatter1 = plt.scatter(video_2d[:, 0], video_2d[:, 1], 
                          c='red', s=100, marker='o', 
                          alpha=0.7, label='Video', edgecolors='black', linewidth=0.5)
    scatter2 = plt.scatter(audio_2d[:, 0], audio_2d[:, 1], 
                          c='blue', s=100, marker='s', 
                          alpha=0.7, label='Audio', edgecolors='black', linewidth=0.5)
    scatter3 = plt.scatter(text_2d[:, 0], text_2d[:, 1], 
                          c='green', s=100, marker='^', 
                          alpha=0.7, label='Text', edgecolors='black', linewidth=0.5)
    
    # Add vector indices as labels for each point
    for i, (x, y) in enumerate(video_2d):
        plt.annotate(f'{top_indices_1[i]}', (x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8, color='darkred', fontweight='bold')
    
    for i, (x, y) in enumerate(audio_2d):
        plt.annotate(f'{top_indices_1[i]}', (x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8, color='darkblue', fontweight='bold')
    
    for i, (x, y) in enumerate(text_2d):
        plt.annotate(f'{top_indices_1[i]}', (x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8, color='darkgreen', fontweight='bold')
    
    plt.title(f'All Modalities Combined - Iteration {iteration}', fontsize=14, fontweight='bold')
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'combined_modalities_iter_{iteration}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {output_path}")
    return output_path

def collate_func_AVT(samples):
    bsz = len(samples)
    text_prompts = [sample['text_fea'] for sample in samples]
    query = []
    query_words = []
    
    for text in text_prompts:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
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
        'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
    }

def main():
    # Parse arguments
    parser_local = argparse.ArgumentParser(description='Pretraining with Codebook Visualization')
    parser_local.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser_local.add_argument('--lr', type=float, default=0.0004, help='Learning rate')
    parser_local.add_argument('--clip_gradient', type=float, default=0.5, help='Gradient clipping')
    parser_local.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser_local.add_argument('--dataset_name', type=str, default='vggsound', help='Dataset name')
    parser_local.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint for finding top vectors')
    parser_local.add_argument('--output_dir', type=str, required=True, help='Output directory for visualizations')
    parser_local.add_argument('--top_k', type=int, default=25, help='Number of top vectors to track')
    parser_local.add_argument('--print_freq', type=int, default=1, help='Print frequency')
    
    args = parser_local.parse_args()
    
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load checkpoint and find top vectors to track
    print("="*50)
    print("STEP 1: Loading checkpoint and finding top vectors to track")
    print("="*50)
    top_indices = load_checkpoint_and_find_top_vectors(args.checkpoint_path, args.top_k)
    
    # Dataset setup
    print("\nSTEP 2: Setting up dataset")
    print("="*50)
    if args.dataset_name == 'vggsound':
        from dataset.VGG_dataset_novel import VGGSoundDataset_AVT as AVEDataset
    else:
        raise NotImplementedError 

    # Data paths for 40k dataset
    meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv'
    audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip'
    video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip'
    
    train_dataloader = DataLoader(
        AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        collate_fn=collate_func_AVT
    )
    
    # Model setup
    print("\nSTEP 3: Setting up models")
    print("="*50)
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    n_embeddings = 400
    embedding_dim = 256
    
    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim, n_embeddings, embedding_dim)
    CPC = Cross_CPC_AVT(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim)

    Text_ar_lstm.double()
    Encoder.double()
    CPC.double()
    Decoder.double()

    Text_ar_lstm.to(device)
    Encoder.to(device)
    CPC.to(device)
    Decoder.to(device)
    
    # Optimizer setup
    optimizer = torch.optim.Adam(chain(Text_ar_lstm.parameters(), 
                                       Encoder.parameters(), 
                                       CPC.parameters(), 
                                       Decoder.parameters()), lr=args.lr)
    
    # Loss functions
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()
    
    # Training for 1 epoch with visualizations
    print("\nSTEP 4: Starting training with visualizations")
    print("="*50)
    print(f"Will create visualizations at iterations: 0, 20, 40, 80, 120, 160, 320, 620")
    # print(f"Tracking top {args.top_k} vectors: {top_indices}")
    
    train_with_visualization(CPC, Encoder, Text_ar_lstm, Decoder, train_dataloader, 
                           criterion, criterion_event, optimizer, top_indices, args.output_dir, logger, args.top_k)
    
    print("\nVisualization-based pretraining complete!")

def train_with_visualization(CPC, Encoder, Text_ar_lstm, Decoder, train_dataloader, 
                           criterion, criterion_event, optimizer, top_indices, output_dir, logger, top_k=25):
    """
    Train for one epoch while creating visualizations at specific iterations
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()
    
    # Set models to training mode
    models = [CPC, Encoder, Text_ar_lstm, Decoder]
    for m in models:
        m.train()
    
    # Iterations at which to create visualizations
    viz_iterations = [0, 20, 40, 80, 120, 160, 320, 620]
    
    for n_iter, batch_data in enumerate(train_dataloader):
        
        # Create visualization at specified iterations
        if n_iter in viz_iterations:
            print(f"\n{'='*60}")
            print(f"Creating visualization at iteration {n_iter}")
            print(f"{'='*60}")
            
            # Set models to eval mode for visualization
            for m in models:
                m.eval()
            
            with torch.no_grad():
                create_combined_modality_plot(Encoder.Cross_quantizer, top_indices, 
                                            output_dir, n_iter, top_k)
            
            # Set models back to training mode
            for m in models:
                m.train()
        
        data_time.update(time.time() - end_time)
        
        # Forward pass
        query, audio_feature, video_feature = batch_data['query'], batch_data['audio_fea'], batch_data['video_fea']
        query = query.double().cuda()

        batch_dim = query.size()[0]
        hidden_dim = 128
        num_layers = 2
        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                      torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(query, text_hidden)

        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        # Encoder forward pass
        audio_semantic_result, audio_encoder_result, video_semantic_result, video_spatial, \
        text_semantic_result, text_encoder_result, \
        out_vq_video, video_vq, out_vq_audio, audio_vq,\
        out_vq_text, text_vq, video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, equal_num, cmcm_loss, segment_loss \
        = Encoder(audio_feature, video_feature, text_feature, 0)  # epoch=0 for single epoch

        # CPC forward pass
        accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, \
        cpc_loss = CPC(audio_semantic_result, video_semantic_result, text_semantic_result)

        # Decoder forward pass
        audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class \
            = Decoder(audio_feature, video_feature, text_feature, audio_encoder_result, video_spatial, text_encoder_result, 
                     out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq)

        # Compute total loss
        loss = audio_recon_loss + video_recon_loss + text_recon_loss + audio_embedding_loss + \
               video_embedding_loss + text_embedding_loss + cpc_loss + cmcm_loss

        # Backward pass
        loss.backward()
        
        # Gradient clipping
        for model in models:
            clip_grad_norm_(model.parameters(), 0.5)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update meters
        losses.update(loss.item(), audio_feature.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # Logging
        if n_iter % 20 == 0:
            msg = f'Iteration {n_iter}, Loss: {loss.item():.4f}, ' \
                  f'Audio Recon: {audio_recon_loss.item():.4f}, ' \
                  f'Video Recon: {video_recon_loss.item():.4f}, ' \
                  f'Text Recon: {text_recon_loss.item():.4f}'
            logger.info(msg)
        
        # Stop after reasonable number of iterations for visualization purposes
        if n_iter >= 650:
            print(f"Stopping training after {n_iter} iterations")
            break

    return losses.avg

if __name__ == "__main__":
    main()