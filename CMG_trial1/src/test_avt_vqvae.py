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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
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

def create_confusion_matrix_plot(cm, class_names, modality, output_dir, k=20):
    """Create and save a confusion matrix visualization for top-k confused classes"""
    # Find the most confused classes
    np.fill_diagonal(cm, 0)  # Ignore correct classifications to find confusion
    
    # Get sum of misclassifications for each class (row-wise)
    class_error_totals = np.sum(cm, axis=1)
    
    # Get top k classes with highest misclassification
    top_confused_indices = np.argsort(class_error_totals)[-k:]
    
    # Extract the relevant subset of the confusion matrix
    cm_subset = cm[top_confused_indices][:, top_confused_indices]
    names_subset = [class_names[i] for i in top_confused_indices]
    
    # Create the figure
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_subset, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=names_subset, yticklabels=names_subset)
    plt.title(f'Top {k} Most Confused Classes - {modality} Modality')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{modality}_top{k}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_per_class_accuracy_plot(accuracies, class_names, modality, output_dir, k=30):
    """Create and save bar plots for top and bottom k classes by accuracy"""
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'class': class_names, 'accuracy': accuracies})
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy')
    
    # Plot bottom k classes (worst performing)
    plt.figure(figsize=(14, 10))
    bottom_k = df_sorted.head(k)
    sns.barplot(x='accuracy', y='class', data=bottom_k, palette="YlOrRd_r")
    plt.title(f'Bottom {k} Classes by Accuracy - {modality} Modality')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{modality}_bottom{k}_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top k classes (best performing)
    plt.figure(figsize=(14, 10))
    top_k = df_sorted.tail(k).iloc[::-1]  # Reverse to show highest on top
    sns.barplot(x='accuracy', y='class', data=top_k, palette="YlGn")
    plt.title(f'Top {k} Classes by Accuracy - {modality} Modality')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{modality}_top{k}_accuracies.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_most_misclassified_pairs_plot(cm, class_names, modality, output_dir, k=20):
    """Identify and visualize the most common misclassification pairs"""
    # Copy the confusion matrix to avoid modifying the original
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Remove correct classifications
    
    # Flatten and find indices of top k confused pairs
    flat_indices = np.argsort(cm_copy.flatten())[-k:]
    row_indices, col_indices = np.unravel_index(flat_indices, cm_copy.shape)
    
    # Create DataFrame for the pairs
    pairs = []
    for i in range(k):
        true_class = class_names[row_indices[i]]
        pred_class = class_names[col_indices[i]]
        count = cm_copy[row_indices[i], col_indices[i]]
        pairs.append({
            'True Class': true_class,
            'Predicted Class': pred_class,
            'Count': count
        })
    
    df_pairs = pd.DataFrame(pairs)
    
    # Plot
    plt.figure(figsize=(16, 10))
    sns.barplot(x='Count', y='True Class', hue='Predicted Class', data=df_pairs)
    plt.title(f'Top {k} Misclassification Pairs - {modality} Modality')
    plt.xlabel('Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{modality}_top{k}_misclassification_pairs.png'), dpi=300, bbox_inches='tight')
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
    category_list.append("background")
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
    # logger.info(f'Loading pretrained model from {args.model_path}')
    
    # path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVT_100k_Test/checkpoint/DCID-model-5.pt"
    # path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVT_100k_classloss_final/checkpoint/DCID-model-5.pt"

    path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVT_100k_classloss_foreground/checkpoint/DCID-model-5.pt"

    checkpoint = torch.load(path_checkpoints)
    Encoder.load_state_dict(checkpoint['Encoder_parameters'])
    # start_epoch = checkpoint['epoch']
    # logger.info("Resume from number {}-th model.".format(start_epoch))
    
    # If we have the Text_ar_lstm in the checkpoint
    if 'Text_ar_lstm_parameters' in checkpoint:
        Text_ar_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
    
    # If we have the Decoder in the checkpoint (optional)
    if 'Decoder_parameters' in checkpoint:
        Decoder.load_state_dict(checkpoint['Decoder_parameters'])
    
    # Loss function for classification evaluation
    # criterion_event = nn.CrossEntropyLoss().cuda()
    criterion_event = nn.CrossEntropyLoss(reduction='none').cuda()
    
    # Set models to evaluation mode
    Encoder.eval()
    Text_ar_lstm.eval()
    Decoder.eval()
    
    # Initialize containers for metrics
    all_video_preds = []
    all_audio_preds = []
    all_text_preds = []
    all_true_labels = []
    
    # Per-class metrics containers
    # num_classes = (len(category_list) + 1)
    class_correct_audio = np.zeros(num_classes)
    class_total_audio = np.zeros(num_classes)
    class_correct_video = np.zeros(num_classes)
    class_total_video = np.zeros(num_classes)
    
    # Per-sample loss tracking
    audio_losses = []
    video_losses = []
    sample_categories = []
    
    logger.info('Starting evaluation...')
    
    # Process all batches
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Get data from batch
            query = batch_data['query'].double().to(device)
            audio_feature = batch_data['audio_fea'].double().to(device)
            video_feature = batch_data['video_fea'].double().to(device)
            labels = batch_data['avel_label'].to(device)
            
            # Extract labels
            # avel_label shape: [batch_size, 10, num_classes+1]
            # labels_foreground = avel_label[:, :, :-1]  # Remove background class
            
            # Get the max across classes (excluding background)
            # This gives us the event class index for each time step
            # labels_event shape: [batch_size, 10]
            # labels_event = torch.argmax(labels_foreground, dim=2)

            # labels = labels.double().cuda()
            # # labels_foreground = labels[:, :, :-1]  
            # # labels_BCE, labels_evn = labels_foreground.max(-1)
            # # labels_event, _ = labels_evn.max(-1)
            # labels_BCE, labels_evn = labels.max(-1)
            # labels_event = torch.mode(labels_evn, dim=1).values


            labels = labels.double().cuda()
            labels_BCE, labels_evn = labels.max(-1)  # Shape: [batch, 10]
            batch_foreground_classes = []
            for i in range(labels_evn.size(0)):  # For each sample in the batch
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
                


            # Get the class for each sample (max across time)
            # sample_classes shape: [batch_size]
            # sample_classes = torch.mode(labels_event, dim=1).values
            # sample_classes = labels_event
            
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
            
            # Forward pass through decoder (for classification)
            _, _, _, audio_class, video_class, text_class = Decoder(
                audio_feature, video_feature, text_feature,
                audio_encoder_result, video_encoder_result, text_encoder_result,
                audio_vq, video_vq, text_vq
            )
            
            # Get predictions
            audio_preds = torch.argmax(audio_class, dim=1)  # [batch, time]
            video_preds = torch.argmax(video_class, dim=1)  # [batch, time]
            text_preds = torch.argmax(text_class, dim=1)    # [batch, time]
            

            # # Calculate per-class statistics
            # for i in range(batch_size):
            #     true_class = sample_classes[i].item()
            #     audio_pred = audio_sample_preds[i].item()
            #     video_pred = video_sample_preds[i].item()
                
            #     # Update counts
            #     class_total_audio[true_class] += 1
            #     class_total_video[true_class] += 1
                
            #     if audio_pred == true_class:
            #         class_correct_audio[true_class] += 1
                
            #     if video_pred == true_class:
            #         class_correct_video[true_class] += 1

            # Calculate per-class statistics
            for i in range(batch_size):
                true_class = labels_event[i].item()
                audio_pred = audio_preds[i].item()
                video_pred = video_preds[i].item()
                
                # Update counts
                class_total_audio[true_class] += 1
                class_total_video[true_class] += 1
                
                if audio_pred == true_class:
                    class_correct_audio[true_class] += 1
                
                if video_pred == true_class:
                    class_correct_video[true_class] += 1
            
            # Calculate per-sample losses
            # for t in range(10):  # For each time step
                # Get losses for this time step
                # audio_loss = criterion_event(audio_class[:, t], labels_event[:, t])
                # video_loss = criterion_event(video_class[:, t], labels_event[:, t])

            audio_loss = criterion_event(audio_class, labels_event)
            video_loss = criterion_event(video_class, labels_event)
                
                # Store losses with corresponding true labels
                # for i in range(batch_size):
                #     audio_losses.append((labels_event[i, t].item(), audio_loss[i].item()))
                #     video_losses.append((labels_event[i, t].item(), video_loss[i].item()))
                #     sample_categories.append(category_list[labels_event[i, t].item()])
                # Store losses with corresponding true labels
            for i in range(batch_size):
                class_idx = labels_event[i].item()
                # For the background class (last index), use the label "background"
                if class_idx == len(category_list) -1:
                    category_name = "background"
                else:
                    category_name = category_list[class_idx]

                # audio_losses.append((labels_event[i].item(), audio_loss[i].item()))
                # video_losses.append((labels_event[i].item(), video_loss[i].item()))
                # # sample_categories.append(category_list[labels_event[i].item()])
                # sample_categories.append(category_name)

                audio_losses.append((class_idx, audio_loss[i].item()))
                video_losses.append((class_idx, video_loss[i].item()))
                sample_categories.append(category_name)
        
            # Store predictions and true labels for overall metrics
            # all_video_preds.extend(video_sample_preds.cpu().numpy())
            # all_audio_preds.extend(audio_sample_preds.cpu().numpy())
            # all_text_preds.extend(text_sample_preds.cpu().numpy())
            # all_true_labels.extend(sample_classes.cpu().numpy())

            all_video_preds.extend(video_preds.cpu().numpy())
            all_audio_preds.extend(audio_preds.cpu().numpy())
            all_text_preds.extend(text_preds.cpu().numpy())
            all_true_labels.extend(labels_event.cpu().numpy())
    
    
    logger.info('Evaluation complete. Computing metrics...')
    
    # Calculate overall accuracy
    audio_accuracy = accuracy_score(all_true_labels, all_audio_preds)
    video_accuracy = accuracy_score(all_true_labels, all_video_preds)
    text_accuracy = accuracy_score(all_true_labels, all_text_preds)
    
    logger.info(f'Overall Audio Accuracy: {audio_accuracy:.4f}')
    logger.info(f'Overall Video Accuracy: {video_accuracy:.4f}')
    logger.info(f'Overall Text Accuracy: {text_accuracy:.4f}')
    
    # Calculate per-class accuracies
    class_accuracy_audio = np.zeros(num_classes)
    class_accuracy_video = np.zeros(num_classes)
    
    for i in range(num_classes):
        if class_total_audio[i] > 0:
            class_accuracy_audio[i] = class_correct_audio[i] / class_total_audio[i]
        
        if class_total_video[i] > 0:
            class_accuracy_video[i] = class_correct_video[i] / class_total_video[i]
    
    # Create confusion matrices
    audio_cm = confusion_matrix(all_true_labels, all_audio_preds, labels=range(num_classes))
    video_cm = confusion_matrix(all_true_labels, all_video_preds, labels=range(num_classes))
    
    # Generate detailed classification reports
    # category_list.append("background")
    audio_report = classification_report(all_true_labels, all_audio_preds, target_names=category_list, output_dict=True)
    video_report = classification_report(all_true_labels, all_video_preds, target_names=category_list, output_dict=True)
    
    # Convert reports to DataFrames
    audio_df = pd.DataFrame(audio_report).transpose()
    video_df = pd.DataFrame(video_report).transpose()
    
    # Save classification reports
    audio_df.to_csv(os.path.join(args.output_dir, 'statistics', 'audio_classification_report.csv'))
    video_df.to_csv(os.path.join(args.output_dir, 'statistics', 'video_classification_report.csv'))
    
    # Create and save per-class statistics
    class_stats = []
    for i in range(num_classes):
        stat = {
            'category': category_list[i],
            'total_samples': int(class_total_audio[i]),
            'audio_correct': int(class_correct_audio[i]),
            'audio_accuracy': class_accuracy_audio[i],
            'video_correct': int(class_correct_video[i]),
            'video_accuracy': class_accuracy_video[i]
        }
        class_stats.append(stat)
    
    class_stats_df = pd.DataFrame(class_stats)
    class_stats_df.to_csv(os.path.join(args.output_dir, 'statistics', 'class_statistics.csv'), index=False)
    
    # Save per-sample losses
    audio_loss_df = pd.DataFrame(audio_losses, columns=['class_idx', 'loss'])
    audio_loss_df['category'] = audio_loss_df['class_idx'].apply(lambda x: category_list[x])
    audio_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'audio_sample_losses.csv'), index=False)
    
    video_loss_df = pd.DataFrame(video_losses, columns=['class_idx', 'loss'])
    video_loss_df['category'] = video_loss_df['class_idx'].apply(lambda x: category_list[x])
    video_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'video_sample_losses.csv'), index=False)
    
    # Calculate average loss per class
    audio_class_avg_loss = audio_loss_df.groupby('class_idx')['loss'].mean()
    video_class_avg_loss = video_loss_df.groupby('class_idx')['loss'].mean()
    
    avg_loss_df = pd.DataFrame({
        'category': category_list,
        'audio_avg_loss': audio_class_avg_loss.values,
        'video_avg_loss': video_class_avg_loss.values
    })
    avg_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'average_class_loss.csv'), index=False)
    
    # Create visualizations
    logger.info('Generating visualizations...')
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    
    # Create confusion matrix visualizations
    create_confusion_matrix_plot(audio_cm, category_list, 'Audio', vis_dir)
    create_confusion_matrix_plot(video_cm, category_list, 'Video', vis_dir)
    
    # Create per-class accuracy plots
    create_per_class_accuracy_plot(class_accuracy_audio, category_list, 'Audio', vis_dir)
    create_per_class_accuracy_plot(class_accuracy_video, category_list, 'Video', vis_dir)
    
    # Create misclassification pair plots
    create_most_misclassified_pairs_plot(audio_cm, category_list, 'Audio', vis_dir)
    create_most_misclassified_pairs_plot(video_cm, category_list, 'Video', vis_dir)
    
    # Find the most ambiguous categories (highest confusion)
    audio_confusion_sum = np.sum(audio_cm, axis=1) - np.diag(audio_cm)
    video_confusion_sum = np.sum(video_cm, axis=1) - np.diag(video_cm)
    
    # Normalize by number of samples
    audio_confusion_rate = audio_confusion_sum / np.sum(audio_cm, axis=1)
    video_confusion_rate = video_confusion_sum / np.sum(video_cm, axis=1)
    
    # Create DataFrame for ambiguous classes
    ambiguous_df = pd.DataFrame({
        'category': category_list,
        'audio_confusion_rate': audio_confusion_rate,
        'video_confusion_rate': video_confusion_rate,
        'sample_count': np.sum(audio_cm, axis=1)
    })
    
    # Save to CSV
    ambiguous_df.sort_values('audio_confusion_rate', ascending=False).to_csv(
        os.path.join(args.output_dir, 'statistics', 'audio_ambiguous_categories.csv'), index=False)
    
    ambiguous_df.sort_values('video_confusion_rate', ascending=False).to_csv(
        os.path.join(args.output_dir, 'statistics', 'video_ambiguous_categories.csv'), index=False)
    
    # Find the most confused pairs for each category
    most_confused_pairs = []
    for i in range(num_classes):
        # Skip diagonal (correct classification)
        confusion_row = audio_cm[i].copy()
        confusion_row[i] = 0
        
        # Find top confused class
        if np.sum(confusion_row) > 0:
            most_confused_idx = np.argmax(confusion_row)
            most_confused_pairs.append({
                'category': category_list[i],
                'most_confused_with': category_list[most_confused_idx],
                'confusion_count': confusion_row[most_confused_idx],
                'total_samples': np.sum(audio_cm[i]),
                'confusion_rate': confusion_row[most_confused_idx] / np.sum(audio_cm[i])
            })
    
    confused_pairs_df = pd.DataFrame(most_confused_pairs)
    confused_pairs_df.sort_values('confusion_rate', ascending=False).to_csv(
        os.path.join(args.output_dir, 'statistics', 'audio_most_confused_pairs.csv'), index=False)
    
    # Same for video
    most_confused_pairs = []
    for i in range(num_classes):
        confusion_row = video_cm[i].copy()
        confusion_row[i] = 0
        
        if np.sum(confusion_row) > 0:
            most_confused_idx = np.argmax(confusion_row)
            most_confused_pairs.append({
                'category': category_list[i],
                'most_confused_with': category_list[most_confused_idx],
                'confusion_count': confusion_row[most_confused_idx],
                'total_samples': np.sum(video_cm[i]),
                'confusion_rate': confusion_row[most_confused_idx] / np.sum(video_cm[i])
            })
    
    confused_pairs_df = pd.DataFrame(most_confused_pairs)
    confused_pairs_df.sort_values('confusion_rate', ascending=False).to_csv(
        os.path.join(args.output_dir, 'statistics', 'video_most_confused_pairs.csv'), index=False)
    
    
        # Calculate background class statistics
    bg_class_idx = len(category_list) - 1  # Background is the last class
    bg_true_count = np.sum(np.array(all_true_labels) == bg_class_idx)
    bg_audio_pred_count = np.sum(np.array(all_audio_preds) == bg_class_idx)
    bg_video_pred_count = np.sum(np.array(all_video_preds) == bg_class_idx)
    
    # Calculate percentage of samples predicted as background
    total_samples = len(all_true_labels)
    bg_true_percent = (bg_true_count / total_samples) * 100 if total_samples > 0 else 0
    bg_audio_pred_percent = (bg_audio_pred_count / total_samples) * 100 if total_samples > 0 else 0
    bg_video_pred_percent = (bg_video_pred_count / total_samples) * 100 if total_samples > 0 else 0
    
    # Log background class statistics
    logger.info(f"\nBackground Class Statistics:")
    logger.info(f"True background samples: {bg_true_count} ({bg_true_percent:.2f}% of total)")
    logger.info(f"Audio predictions as background: {bg_audio_pred_count} ({bg_audio_pred_percent:.2f}% of total)")
    logger.info(f"Video predictions as background: {bg_video_pred_count} ({bg_video_pred_percent:.2f}% of total)")
    
    # Create summary of results
    summary = {
        'audio_accuracy': audio_accuracy,
        'video_accuracy': video_accuracy,
        'text_accuracy': text_accuracy,
        'total_samples': len(all_true_labels),
        'total_classes': num_classes,
        'background_true_count': int(bg_true_count),
        'background_true_percent': bg_true_percent,
        'background_audio_pred_count': int(bg_audio_pred_count),
        'background_audio_pred_percent': bg_audio_pred_percent,
        'background_video_pred_count': int(bg_video_pred_count),
        'background_video_pred_percent': bg_video_pred_percent,
        'best_audio_class': category_list[np.argmax(class_accuracy_audio)],
        'best_audio_accuracy': float(np.max(class_accuracy_audio)),
        'worst_audio_class': category_list[np.argmin(class_accuracy_audio)],
        'worst_audio_accuracy': float(np.min(class_accuracy_audio)),
        'best_video_class': category_list[np.argmax(class_accuracy_video)],
        'best_video_accuracy': float(np.max(class_accuracy_video)),
        'worst_video_class': category_list[np.argmin(class_accuracy_video)],
        'worst_video_accuracy': float(np.min(class_accuracy_video))
    }
    



    # # Create summary of results
    # summary = {
    #     'audio_accuracy': audio_accuracy,
    #     'video_accuracy': video_accuracy,
    #     'text_accuracy': text_accuracy,
    #     'total_samples': len(all_true_labels),
    #     'total_classes': num_classes,
    #     'best_audio_class': category_list[np.argmax(class_accuracy_audio)],
    #     'best_audio_accuracy': np.max(class_accuracy_audio),
    #     'worst_audio_class': category_list[np.argmin(class_accuracy_audio)],
    #     'worst_audio_accuracy': np.min(class_accuracy_audio),
    #     'best_video_class': category_list[np.argmax(class_accuracy_video)],
    #     'best_video_accuracy': np.max(class_accuracy_video),
    #     'worst_video_class': category_list[np.argmin(class_accuracy_video)],
    #     'worst_video_accuracy': np.min(class_accuracy_video)
    # }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info('Testing complete! Results saved to {}'.format(args.output_dir))
    logger.info(f'Summary: Audio Accuracy: {audio_accuracy:.4f}, Video Accuracy: {video_accuracy:.4f}')
    
    # Print the most ambiguous categories
    logger.info("\nTop 10 most ambiguous audio categories:")
    for idx in np.argsort(audio_confusion_rate)[-10:]:
        if class_total_audio[idx] > 5:  # Only consider classes with sufficient samples
            logger.info(f"{category_list[idx]}: Confusion rate {audio_confusion_rate[idx]:.4f}, Samples: {int(class_total_audio[idx])}")
    
    logger.info("\nTop 10 most ambiguous video categories:")
    for idx in np.argsort(video_confusion_rate)[-10:]:
        if class_total_video[idx] > 5:  # Only consider classes with sufficient samples
            logger.info(f"{category_list[idx]}: Confusion rate {video_confusion_rate[idx]:.4f}, Samples: {int(class_total_video[idx])}")

if __name__ == '__main__':
    main()


# import logging
# import os
# import time
# import random
# import json
# from tqdm import tqdm
# import sys
# import torch
# from itertools import chain
# import torch.nn as nn
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
# import pandas as pd
# from configs.opts import parser
# from model.main_model_2 import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
# from utils import AverageMeter, Prepare_logger, get_and_save_args
# from utils.container import metricsContainer
# from transformers import BertTokenizer, BertModel

# # =================================  seed config ============================
# SEED = 43
# random.seed(SEED)
# np.random.seed(seed=SEED)
# torch.manual_seed(seed=SEED)
# torch.cuda.manual_seed(seed=SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # Initialize BERT model and tokenizer for text processing
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# def generate_category_list():
#     """Load category names from file"""
#     file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/100kcategories.txt'
#     category_list = []
#     with open(file_path, 'r') as fr:
#         for line in fr.readlines():
#             category_list.append(line.strip())
#     return category_list

# def collate_func_AVT(samples):
#     """Collate function for processing batches"""
#     bsz = len(samples)
    
#     # Get text prompts from samples
#     text_prompts = [sample['text_fea'] for sample in samples]
    
#     # Process using Transformers
#     query = []
#     query_words = []
    
#     for text in text_prompts:
#         # Tokenize
#         inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        
#         # Get embeddings
#         with torch.no_grad():
#             outputs = bert_model(**inputs)
#             # Get the last hidden state for each token
#             embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
#         # Get token IDs and convert back to tokens
#         token_ids = inputs.input_ids[0].tolist()
#         tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
#         # Remove special tokens [CLS] and [SEP]
#         non_special_tokens = tokens[1:-1]
#         non_special_embeddings = embeddings[1:-1]
        
#         # Filter tokens
#         words = []
#         words_emb = []
        
#         for token, emb in zip(non_special_tokens, non_special_embeddings):
#             # Get the token ID from the tokenizer
#             idx = tokenizer.convert_tokens_to_ids(token)
            
#             if idx != 0:
#                 words_emb.append(emb)
#                 words.append(idx)
        
#         query.append(np.asarray(words_emb))
#         query_words.append(words)

#     query_len = []
#     for i, sample in enumerate(query):
#         query_len.append(10)  # max_num_words:10
    
#     query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
#     query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
#     for i, sample in enumerate(query):
#         keep = min(sample.shape[0], query1.shape[1])
#         query1[i, :keep] = sample[:keep]
#         query_idx[i, :keep] = query_words[i][:keep]
    
#     query_len = np.asarray(query_len)
#     query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
#     query_idx = torch.from_numpy(query_idx).long()

#     return {
#         'query': query,
#         'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
#         'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float(),
#         'avel_label': torch.from_numpy(np.asarray([sample['avel_label'] for sample in samples])).float()
#     }

# def create_confusion_matrix_plot(cm, class_names, modality, output_dir, k=20):
#     """Create and save a confusion matrix visualization for top-k confused classes"""
#     # Find the most confused classes
#     np.fill_diagonal(cm, 0)  # Ignore correct classifications to find confusion
    
#     # Get sum of misclassifications for each class (row-wise)
#     class_error_totals = np.sum(cm, axis=1)
    
#     # Get top k classes with highest misclassification
#     top_confused_indices = np.argsort(class_error_totals)[-k:]
    
#     # Extract the relevant subset of the confusion matrix
#     cm_subset = cm[top_confused_indices][:, top_confused_indices]
#     names_subset = [class_names[i] for i in top_confused_indices]
    
#     # Create the figure
#     plt.figure(figsize=(16, 14))
#     sns.heatmap(cm_subset, annot=True, fmt="d", cmap="YlGnBu",
#                 xticklabels=names_subset, yticklabels=names_subset)
#     plt.title(f'Top {k} Most Confused Classes - {modality} Modality')
#     plt.ylabel('True Class')
#     plt.xlabel('Predicted Class')
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
    
#     # Save the figure
#     plt.savefig(os.path.join(output_dir, f'{modality}_top{k}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def create_per_class_accuracy_plot(accuracies, class_names, modality, output_dir, k=30):
#     """Create and save bar plots for top and bottom k classes by accuracy"""
#     # Convert to DataFrame for easier manipulation
#     df = pd.DataFrame({'class': class_names, 'accuracy': accuracies})
    
#     # Sort by accuracy
#     df_sorted = df.sort_values('accuracy')
    
#     # Plot bottom k classes (worst performing)
#     plt.figure(figsize=(14, 10))
#     bottom_k = df_sorted.head(k)
#     sns.barplot(x='accuracy', y='class', data=bottom_k, palette="YlOrRd_r")
#     plt.title(f'Bottom {k} Classes by Accuracy - {modality} Modality')
#     plt.xlabel('Accuracy')
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{modality}_bottom{k}_accuracies.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Plot top k classes (best performing)
#     plt.figure(figsize=(14, 10))
#     top_k = df_sorted.tail(k).iloc[::-1]  # Reverse to show highest on top
#     sns.barplot(x='accuracy', y='class', data=top_k, palette="YlGn")
#     plt.title(f'Top {k} Classes by Accuracy - {modality} Modality')
#     plt.xlabel('Accuracy')
#     plt.xlim(0, 1.0)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{modality}_top{k}_accuracies.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def create_most_misclassified_pairs_plot(cm, class_names, modality, output_dir, k=20):
#     """Identify and visualize the most common misclassification pairs"""
#     # Copy the confusion matrix to avoid modifying the original
#     cm_copy = cm.copy()
#     np.fill_diagonal(cm_copy, 0)  # Remove correct classifications
    
#     # Flatten and find indices of top k confused pairs
#     flat_indices = np.argsort(cm_copy.flatten())[-k:]
#     row_indices, col_indices = np.unravel_index(flat_indices, cm_copy.shape)
    
#     # Create DataFrame for the pairs
#     pairs = []
#     for i in range(k):
#         true_class = class_names[row_indices[i]]
#         pred_class = class_names[col_indices[i]]
#         count = cm_copy[row_indices[i], col_indices[i]]
#         pairs.append({
#             'True Class': true_class,
#             'Predicted Class': pred_class,
#             'Count': count
#         })
    
#     df_pairs = pd.DataFrame(pairs)
    
#     # Plot
#     plt.figure(figsize=(16, 10))
#     sns.barplot(x='Count', y='True Class', hue='Predicted Class', data=df_pairs)
#     plt.title(f'Top {k} Misclassification Pairs - {modality} Modality')
#     plt.xlabel('Number of Samples')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{modality}_top{k}_misclassification_pairs.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     # Get configurations
#     global args, logger
#     dataset_configs = get_and_save_args(parser)
#     parser.set_defaults(**dataset_configs)
#     args = parser.parse_args()
    
#     # Create output directories
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'statistics'), exist_ok=True)
    
#     # Set up logger
#     logger = Prepare_logger(args, eval=True)
#     logger.info(f'\nOutput will be saved to {args.output_dir}')
#     logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    
#     # Set GPU device
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load category list
#     category_list = generate_category_list()
#     logger.info(f'Loaded {len(category_list)} categories')
    
#     # Load dataset
#     from dataset.vgg_test_dataset import VGGSoundDataset_AVT_test as AVEDataset
    
#     meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv'
#     audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/audio80k_features_new'
#     video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/video80k_features_keras'
#     avc_label_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/100klabels'
    
#     logger.info('Loading test dataset...')
#     test_dataset = AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='test')
    
#     # Keep track of processed video IDs to avoid duplicates
#     processed_ids = set()
    
#     # Option to filter out specific categories
#     categories_to_exclude = []  # Add categories to exclude if needed
    
#     # Filter the dataset if needed
#     filtered_indices = []
#     for i in range(len(test_dataset)):
#         sample = test_dataset.split_df.iloc[i]
#         category = sample['category']
#         if category not in categories_to_exclude:
#             filtered_indices.append(i)
    
#     # Create DataLoader with filtered dataset
#     from torch.utils.data.sampler import SubsetRandomSampler
#     sampler = SubsetRandomSampler(filtered_indices)
    
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,  # No shuffle for consistent results
#         num_workers=8,
#         pin_memory=False,
#         collate_fn=collate_func_AVT,
#         sampler=sampler if len(filtered_indices) < len(test_dataset) else None
#     )
#     logger.info(f'Test dataset loaded with {len(filtered_indices) if filtered_indices else len(test_dataset)} samples')
    
#     # Initialize model
#     logger.info('Initializing model...')
#     video_dim = 512
#     text_dim = 768
#     audio_dim = 128
#     text_lstm_dim = 128
#     video_output_dim = 2048
#     text_output_dim = 256
#     audio_output_dim = 256
#     n_embeddings = 400
#     embedding_dim = 256
    
#     # Text LSTM for processing BERT embeddings
#     Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
#     Text_ar_lstm.double().to(device)
    
#     # Create encoder and decoder
#     Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)
#     Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim)
    
#     Encoder.double().to(device)
#     Decoder.double().to(device)
    
#     # Load pretrained model
#     path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVT_100k_classloss/checkpoint/DCID-model-5.pt"
#     logger.info(f'Loading pretrained model from {path_checkpoints}')
    
#     checkpoint = torch.load(path_checkpoints)
#     Encoder.load_state_dict(checkpoint['Encoder_parameters'])
    
#     # If we have the Text_ar_lstm in the checkpoint
#     if 'Text_ar_lstm_parameters' in checkpoint:
#         Text_ar_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
    
#     # If we have the Decoder in the checkpoint (optional)
#     if 'Decoder_parameters' in checkpoint:
#         Decoder.load_state_dict(checkpoint['Decoder_parameters'])
    
#     # Loss function for classification evaluation
#     criterion_event = nn.CrossEntropyLoss(reduction='none').cuda()
    
#     # Set models to evaluation mode
#     Encoder.eval()
#     Text_ar_lstm.eval()
#     Decoder.eval()
    
#     # Initialize containers for metrics
#     all_video_preds = []
#     all_audio_preds = []
#     all_text_preds = []
#     all_true_labels = []
#     all_video_ids = []  # Track video IDs for debugging
    
#     # Per-class metrics containers
#     num_classes = len(category_list)
#     class_correct_audio = np.zeros(num_classes)
#     class_total_audio = np.zeros(num_classes)
#     class_correct_video = np.zeros(num_classes)
#     class_total_video = np.zeros(num_classes)
    
#     # Per-sample loss tracking
#     audio_losses = []
#     video_losses = []
#     sample_categories = []
    
#     logger.info('Starting evaluation...')
    
#     # Process all batches
#     with torch.no_grad():
#         processed_video_ids = set()
#         for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Evaluating")):
#             # Get data from batch
#             batch_video_ids = [test_dataset.split_df.iloc[idx]['video_id'] for idx in batch_indices]
#             query = batch_data['query'].double().to(device)
#             audio_feature = batch_data['audio_fea'].double().to(device)
#             video_feature = batch_data['video_fea'].double().to(device)
#             labels = batch_data['avel_label'].to(device)
            
#             # Get video IDs from the dataset for this batch
#             batch_indices = list(range(batch_idx * args.batch_size, 
#                                  min((batch_idx + 1) * args.batch_size, len(filtered_indices or range(len(test_dataset))))))
            
#             # Extract class labels from the label tensor
#             labels = labels.double().cuda()
#             labels_foreground = labels[:, :, :-1]
#             labels_BCE, labels_evn = labels_foreground.max(-1)
#             labels_event, _ = labels_evn.max(-1)
            
#             # Use labels_event directly as sample_classes since we've already taken max across time
#             sample_classes = labels_event
            
#             # Process text through LSTM
#             batch_size = query.size()[0]
#             hidden_dim = 128
#             num_layers = 2
#             text_hidden = (
#                 torch.zeros(2*num_layers, batch_size, hidden_dim).double().to(device),
#                 torch.zeros(2*num_layers, batch_size, hidden_dim).double().to(device)
#             )
#             text_feature, _ = Text_ar_lstm(query, text_hidden)
            
#             # Forward pass through encoder
#             audio_semantic_result, video_semantic_result, text_semantic_result, \
#             audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
#             audio_vq, video_vq, text_vq, _, _, _, _, _ = Encoder(audio_feature, video_feature, text_feature, 0)
            
#             # Debug the shapes
#             if batch_idx == 0:
#                 logger.info(f"Shape of audio_semantic_result: {audio_semantic_result.shape}")
#                 logger.info(f"Shape of video_semantic_result: {video_semantic_result.shape}")
#                 logger.info(f"Shape of audio_vq: {audio_vq.shape}")
#                 logger.info(f"Shape of video_vq: {video_vq.shape}")
            
#             # Forward pass through decoder (for classification)
#             _, _, _, audio_class, video_class, text_class = Decoder(
#                 audio_feature, video_feature, text_feature,
#                 audio_encoder_result, video_encoder_result, text_encoder_result,
#                 audio_vq, video_vq, text_vq
#             )
            
#             # Debug the shapes of class outputs
#             if batch_idx == 0:
#                 logger.info(f"Shape of audio_class: {audio_class.shape}")
#                 logger.info(f"Shape of video_class: {video_class.shape}")
#                 logger.info(f"Shape of text_class: {text_class.shape}")
            
#             # Get predictions
#             audio_preds = torch.argmax(audio_class, dim=1)
#             video_preds = torch.argmax(video_class, dim=1)
#             text_preds = torch.argmax(text_class, dim=1)
            
#             # Calculate per-class statistics
#             # for i in range(batch_size):
#             #     true_class = sample_classes[i].item()
#             #     audio_pred = audio_preds[i].item()
#             #     video_pred = video_preds[i].item()
                
#             #     # Update counts
#             #     class_total_audio[true_class] += 1
#             #     class_total_video[true_class] += 1
                
#             #     if audio_pred == true_class:
#             #         class_correct_audio[true_class] += 1
                
#             #     if video_pred == true_class:
#             #         class_correct_video[true_class] += 1
#             unique_indices = set()  # Track which samples we've already processed
#             for i in range(batch_size):
#                 true_class = sample_classes[i].item()
#                 audio_pred = audio_preds[i].item()
#                 video_pred = video_preds[i].item()
                
#                 # Get a unique identifier for this sample (batch_idx * batch_size + i)
#                 sample_idx = batch_idx * args.batch_size + i
                
#                 # Only count each sample once
#                 if sample_idx not in unique_indices:
#                     unique_indices.add(sample_idx)
                    
#                     # Update counts
#                     class_total_audio[true_class] += 1
#                     class_total_video[true_class] += 1
                    
#                     if audio_pred == true_class:
#                         class_correct_audio[true_class] += 1
                    
#                     if video_pred == true_class:
#                         class_correct_video[true_class] += 1
            
#             # Calculate and store per-sample losses (only once per sample)
#             audio_loss = criterion_event(audio_class, labels_event)
#             video_loss = criterion_event(video_class, labels_event)
            
#             for i in range(batch_size):
#                 audio_losses.append((labels_event[i].item(), audio_loss[i].item()))
#                 video_losses.append((labels_event[i].item(), video_loss[i].item()))
#                 sample_categories.append(category_list[labels_event[i].item()])
            
#             # Store predictions and true labels for overall metrics
#             all_video_preds.extend(video_preds.cpu().numpy())
#             all_audio_preds.extend(audio_preds.cpu().numpy())
#             all_text_preds.extend(text_preds.cpu().numpy())
#             all_true_labels.extend(sample_classes.cpu().numpy())
            
#             # Print periodic stats for debugging
#             if (batch_idx + 1) % 10 == 0:
#                 logger.info(f"Processed {len(all_true_labels)} samples so far")
    
#     logger.info('Evaluation complete. Computing metrics...')
    
#     # Calculate overall accuracy
#     audio_accuracy = accuracy_score(all_true_labels, all_audio_preds)
#     video_accuracy = accuracy_score(all_true_labels, all_video_preds)
#     text_accuracy = accuracy_score(all_true_labels, all_text_preds)
    
#     logger.info(f'Overall Audio Accuracy: {audio_accuracy:.4f}')
#     logger.info(f'Overall Video Accuracy: {video_accuracy:.4f}')
#     logger.info(f'Overall Text Accuracy: {text_accuracy:.4f}')
    
#     # Calculate per-class accuracies
#     class_accuracy_audio = np.zeros(num_classes)
#     class_accuracy_video = np.zeros(num_classes)
    
#     for i in range(num_classes):
#         if class_total_audio[i] > 0:
#             class_accuracy_audio[i] = class_correct_audio[i] / class_total_audio[i]
        
#         if class_total_video[i] > 0:
#             class_accuracy_video[i] = class_correct_video[i] / class_total_video[i]
    
#     # Log sample counts for debugging
#     for i in range(num_classes):
#         if class_total_audio[i] > 0:
#             logger.info(f"Category '{category_list[i]}': {int(class_total_audio[i])} samples")
    
#     # Create confusion matrices
#     audio_cm = confusion_matrix(all_true_labels, all_audio_preds, labels=range(num_classes))
#     video_cm = confusion_matrix(all_true_labels, all_video_preds, labels=range(num_classes))
    
#     # Generate detailed classification reports
#     audio_report = classification_report(all_true_labels, all_audio_preds, target_names=category_list, output_dict=True)
#     video_report = classification_report(all_true_labels, all_video_preds, target_names=category_list, output_dict=True)
    
#     # Convert reports to DataFrames
#     audio_df = pd.DataFrame(audio_report).transpose()
#     video_df = pd.DataFrame(video_report).transpose()
    
#     # Save classification reports
#     audio_df.to_csv(os.path.join(args.output_dir, 'statistics', 'audio_classification_report.csv'))
#     video_df.to_csv(os.path.join(args.output_dir, 'statistics', 'video_classification_report.csv'))
    
#     # Create and save per-class statistics
#     class_stats = []
#     for i in range(num_classes):
#         stat = {
#             'category': category_list[i],
#             'total_samples': int(class_total_audio[i]),
#             'audio_correct': int(class_correct_audio[i]),
#             'audio_accuracy': class_accuracy_audio[i],
#             'video_correct': int(class_correct_video[i]),
#             'video_accuracy': class_accuracy_video[i]
#         }
#         class_stats.append(stat)
    
#     class_stats_df = pd.DataFrame(class_stats)
#     class_stats_df.to_csv(os.path.join(args.output_dir, 'statistics', 'class_statistics.csv'), index=False)
    
#     # Save per-sample losses
#     audio_loss_df = pd.DataFrame(audio_losses, columns=['class_idx', 'loss'])
#     audio_loss_df['category'] = audio_loss_df['class_idx'].apply(lambda x: category_list[x])
#     audio_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'audio_sample_losses.csv'), index=False)
    
#     video_loss_df = pd.DataFrame(video_losses, columns=['class_idx', 'loss'])
#     video_loss_df['category'] = video_loss_df['class_idx'].apply(lambda x: category_list[x])
#     video_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'video_sample_losses.csv'), index=False)
    
#     # Calculate average loss per class
#     audio_class_avg_loss = audio_loss_df.groupby('class_idx')['loss'].mean()
#     video_class_avg_loss = video_loss_df.groupby('class_idx')['loss'].mean()
    
#     avg_loss_df = pd.DataFrame({
#         'category': category_list,
#         'audio_avg_loss': audio_class_avg_loss.values,
#         'video_avg_loss': video_class_avg_loss.values
#     })
#     avg_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'average_class_loss.csv'), index=False)
    
#     # Create visualizations
#     logger.info('Generating visualizations...')
#     vis_dir = os.path.join(args.output_dir, 'visualizations')
    
#     # Create confusion matrix visualizations
#     create_confusion_matrix_plot(audio_cm, category_list, 'Audio', vis_dir)
#     create_confusion_matrix_plot(video_cm, category_list, 'Video', vis_dir)
    
#     # Create per-class accuracy plots
#     create_per_class_accuracy_plot(class_accuracy_audio, category_list, 'Audio', vis_dir)
#     create_per_class_accuracy_plot(class_accuracy_video, category_list, 'Video', vis_dir)
    
#     # Create misclassification pair plots
#     create_most_misclassified_pairs_plot(audio_cm, category_list, 'Audio', vis_dir)
#     create_most_misclassified_pairs_plot(video_cm, category_list, 'Video', vis_dir)
    
#     # Find the most ambiguous categories (highest confusion)
#     audio_confusion_sum = np.sum(audio_cm, axis=1) - np.diag(audio_cm)
#     video_confusion_sum = np.sum(video_cm, axis=1) - np.diag(video_cm)
    
#     # Normalize by number of samples
#     audio_confusion_rate = np.zeros(num_classes)
#     video_confusion_rate = np.zeros(num_classes)
    
#     for i in range(num_classes):
#         if np.sum(audio_cm[i]) > 0:
#             audio_confusion_rate[i] = audio_confusion_sum[i] / np.sum(audio_cm[i])
#         if np.sum(video_cm[i]) > 0:
#             video_confusion_rate[i] = video_confusion_sum[i] / np.sum(video_cm[i])
    
#     # Create DataFrame for ambiguous classes
#     ambiguous_df = pd.DataFrame({
#         'category': category_list,
#         'audio_confusion_rate': audio_confusion_rate,
#         'video_confusion_rate': video_confusion_rate,
#         'sample_count': np.sum(audio_cm, axis=1)
#     })
    
#     # Save to CSV
#     ambiguous_df.sort_values('audio_confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'audio_ambiguous_categories.csv'), index=False)
    
#     ambiguous_df.sort_values('video_confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'video_ambiguous_categories.csv'), index=False)
    
#     # Find the most confused pairs for each category
#     most_confused_pairs = []
#     for i in range(num_classes):
#         # Skip categories with no samples
#         if np.sum(audio_cm[i]) == 0:
#             continue
            
#         # Skip diagonal (correct classification)
#         confusion_row = audio_cm[i].copy()
#         confusion_row[i] = 0
        
#         # Find top confused class
#         if np.sum(confusion_row) > 0:
#             most_confused_idx = np.argmax(confusion_row)
#             confusion_rate = confusion_row[most_confused_idx] / np.sum(audio_cm[i])
#             most_confused_pairs.append({
#                 'category': category_list[i],
#                 'most_confused_with': category_list[most_confused_idx],
#                 'confusion_count': int(confusion_row[most_confused_idx]),
#                 'total_samples': int(np.sum(audio_cm[i])),
#                 'confusion_rate': confusion_rate
#             })
    
#     confused_pairs_df = pd.DataFrame(most_confused_pairs)
#     confused_pairs_df.sort_values('confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'audio_most_confused_pairs.csv'), index=False)
    
#     # Same for video
#     most_confused_pairs = []
#     for i in range(num_classes):
#         # Skip categories with no samples
#         if np.sum(video_cm[i]) == 0:
#             continue
            
#         confusion_row = video_cm[i].copy()
#         confusion_row[i] = 0
        
#         if np.sum(confusion_row) > 0:
#             most_confused_idx = np.argmax(confusion_row)
#             confusion_rate = confusion_row[most_confused_idx] / np.sum(video_cm[i])
#             most_confused_pairs.append({
#                 'category': category_list[i],
#                 'most_confused_with': category_list[most_confused_idx],
#                 'confusion_count': int(confusion_row[most_confused_idx]),
#                 'total_samples': int(np.sum(video_cm[i])),
#                 'confusion_rate': confusion_rate
#             })
    
#     confused_pairs_df = pd.DataFrame(most_confused_pairs)
#     confused_pairs_df.sort_values('confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'video_most_confused_pairs.csv'), index=False)
    
#     # Create summary of results
#     summary = {
#         'audio_accuracy': float(audio_accuracy),
#         'video_accuracy': float(video_accuracy),
#         'text_accuracy': float(text_accuracy),
#         'total_samples': int(len(all_true_labels)),
#         'total_classes': int(num_classes),
#         'best_audio_class': category_list[np.argmax(class_accuracy_audio)],
#         'best_audio_accuracy': float(np.max(class_accuracy_audio)),
#         'worst_audio_class': category_list[np.argmin(class_accuracy_audio)],
#         'worst_audio_accuracy': float(np.min(class_accuracy_audio)),
#         'best_video_class': category_list[np.argmax(class_accuracy_video)],
#         'best_video_accuracy': float(np.max(class_accuracy_video)),
#         'worst_video_class': category_list[np.argmin(class_accuracy_video)],
#         'worst_video_accuracy': float(np.min(class_accuracy_video))
#     }
    
#     with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=4)
    
#     logger.info('Testing complete! Results saved to {}'.format(args.output_dir))
#     logger.info(f'Summary: Audio Accuracy: {audio_accuracy:.4f}, Video Accuracy: {video_accuracy:.4f}')
    
#     # Print the most ambiguous categories
#     logger.info("\nTop 10 most ambiguous audio categories:")
#     for idx in np.argsort(audio_confusion_rate)[-10:]:
#         if class_total_audio[idx] > 5:  # Only consider classes with sufficient samples
#             logger.info(f"{category_list[idx]}: Confusion rate {audio_confusion_rate[idx]:.4f}, Samples: {int(class_total_audio[idx])}")
    
#     logger.info("\nTop 10 most ambiguous video categories:")
#     for idx in np.argsort(video_confusion_rate)[-10:]:
#         if class_total_video[idx] > 5:  # Only consider classes with sufficient samples
#             logger.info(f"{category_list[idx]}: Confusion rate {video_confusion_rate[idx]:.4f}, Samples: {int(class_total_video[idx])}")

# if __name__ == '__main__':
#     main()

# import logging
# import os
# import time
# import random
# import json
# from tqdm import tqdm
# import sys
# import torch
# from itertools import chain
# import torch.nn as nn
# from torch.nn.utils import clip_grad_norm_
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
# import pandas as pd
# from configs.opts import parser
# from model.main_model_2 import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
# from utils import AverageMeter, Prepare_logger, get_and_save_args
# from utils.container import metricsContainer
# from transformers import BertTokenizer, BertModel

# # =================================  seed config ============================
# SEED = 43
# random.seed(SEED)
# np.random.seed(seed=SEED)
# torch.manual_seed(seed=SEED)
# torch.cuda.manual_seed(seed=SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # Initialize BERT model and tokenizer for text processing
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')

# def generate_category_list():
#     """Load category names from file"""
#     file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/100kcategories.txt'
#     category_list = []
#     with open(file_path, 'r') as fr:
#         for line in fr.readlines():
#             category_list.append(line.strip())
#     return category_list

# def collate_func_AVT(samples):
#     """Collate function for processing batches"""
#     bsz = len(samples)
    
#     # Get text prompts from samples
#     text_prompts = [sample['text_fea'] for sample in samples]
    
#     # Process using Transformers
#     query = []
#     query_words = []
    
#     for text in text_prompts:
#         # Tokenize
#         inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        
#         # Get embeddings
#         with torch.no_grad():
#             outputs = bert_model(**inputs)
#             # Get the last hidden state for each token
#             embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
#         # Get token IDs and convert back to tokens
#         token_ids = inputs.input_ids[0].tolist()
#         tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
#         # Remove special tokens [CLS] and [SEP]
#         non_special_tokens = tokens[1:-1]
#         non_special_embeddings = embeddings[1:-1]
        
#         # Filter tokens
#         words = []
#         words_emb = []
        
#         for token, emb in zip(non_special_tokens, non_special_embeddings):
#             # Get the token ID from the tokenizer
#             idx = tokenizer.convert_tokens_to_ids(token)
            
#             if idx != 0:
#                 words_emb.append(emb)
#                 words.append(idx)
        
#         query.append(np.asarray(words_emb))
#         query_words.append(words)

#     query_len = []
#     for i, sample in enumerate(query):
#         query_len.append(10)  # max_num_words:10
    
#     query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
#     query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
#     for i, sample in enumerate(query):
#         keep = min(sample.shape[0], query1.shape[1])
#         query1[i, :keep] = sample[:keep]
#         query_idx[i, :keep] = query_words[i][:keep]
    
#     query_len = np.asarray(query_len)
#     query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
#     query_idx = torch.from_numpy(query_idx).long()

#     return {
#         'query': query,
#         'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
#         'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float(),
#         'avel_label': torch.from_numpy(np.asarray([sample['avel_label'] for sample in samples])).float(),
#         'video_ids': [sample.get('video_id', 'unknown') for sample in samples]
#     }

# def create_confusion_matrix_plot(cm, class_names, modality, output_dir, k=20):
#     """Create and save a confusion matrix visualization for top-k confused classes"""
#     # Find the most confused classes
#     np.fill_diagonal(cm, 0)  # Ignore correct classifications to find confusion
    
#     # Get sum of misclassifications for each class (row-wise)
#     class_error_totals = np.sum(cm, axis=1)
    
#     # Get top k classes with highest misclassification
#     # Make sure we have samples for these classes (error_totals > 0)
#     valid_indices = np.where(class_error_totals > 0)[0]
#     top_k_valid = min(k, len(valid_indices))
#     top_confused_indices = valid_indices[np.argsort(class_error_totals[valid_indices])[-top_k_valid:]]
    
#     # Extract the relevant subset of the confusion matrix
#     cm_subset = cm[top_confused_indices][:, top_confused_indices]
#     names_subset = [class_names[i] for i in top_confused_indices]
    
#     # Create the figure
#     plt.figure(figsize=(16, 14))
#     sns.heatmap(cm_subset, annot=True, fmt="d", cmap="YlGnBu",
#                 xticklabels=names_subset, yticklabels=names_subset)
#     plt.title(f'Top {top_k_valid} Most Confused Classes - {modality} Modality')
#     plt.ylabel('True Class')
#     plt.xlabel('Predicted Class')
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
    
#     # Save the figure
#     plt.savefig(os.path.join(output_dir, f'{modality}_top{top_k_valid}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def create_per_class_accuracy_plot(accuracies, class_names, class_samples, modality, output_dir, k=30, min_samples=5):
#     """Create and save bar plots for top and bottom k classes by accuracy"""
#     # Filter for classes with minimum samples
#     valid_indices = np.where(class_samples >= min_samples)[0]
#     valid_accuracies = accuracies[valid_indices]
#     valid_names = [class_names[i] for i in valid_indices]
    
#     # Convert to DataFrame for easier manipulation
#     df = pd.DataFrame({
#         'class': valid_names, 
#         'accuracy': valid_accuracies,
#         'samples': class_samples[valid_indices]
#     })
    
#     # Sort by accuracy
#     df_sorted = df.sort_values('accuracy')
    
#     # Plot bottom k classes (worst performing)
#     k_bottom = min(k, len(df_sorted))
#     plt.figure(figsize=(14, 10))
#     bottom_k = df_sorted.head(k_bottom)
#     sns.barplot(x='accuracy', y='class', data=bottom_k, palette="YlOrRd_r")
#     plt.title(f'Bottom {k_bottom} Classes by Accuracy - {modality} Modality')
#     plt.xlabel('Accuracy')
#     plt.xlim(0, 1.0)
    
#     # Add sample count annotations
#     for i, row in enumerate(bottom_k.itertuples()):
#         plt.text(0.01, i, f"n={int(row.samples)}", va='center', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{modality}_bottom{k_bottom}_accuracies.png'), dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Plot top k classes (best performing)
#     k_top = min(k, len(df_sorted))
#     plt.figure(figsize=(14, 10))
#     top_k = df_sorted.tail(k_top).iloc[::-1]  # Reverse to show highest on top
#     sns.barplot(x='accuracy', y='class', data=top_k, palette="YlGn")
#     plt.title(f'Top {k_top} Classes by Accuracy - {modality} Modality')
#     plt.xlabel('Accuracy')
#     plt.xlim(0, 1.0)
    
#     # Add sample count annotations
#     for i, row in enumerate(top_k.itertuples()):
#         plt.text(0.01, i, f"n={int(row.samples)}", va='center', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{modality}_top{k_top}_accuracies.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def create_most_misclassified_pairs_plot(cm, class_names, modality, output_dir, k=20):
#     """Identify and visualize the most common misclassification pairs"""
#     # Copy the confusion matrix to avoid modifying the original
#     cm_copy = cm.copy()
#     np.fill_diagonal(cm_copy, 0)  # Remove correct classifications
    
#     # Check if there are enough non-zero values
#     if np.sum(cm_copy > 0) == 0:
#         print(f"Warning: No misclassifications found for {modality} modality")
#         return
    
#     # Flatten and find indices of top k confused pairs
#     flat_indices = np.argsort(cm_copy.flatten())[-k:]
#     row_indices, col_indices = np.unravel_index(flat_indices, cm_copy.shape)
    
#     # Create DataFrame for the pairs
#     pairs = []
#     for i in range(min(k, len(flat_indices))):
#         true_class = class_names[row_indices[i]]
#         pred_class = class_names[col_indices[i]]
#         count = cm_copy[row_indices[i], col_indices[i]]
#         if count > 0:  # Only add non-zero counts
#             pairs.append({
#                 'True Class': true_class,
#                 'Predicted Class': pred_class,
#                 'Count': count
#             })
    
#     if not pairs:
#         print(f"Warning: No valid misclassification pairs found for {modality} modality")
#         return
        
#     df_pairs = pd.DataFrame(pairs)
    
#     # Plot
#     plt.figure(figsize=(16, 10))
#     sns.barplot(x='Count', y='True Class', hue='Predicted Class', data=df_pairs)
#     plt.title(f'Top {len(pairs)} Misclassification Pairs - {modality} Modality')
#     plt.xlabel('Number of Samples')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{modality}_top{len(pairs)}_misclassification_pairs.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     # Get configurations
#     global args, logger
#     dataset_configs = get_and_save_args(parser)
#     parser.set_defaults(**dataset_configs)
#     args = parser.parse_args()
    
#     # Create output directories
#     os.makedirs(args.output_dir, exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
#     os.makedirs(os.path.join(args.output_dir, 'statistics'), exist_ok=True)
    
#     # Set up logger
#     logger = Prepare_logger(args, eval=True)
#     logger.info(f'\nOutput will be saved to {args.output_dir}')
#     logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    
#     # Set GPU device
#     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Load category list
#     category_list = generate_category_list()
#     logger.info(f'Loaded {len(category_list)} categories')
    
#     # Load dataset
#     from dataset.vgg_test_dataset import VGGSoundDataset_AVT_test as AVEDataset
    
#     meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv'
#     audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/audio80k_features_new'
#     video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/video80k_features_keras'
#     avc_label_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/100klabels'
    
#     logger.info('Loading test dataset...')
#     test_dataset = AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='test')
    
#     # Check for duplicates in the dataset
#     all_video_ids = test_dataset.split_df['video_id'].tolist()
#     unique_video_ids = set(all_video_ids)
#     logger.info(f"Total test videos: {len(all_video_ids)}")
#     logger.info(f"Unique test videos: {len(unique_video_ids)}")
    
#     if len(all_video_ids) != len(unique_video_ids):
#         logger.warning("WARNING: Duplicate video IDs found in test set!")
    
#     # ===== THIS IS THE KEY CHANGE: Create a mapping to process each video exactly once =====
#     # Create a mapping from video_ids to their dataframe index
#     video_id_to_idx = {}
#     for i, video_id in enumerate(all_video_ids):
#         # Only store the first occurrence of each video_id
#         if video_id not in video_id_to_idx:
#             video_id_to_idx[video_id] = i
            
#     logger.info(f"After deduplication: {len(video_id_to_idx)} unique videos will be processed")
    
#     # Initialize model
#     logger.info('Initializing model...')
#     video_dim = 512
#     text_dim = 768
#     audio_dim = 128
#     text_lstm_dim = 128
#     video_output_dim = 2048
#     text_output_dim = 256
#     audio_output_dim = 256
#     n_embeddings = 400
#     embedding_dim = 256
    
#     # Text LSTM for processing BERT embeddings
#     Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
#     Text_ar_lstm.double().to(device)
    
#     # Create encoder and decoder
#     Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)
#     Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim)
    
#     Encoder.double().to(device)
#     Decoder.double().to(device)
    
#     # Load pretrained model
#     path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVT_100k_classloss/checkpoint/DCID-model-5.pt"
#     logger.info(f'Loading pretrained model from {path_checkpoints}')
    
#     checkpoint = torch.load(path_checkpoints)
#     Encoder.load_state_dict(checkpoint['Encoder_parameters'])
    
#     # If we have the Text_ar_lstm in the checkpoint
#     if 'Text_ar_lstm_parameters' in checkpoint:
#         Text_ar_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
    
#     # If we have the Decoder in the checkpoint (optional)
#     if 'Decoder_parameters' in checkpoint:
#         Decoder.load_state_dict(checkpoint['Decoder_parameters'])
    
#     # Loss function for classification evaluation
#     criterion_event = nn.CrossEntropyLoss(reduction='none').cuda()
    
#     # Set models to evaluation mode
#     Encoder.eval()
#     Text_ar_lstm.eval()
#     Decoder.eval()
    
#     # Initialize containers for metrics
#     all_video_preds = []
#     all_audio_preds = []
#     all_text_preds = []
#     all_true_labels = []
#     all_processed_ids = set()  # Track processed video IDs to avoid duplicates
    
#     # Per-class metrics containers
#     num_classes = len(category_list)
#     class_correct_audio = np.zeros(num_classes)
#     class_total_audio = np.zeros(num_classes)
#     class_correct_video = np.zeros(num_classes)
#     class_total_video = np.zeros(num_classes)
    
#     # Per-sample loss tracking
#     audio_losses = []
#     video_losses = []
#     sample_categories = []
    
#     # Process unique samples directly, one by one
#     logger.info('Starting evaluation...')
    
#     # ===== THIS IS THE KEY CHANGE: Process each unique video individually =====
#     with torch.no_grad():
#         for idx, (video_id, idx_in_dataset) in enumerate(tqdm(video_id_to_idx.items(), desc="Evaluating")):
#             # Skip if already processed (safety check)
#             if video_id in all_processed_ids:
#                 continue
            
#             all_processed_ids.add(video_id)
            
#             # Get a single sample
#             batch_data = collate_func_AVT([test_dataset[idx_in_dataset]])
            
#             # Process the single sample through model
#             query = batch_data['query'].double().to(device)
#             audio_feature = batch_data['audio_fea'].double().to(device)
#             video_feature = batch_data['video_fea'].double().to(device)
#             labels = batch_data['avel_label'].to(device)
            
#             # Extract true class
#             labels = labels.double().cuda()
#             labels_foreground = labels[:, :, :-1]
#             labels_BCE, labels_evn = labels_foreground.max(-1)
#             labels_event, _ = labels_evn.max(-1)
#             true_class = labels_event.item()
            
#             # Process text through LSTM
#             batch_size = 1  # We're processing one sample at a time
#             hidden_dim = 128
#             num_layers = 2
#             text_hidden = (
#                 torch.zeros(2*num_layers, batch_size, hidden_dim).double().to(device),
#                 torch.zeros(2*num_layers, batch_size, hidden_dim).double().to(device)
#             )
#             text_feature, _ = Text_ar_lstm(query, text_hidden)
            
#             # Forward pass through encoder
#             audio_semantic_result, video_semantic_result, text_semantic_result, \
#             audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
#             audio_vq, video_vq, text_vq, _, _, _, _, _ = Encoder(audio_feature, video_feature, text_feature, 0)
            
#             # Debug shapes for first sample
#             if idx == 0:
#                 logger.info(f"Shape of audio_semantic_result: {audio_semantic_result.shape}")
#                 logger.info(f"Shape of video_semantic_result: {video_semantic_result.shape}")
#                 logger.info(f"Shape of audio_vq: {audio_vq.shape}")
#                 logger.info(f"Shape of video_vq: {video_vq.shape}")
            
#             # Forward pass through decoder (for classification)
#             _, _, _, audio_class, video_class, text_class = Decoder(
#                 audio_feature, video_feature, text_feature,
#                 audio_encoder_result, video_encoder_result, text_encoder_result,
#                 audio_vq, video_vq, text_vq
#             )
            
#             # Debug shapes for first sample
#             if idx == 0:
#                 logger.info(f"Shape of audio_class: {audio_class.shape}")
#                 logger.info(f"Shape of video_class: {video_class.shape}")
#                 logger.info(f"Shape of text_class: {text_class.shape}")
            
#             # Get predictions
#             audio_pred = torch.argmax(audio_class, dim=1).item()
#             video_pred = torch.argmax(video_class, dim=1).item()
#             text_pred = torch.argmax(text_class, dim=1).item()
            
#             # Update per-class statistics
#             class_total_audio[true_class] += 1
#             class_total_video[true_class] += 1
            
#             if audio_pred == true_class:
#                 class_correct_audio[true_class] += 1
            
#             if video_pred == true_class:
#                 class_correct_video[true_class] += 1
            
#             # Calculate losses
#             audio_loss = criterion_event(audio_class, labels_event).item()
#             video_loss = criterion_event(video_class, labels_event).item()
            
#             # Store losses and categories
#             audio_losses.append((true_class, audio_loss))
#             video_losses.append((true_class, video_loss))
#             sample_categories.append(category_list[true_class])
            
#             # Store predictions and true labels
#             all_video_preds.append(video_pred)
#             all_audio_preds.append(audio_pred)
#             all_text_preds.append(text_pred)
#             all_true_labels.append(true_class)
            
#             # Print periodic progress
#             if (idx + 1) % 100 == 0:
#                 logger.info(f"Processed {idx+1}/{len(video_id_to_idx)} unique videos")
    
#     logger.info(f'Evaluation complete. Processed {len(all_processed_ids)} unique videos.')
    
#     # Verify counts match expectations
#     category_counts = {}
#     for label in all_true_labels:
#         category = category_list[label]
#         category_counts[category] = category_counts.get(category, 0) + 1
    
#     logger.info("Category distribution in test predictions:")
#     for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
#         logger.info(f"  {category}: {count} samples")
    
#     # Calculate overall accuracy
#     audio_accuracy = accuracy_score(all_true_labels, all_audio_preds)
#     video_accuracy = accuracy_score(all_true_labels, all_video_preds)
#     text_accuracy = accuracy_score(all_true_labels, all_text_preds)
    
#     logger.info(f'Overall Audio Accuracy: {audio_accuracy:.4f}')
#     logger.info(f'Overall Video Accuracy: {video_accuracy:.4f}')
#     logger.info(f'Overall Text Accuracy: {text_accuracy:.4f}')
    
#     # Calculate per-class accuracies
#     class_accuracy_audio = np.zeros(num_classes)
#     class_accuracy_video = np.zeros(num_classes)
    
#     for i in range(num_classes):
#         if class_total_audio[i] > 0:
#             class_accuracy_audio[i] = class_correct_audio[i] / class_total_audio[i]
        
#         if class_total_video[i] > 0:
#             class_accuracy_video[i] = class_correct_video[i] / class_total_video[i]
    
#     # Create confusion matrices
#     audio_cm = confusion_matrix(all_true_labels, all_audio_preds, labels=range(num_classes))
#     video_cm = confusion_matrix(all_true_labels, all_video_preds, labels=range(num_classes))
    
#     # Generate detailed classification reports
#     audio_report = classification_report(all_true_labels, all_audio_preds, target_names=category_list, output_dict=True)
#     video_report = classification_report(all_true_labels, all_video_preds, target_names=category_list, output_dict=True)
    
#     # Convert reports to DataFrames
#     audio_df = pd.DataFrame(audio_report).transpose()
#     video_df = pd.DataFrame(video_report).transpose()
    
#     # Save classification reports
#     audio_df.to_csv(os.path.join(args.output_dir, 'statistics', 'audio_classification_report.csv'))
#     video_df.to_csv(os.path.join(args.output_dir, 'statistics', 'video_classification_report.csv'))
    
#     # Create and save per-class statistics
#     class_stats = []
#     for i in range(num_classes):
#         stat = {
#             'category': category_list[i],
#             'total_samples': int(class_total_audio[i]),
#             'audio_correct': int(class_correct_audio[i]),
#             'audio_accuracy': class_accuracy_audio[i],
#             'video_correct': int(class_correct_video[i]),
#             'video_accuracy': class_accuracy_video[i]
#         }
#         class_stats.append(stat)
    
#     class_stats_df = pd.DataFrame(class_stats)
#     class_stats_df.to_csv(os.path.join(args.output_dir, 'statistics', 'class_statistics.csv'), index=False)
    
#     # Save per-sample losses
#     audio_loss_df = pd.DataFrame(audio_losses, columns=['class_idx', 'loss'])
#     audio_loss_df['category'] = audio_loss_df['class_idx'].apply(lambda x: category_list[x])
#     audio_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'audio_sample_losses.csv'), index=False)
    
#     video_loss_df = pd.DataFrame(video_losses, columns=['class_idx', 'loss'])
#     video_loss_df['category'] = video_loss_df['class_idx'].apply(lambda x: category_list[x])
#     video_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'video_sample_losses.csv'), index=False)
    
#     # Calculate average loss per class
#     audio_class_avg_loss = audio_loss_df.groupby('class_idx')['loss'].mean()
#     video_class_avg_loss = video_loss_df.groupby('class_idx')['loss'].mean()
    
#     # Ensure all classes have a loss value (even if zero samples)
#     all_class_indices = list(range(num_classes))
#     missing_audio_classes = set(all_class_indices) - set(audio_class_avg_loss.index)
#     missing_video_classes = set(all_class_indices) - set(video_class_avg_loss.index)
    
#     # Add missing classes with NaN values
#     for idx in missing_audio_classes:
#         audio_class_avg_loss[idx] = np.nan
#     for idx in missing_video_classes:
#         video_class_avg_loss[idx] = np.nan
    
#     # Sort by index to ensure alignment with category_list
#     audio_class_avg_loss = audio_class_avg_loss.sort_index()
#     video_class_avg_loss = video_class_avg_loss.sort_index()
    
#     avg_loss_df = pd.DataFrame({
#         'category': category_list,
#         'audio_avg_loss': audio_class_avg_loss.values,
#         'video_avg_loss': video_class_avg_loss.values,
#         'sample_count': [class_total_audio[i] for i in range(num_classes)]
#     })
#     avg_loss_df.to_csv(os.path.join(args.output_dir, 'statistics', 'average_class_loss.csv'), index=False)
    
#     # Create visualizations
#     logger.info('Generating visualizations...')
#     vis_dir = os.path.join(args.output_dir, 'visualizations')
    
#     # Create confusion matrix visualizations
#     create_confusion_matrix_plot(audio_cm, category_list, 'Audio', vis_dir)
#     create_confusion_matrix_plot(video_cm, category_list, 'Video', vis_dir)
    
#     # Create per-class accuracy plots
#     create_per_class_accuracy_plot(class_accuracy_audio, category_list, class_total_audio, 'Audio', vis_dir)
#     create_per_class_accuracy_plot(class_accuracy_video, category_list, class_total_video, 'Video', vis_dir)
    
#     # Create misclassification pair plots
#     create_most_misclassified_pairs_plot(audio_cm, category_list, 'Audio', vis_dir)
#     create_most_misclassified_pairs_plot(video_cm, category_list, 'Video', vis_dir)
    
#     # Find the most ambiguous categories (highest confusion)
#     audio_confusion_sum = np.sum(audio_cm, axis=1) - np.diag(audio_cm)
#     video_confusion_sum = np.sum(video_cm, axis=1) - np.diag(video_cm)
    
#     # Normalize by number of samples (avoiding division by zero)
#     audio_confusion_rate = np.zeros(num_classes)
#     video_confusion_rate = np.zeros(num_classes)
    
#     for i in range(num_classes):
#         if np.sum(audio_cm[i]) > 0:
#             audio_confusion_rate[i] = audio_confusion_sum[i] / np.sum(audio_cm[i])
#         if np.sum(video_cm[i]) > 0:
#             video_confusion_rate[i] = video_confusion_sum[i] / np.sum(video_cm[i])
    
#     # Create DataFrame for ambiguous classes
#     ambiguous_df = pd.DataFrame({
#         'category': category_list,
#         'audio_confusion_rate': audio_confusion_rate,
#         'video_confusion_rate': video_confusion_rate,
#         'sample_count': np.sum(audio_cm, axis=1)
#     })
    
#     # Save to CSV
#     ambiguous_df.sort_values('audio_confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'audio_ambiguous_categories.csv'), index=False)
    
#     ambiguous_df.sort_values('video_confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'video_ambiguous_categories.csv'), index=False)
    
#     # Find the most confused pairs for each category
#     most_confused_pairs = []
#     for i in range(num_classes):
#         # Skip categories with no samples
#         if np.sum(audio_cm[i]) == 0:
#             continue
            
#         # Skip diagonal (correct classification)
#         confusion_row = audio_cm[i].copy()
#         confusion_row[i] = 0
        
#         # Find top confused class
#         if np.sum(confusion_row) > 0:
#             most_confused_idx = np.argmax(confusion_row)
#             confusion_rate = confusion_row[most_confused_idx] / np.sum(audio_cm[i])
#             most_confused_pairs.append({
#                 'category': category_list[i],
#                 'most_confused_with': category_list[most_confused_idx],
#                 'confusion_count': int(confusion_row[most_confused_idx]),
#                 'total_samples': int(np.sum(audio_cm[i])),
#                 'confusion_rate': confusion_rate
#             })
    
#     confused_pairs_df = pd.DataFrame(most_confused_pairs)
#     confused_pairs_df.sort_values('confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'audio_most_confused_pairs.csv'), index=False)
    
#     # Same for video
#     most_confused_pairs = []
#     for i in range(num_classes):
#         # Skip categories with no samples
#         if np.sum(video_cm[i]) == 0:
#             continue
            
#         confusion_row = video_cm[i].copy()
#         confusion_row[i] = 0
        
#         if np.sum(confusion_row) > 0:
#             most_confused_idx = np.argmax(confusion_row)
#             confusion_rate = confusion_row[most_confused_idx] / np.sum(video_cm[i])
#             most_confused_pairs.append({
#                 'category': category_list[i],
#                 'most_confused_with': category_list[most_confused_idx],
#                 'confusion_count': int(confusion_row[most_confused_idx]),
#                 'total_samples': int(np.sum(video_cm[i])),
#                 'confusion_rate': confusion_rate
#             })
    
#     confused_pairs_df = pd.DataFrame(most_confused_pairs)
#     confused_pairs_df.sort_values('confusion_rate', ascending=False).to_csv(
#         os.path.join(args.output_dir, 'statistics', 'video_most_confused_pairs.csv'), index=False)
    
#     # Create summary of results
#     summary = {
#         'audio_accuracy': float(audio_accuracy),
#         'video_accuracy': float(video_accuracy),
#         'text_accuracy': float(text_accuracy),
#         'total_samples': int(len(all_true_labels)),
#         'total_classes': int(num_classes),
#         'best_audio_class': category_list[np.argmax(class_accuracy_audio)],
#         'best_audio_accuracy': float(np.max(class_accuracy_audio)),
#         'worst_audio_class': category_list[np.argmin(class_accuracy_audio)],
#         'worst_audio_accuracy': float(np.min(class_accuracy_audio)),
#         'best_video_class': category_list[np.argmax(class_accuracy_video)],
#         'best_video_accuracy': float(np.max(class_accuracy_video)),
#         'worst_video_class': category_list[np.argmin(class_accuracy_video)],
#         'worst_video_accuracy': float(np.min(class_accuracy_video))
#     }
    
#     with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=4)
    
#     logger.info('Testing complete! Results saved to {}'.format(args.output_dir))
#     logger.info(f'Summary: Audio Accuracy: {audio_accuracy:.4f}, Video Accuracy: {video_accuracy:.4f}')
    
#     # Print the most ambiguous categories
#     logger.info("\nTop 10 most ambiguous audio categories:")
#     ambiguous_indices = np.argsort(audio_confusion_rate)[-10:]
#     for idx in reversed(ambiguous_indices):
#         if class_total_audio[idx] > 5:  # Only consider classes with sufficient samples
#             logger.info(f"{category_list[idx]}: Confusion rate {audio_confusion_rate[idx]:.4f}, Samples: {int(class_total_audio[idx])}")
    
#     logger.info("\nTop 10 most ambiguous video categories:")
#     ambiguous_indices = np.argsort(video_confusion_rate)[-10:]
#     for idx in reversed(ambiguous_indices):
#         if class_total_video[idx] > 5:  # Only consider classes with sufficient samples
#             logger.info(f"{category_list[idx]}: Confusion rate {video_confusion_rate[idx]:.4f}, Samples: {int(class_total_video[idx])}")

# if __name__ == '__main__':
#     main()
