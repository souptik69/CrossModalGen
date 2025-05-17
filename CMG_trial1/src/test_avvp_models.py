import logging
import os
import time
import random
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import argparse
from model.main_model_2 import AVT_VQVAE_Encoder, Semantic_Decoder_AVVP

# Import the dataset class
from dataset.AVVP_dataset import AVVPDataset

# Set up argument parser
parser = argparse.ArgumentParser(description='Test A2V or V2A model on AVVP dataset')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for testing (use 1 for variable length)')
parser.add_argument('--output_dir', default='results', type=str, help='Output directory for results')
parser.add_argument('--model_path', default='', type=str, help='Path to model checkpoint')
parser.add_argument('--model_type', default='', type=str, choices=['A2V', 'V2A'], help='Type of model to test (A2V or V2A)')
parser.add_argument('--meta_csv_path', default='', type=str, help='Path to the test CSV file')
parser.add_argument('--fea_path', default='', type=str, help='Path to features (audio or video)')

# Set random seeds for reproducibility
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def setup_logger(output_dir):
    """Set up logging configuration"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_file = os.path.join(output_dir, 'test_log.txt')
    
    # Configure logger
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# def create_confusion_matrix_plot(cm, class_names, modality, output_dir, k=15):
#     """Create and save a confusion matrix visualization for top-k confused classes"""
#     # Find the most confused classes
#     np.fill_diagonal(cm, 0)  # Ignore correct classifications
    
#     # Get sum of misclassifications for each class (row-wise)
#     class_error_totals = np.sum(cm, axis=1)
    
#     # Get top k classes with highest misclassification (skip those with no samples)
#     valid_indices = np.where(class_error_totals > 0)[0]
#     k = min(k, len(valid_indices))
#     top_confused_indices = valid_indices[np.argsort(class_error_totals[valid_indices])[-k:]]
    
#     # Extract the relevant subset of the confusion matrix
#     cm_subset = cm[top_confused_indices][:, top_confused_indices]
#     names_subset = [class_names[i] for i in top_confused_indices]
    
#     # Create the figure
#     plt.figure(figsize=(20, 18))
#     sns.heatmap(cm_subset, annot=True, fmt="d", cmap="YlGnBu",
#                 xticklabels=names_subset, yticklabels=names_subset)
#     plt.title(f'Top {k} Most Confused Classes - {modality} Model')
#     plt.ylabel('True Class')
#     plt.xlabel('Predicted Class')
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()
    
#     # Save the figure
#     plt.savefig(os.path.join(output_dir, f'top{k}_confusion_matrix.png'), 
#                 dpi=300, bbox_inches='tight')
#     plt.close()

def create_confusion_matrix_plot(cm, class_names, modality, output_dir, k=15):
    """Create and save a confusion matrix visualization for top-k confused classes"""
    # Find the most confused classes
    # np.fill_diagonal(cm, 0)  # Ignore correct classifications
    cm_working = cm.copy()

    # Get sum of misclassifications for each class (row-wise)
    # class_error_totals = np.sum(cm, axis=1)
    np.fill_diagonal(cm_working, 0)
    class_error_totals = np.sum(cm_working, axis=1)
    
    # Get top k classes with highest misclassification (skip those with no samples)
    valid_indices = np.where(class_error_totals > 0)[0]
    k = min(k, len(valid_indices))
    
    if len(valid_indices) == 0:
        print(f"No confusion data available for {modality} model. Skipping confusion matrix plot.")
        return
        
    top_confused_indices = valid_indices[np.argsort(class_error_totals[valid_indices])[-k:]]
    
    # Extract the relevant subset of the confusion matrix
    cm_subset = cm[top_confused_indices][:, top_confused_indices]
    names_subset = [class_names[i] for i in top_confused_indices]
    
    # Convert to integer if values are floats
    if np.issubdtype(cm_subset.dtype, np.floating):
        cm_subset = cm_subset.astype(int)
        fmt = "d"  # Integer format
    else:
        fmt = ".2f"  # Float format with 2 decimal places
    
    # Create the figure
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_subset, annot=True, fmt=fmt, cmap="YlGnBu",
                xticklabels=names_subset, yticklabels=names_subset)
    plt.title(f'Top {k} Most Confused Classes - {modality} Model')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'top{k}_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_per_class_accuracy_plot(accuracies, class_names, sample_counts, modality, output_dir, k=15, min_samples=5):
    """Create and save bar plots for top and bottom k classes by accuracy"""
    # Filter classes with minimum number of samples
    valid_indices = np.where(sample_counts >= min_samples)[0]
    valid_accuracies = accuracies[valid_indices]
    valid_names = [class_names[i] for i in valid_indices]
    valid_counts = sample_counts[valid_indices]
    
    # Create DataFrame with all necessary info
    df = pd.DataFrame({
        'class': valid_names, 
        'accuracy': valid_accuracies,
        'samples': valid_counts
    })
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy')
    
    # Plot bottom k classes (worst performing)
    k_bottom = min(k, len(df_sorted))
    if k_bottom > 0:
        plt.figure(figsize=(14, k_bottom * 0.6))
        bottom_k = df_sorted.head(k_bottom)
        sns.barplot(x='accuracy', y='class', data=bottom_k, palette="YlOrRd_r")
        plt.title(f'Bottom {k_bottom} Classes by Accuracy - {modality} Model')
        plt.xlabel('Accuracy')
        plt.xlim(0, 1.0)
        
        # Add sample count annotations
        for i, row in enumerate(bottom_k.itertuples()):
            plt.text(0.01, i, f"n={int(row.samples)}", va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bottom{k_bottom}_accuracies.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot top k classes (best performing)
    k_top = min(k, len(df_sorted))
    if k_top > 0:
        plt.figure(figsize=(14, k_top * 0.6))
        top_k = df_sorted.tail(k_top).iloc[::-1]  # Reverse to show highest on top
        sns.barplot(x='accuracy', y='class', data=top_k, palette="YlGn")
        plt.title(f'Top {k_top} Classes by Accuracy - {modality} Model')
        plt.xlabel('Accuracy')
        plt.xlim(0, 1.0)
        
        # Add sample count annotations
        for i, row in enumerate(top_k.itertuples()):
            plt.text(0.01, i, f"n={int(row.samples)}", va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top{k_top}_accuracies.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_most_misclassified_pairs_plot(cm, class_names, modality, output_dir, k=15):
    """Identify and visualize the most common misclassification pairs"""
    # Copy the confusion matrix to avoid modifying the original
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Remove correct classifications
    
    # Check if there are enough non-zero values
    nonzero_vals = np.sum(cm_copy > 0)
    if nonzero_vals == 0:
        print(f"No misclassifications found for {modality}")
        return
    
    # Flatten and find indices of top k confused pairs
    flat_indices = np.argsort(cm_copy.flatten())[-min(k, nonzero_vals):]
    row_indices, col_indices = np.unravel_index(flat_indices, cm_copy.shape)
    
    # Create DataFrame for the pairs
    pairs = []
    for i in range(len(flat_indices)):
        true_class = class_names[row_indices[i]]
        pred_class = class_names[col_indices[i]]
        count = cm_copy[row_indices[i], col_indices[i]]
        if count > 0:  # Only add non-zero counts
            pairs.append({
                'True Class': true_class,
                'Predicted Class': pred_class,
                'Count': count
            })
    
    if not pairs:
        print(f"No valid misclassification pairs found for {modality}")
        return
        
    df_pairs = pd.DataFrame(pairs)
    
    # Plot
    plt.figure(figsize=(16, min(12, len(pairs) * 0.6)))
    sns.barplot(x='Count', y='True Class', hue='Predicted Class', data=df_pairs)
    plt.title(f'Top {len(pairs)} Misclassification Pairs - {modality} Model')
    plt.xlabel('Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top{len(pairs)}_misclassification_pairs.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def compute_accuracy_supervised_sigmoid(model_pred, labels):
    """
    Compute precision and recall for multi-label classification
    
    Args:
        model_pred: Model predictions after sigmoid [BxT, C]
        labels: Ground truth labels [BxT, C]
        
    Returns:
        precision, recall
    """
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    
    if pred_one_num == 0:
        return torch.zeros(1).cuda(), torch.zeros(1).cuda()
    
    target_one_num = torch.sum(labels)
    if target_one_num == 0:
        return torch.zeros(1).cuda(), torch.zeros(1).cuda()
    
    true_predict_num = torch.sum(pred_result * labels)
    
    precision = true_predict_num / pred_one_num
    recall = true_predict_num / target_one_num
    
    return precision, recall

def test_model(args, model_path, test_loader, categories, model_type, device, logger):
    """Test a model and generate visualizations"""
    # Create output directories
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    stats_dir = os.path.join(args.output_dir, 'statistics')
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    logger.info(f"Testing {model_type} model from {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path)
    
    # Initialize model components
    video_dim = 512
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    text_output_dim = 256
    audio_output_dim = 256
    n_embeddings = 400
    embedding_dim = 256
    num_classes = len(categories)
    
    # Create encoder and decoder
    encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, 
                               audio_output_dim, video_output_dim, text_output_dim, 
                               n_embeddings, embedding_dim)
    decoder = Semantic_Decoder_AVVP(input_dim=embedding_dim, class_num=26)
    
    # Load weights
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    decoder.load_state_dict(checkpoint['Decoder_parameters'])
    
    # Move to device and set to evaluation mode
    encoder = encoder.double().to(device)
    decoder = decoder.double().to(device)
    encoder.eval()
    decoder.eval()
    
    # Initialize metrics
    all_video_ids = []
    all_precisions = []
    all_recalls = []
    sample_losses = []
    
    # Track per-class metrics
    class_true_positive = np.zeros(num_classes)
    class_false_positive = np.zeros(num_classes)
    class_false_negative = np.zeros(num_classes)
    
    # Sigmoid for converting logits to probabilities
    sigmoid = nn.Sigmoid()
    
    # Test loop
    logger.info(f"Starting evaluation of {model_type} model...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc=f"Testing {model_type}")):
            # feature = batch_data['feature'].double().to(device)
            # label = batch_data['label'].double().to(device)
            # video_id = batch_data['video_id'][0]  # Get the first (and only) video_id
            # category = batch_data['category'][0]  # Get the first (and only) category

            feature, label, video_id = batch_data
            feature = feature.to(torch.float64)
            feature.cuda()

            label = label.double().cuda().to(device)
            B, T, C = label.size()
            label_flat = label.reshape(-1, C)# [B, T, C] -> [BxT, C] e.g.[80*10, 26]
            label_flat = label_flat.to(torch.float32)
            bs = feature.size(0)

            
            # Process according to model type
            if model_type == 'A2V':

                video_vq = encoder.Video_VQ_Encoder(feature)

                B, T, e_dim = video_vq.size()
                video_vq_flat = video_vq.reshape(-1, e_dim)

                logits = decoder(video_vq_flat)

                # logits = logits.reshape(B, T, -1)

            else:  # V2A
               

                audio_vq = encoder.Audio_VQ_Encoder(feature)

                B, T, e_dim = audio_vq.size()
                audio_vq_flat = audio_vq.reshape(-1, e_dim)

                logits = decoder(audio_vq_flat)

                # logits = logits.reshape(B, T, -1)
            
            # Apply sigmoid to get probabilities
            probs = sigmoid(logits)
            
            # Compute precision and recall
            precision, recall = compute_accuracy_supervised_sigmoid(
                probs, 
                label_flat.float().to(device)
            )
            
            if precision.item() > 0 or recall.item() > 0:
                all_precisions.append(precision.item())
                all_recalls.append(recall.item())
            
            # Store video ID
            all_video_ids.append(video_id)
            
            # Get predictions for per-class metrics
            preds_binary = (probs > 0.5).float()

            probs_reshaped = probs.reshape(B, T, -1)
            preds_binary_reshaped = preds_binary.reshape(B, T, -1)
            
            # Track per-class metrics (excluding background class)
            for b in range(B):
                for c in range(num_classes-1):  # Exclude background class
                    # For this class across all time steps in this sample
                    class_preds = preds_binary_reshaped[b, :, c]
                    class_labels = label[b, :, c]
                    
                    # True positives: predicted 1 and ground truth is 1
                    tp = torch.sum((class_preds == 1) & (class_labels == 1)).item()
                    # False positives: predicted 1 but ground truth is 0
                    fp = torch.sum((class_preds == 1) & (class_labels == 0)).item()
                    # False negatives: predicted 0 but ground truth is 1
                    fn = torch.sum((class_preds == 0) & (class_labels == 1)).item()
                    
                    # Update class metrics
                    class_true_positive[c] += tp
                    class_false_positive[c] += fp
                    class_false_negative[c] += fn
                    
                    # Store per-sample, per-class metrics if there's something to record
                    if tp + fp + fn > 0:
                        sample_vid = video_id[b] if isinstance(video_id, list) or isinstance(video_id, tuple) else video_id
                        sample_losses.append({
                            'video_id': sample_vid,
                            'class_idx': c,
                            'category': categories[c],
                            'true_positive': tp,
                            'false_positive': fp,
                            'false_negative': fn,
                            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
                        })
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * args.batch_size} samples")
    
    # Calculate per-class metrics
    class_precision = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    
    for c in range(num_classes-1):  # Exclude background class
        if class_true_positive[c] + class_false_positive[c] > 0:
            class_precision[c] = class_true_positive[c] / (class_true_positive[c] + class_false_positive[c])
        if class_true_positive[c] + class_false_negative[c] > 0:
            class_recall[c] = class_true_positive[c] / (class_true_positive[c] + class_false_negative[c])
        if class_precision[c] + class_recall[c] > 0:
            class_f1[c] = 2 * class_precision[c] * class_recall[c] / (class_precision[c] + class_recall[c])
    
    # Calculate overall metrics
    if all_precisions and all_recalls:
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        f1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    else:
        avg_precision = 0
        avg_recall = 0
        f1_score = 0
    
    logger.info(f"{model_type} Overall Precision: {avg_precision:.4f}")
    logger.info(f"{model_type} Overall Recall: {avg_recall:.4f}")
    logger.info(f"{model_type} Overall F1 Score: {f1_score:.4f}")
    
    # Create confusion matrix for visualization
    cm = np.zeros((num_classes-1, num_classes-1))
    
    # Fill confusion matrix based on true positives and confusions
    for c in range(num_classes-1):
        cm[c, c] = class_true_positive[c]  # Diagonal shows true positives
        for c2 in range(num_classes-1):
            if c != c2:
                # Count how often classes are confused with each other
                confusions = 0
                for loss_item in sample_losses:
                    if loss_item['class_idx'] == c and loss_item['false_negative'] > 0:
                        for loss_item2 in sample_losses:
                            if loss_item2['video_id'] == loss_item['video_id'] and loss_item2['class_idx'] == c2 and loss_item2['false_positive'] > 0:
                                confusions += 1
                cm[c, c2] = confusions
    
    # Generate class statistics
    class_stats = []
    for i in range(num_classes-1):
        stat = {
            'category': categories[i],
            'support': int(class_true_positive[i] + class_false_negative[i]),
            'precision': class_precision[i],
            'recall': class_recall[i],
            'f1-score': class_f1[i],
            'true_positive': int(class_true_positive[i]),
            'false_positive': int(class_false_positive[i]),
            'false_negative': int(class_false_negative[i])
        }
        class_stats.append(stat)
    
    # Save statistics
    class_stats_df = pd.DataFrame(class_stats)
    class_stats_df.to_csv(os.path.join(stats_dir, 'class_statistics.csv'), index=False)
    
    sample_df = pd.DataFrame(sample_losses)
    if len(sample_df) > 0:
        sample_df.to_csv(os.path.join(stats_dir, 'sample_metrics.csv'), index=False)
    
    # Create visualizations
    logger.info(f"Generating visualizations for {model_type} model...")
    
    # Only create visualizations if we have data
    if np.sum(cm) > 0:
        # 1. Confusion matrix
        create_confusion_matrix_plot(cm, categories[:-1], model_type, vis_dir)
        
        # 2. Per-class precision plot
        create_per_class_accuracy_plot(
            class_precision[:-1], 
            categories[:-1], 
            np.array([stat['support'] for stat in class_stats]), 
            f"{model_type}_precision", 
            vis_dir
        )
        
        # 3. Per-class recall plot
        create_per_class_accuracy_plot(
            class_recall[:-1], 
            categories[:-1], 
            np.array([stat['support'] for stat in class_stats]), 
            f"{model_type}_recall", 
            vis_dir
        )
        
        # 4. Per-class F1 plot
        create_per_class_accuracy_plot(
            class_f1[:-1], 
            categories[:-1], 
            np.array([stat['support'] for stat in class_stats]), 
            f"{model_type}_f1", 
            vis_dir
        )
        
        # 5. Misclassification pairs
        create_most_misclassified_pairs_plot(cm, categories[:-1], model_type, vis_dir)
    else:
        logger.warning(f"No data for visualizations in {model_type} model")
    
    # Create summary
    summary = {
        'precision': float(avg_precision),
        'recall': float(avg_recall),
        'f1_score': float(f1_score),
        'total_samples': len(all_video_ids),
        'total_classes': num_classes-1,  # Exclude background class
        'best_class_precision': categories[np.argmax(class_precision[:-1])] if np.max(class_precision[:-1]) > 0 else "None",
        'best_precision': float(np.max(class_precision[:-1])),
        'best_class_recall': categories[np.argmax(class_recall[:-1])] if np.max(class_recall[:-1]) > 0 else "None",
        'best_recall': float(np.max(class_recall[:-1])),
        'best_class_f1': categories[np.argmax(class_f1[:-1])] if np.max(class_f1[:-1]) > 0 else "None",
        'best_f1': float(np.max(class_f1[:-1]))
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"{model_type} testing complete. Results saved to {args.output_dir}")
    
    return avg_precision, avg_recall, f1_score
 

def main():
    """Main function for testing AVVP models"""
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting testing with arguments: {args}")
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Determine modality based on model type
    modality = 'audio' if args.model_type == 'V2A' else 'video'
    
    # Load test dataset
    logger.info(f"Loading AVVP test dataset from {args.meta_csv_path}")
    test_dataset = AVVPDataset(args.meta_csv_path, args.fea_path, split='test', modality= modality)

    
    # Create dataloader with batch_size=1 to handle variable length sequences
    test_dataloader = DataLoader(
        AVVPDataset(args.meta_csv_path, args.fea_path, split='test', modality= modality),
        batch_size=args.batch_size,  # Use batch_size=1 to handle variable length sequences
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples")
    
    # Get categories and add background class
    categories = test_dataset.get_categories()
    categories.append("background")
    logger.info(f"Found {len(categories)} categories in the AVVP dataset (including background)")
    
    # Test the model
    precision, recall, f1_score = test_model(
        args, 
        args.model_path, 
        test_dataloader, 
        categories, 
        args.model_type, 
        device, 
        logger
    )
    
    logger.info(f"Testing complete! {args.model_type} results:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1_score:.4f}")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
