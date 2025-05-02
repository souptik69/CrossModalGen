import logging
import os
import time
import random
import json
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import argparse

from model.main_model_2 import AVT_VQVAE_Encoder, Semantic_Decoder
from dataset.AVE_dataset import AVETestDataset  

# Set up argument parser
parser = argparse.ArgumentParser(description='Test A2V or V2A model on AVE dataset')
parser.add_argument('--gpu', default='0', type=str, help='GPU ID to use')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for testing')
parser.add_argument('--output_dir', default='results', type=str, help='Output directory for results')
parser.add_argument('--model_path', default='', type=str, help='Path to model checkpoint')
parser.add_argument('--model_type', default='', type=str, choices=['A2V', 'V2A'], help='Type of model to test (A2V or V2A)')
parser.add_argument('--data_root', default='', type=str, help='Path to AVE dataset')
parser.add_argument('--annotations_path', default='', type=str, help='Path to testSet.txt file')

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

def create_confusion_matrix_plot(cm, class_names, modality, output_dir, k=10):
    """
    Create and save a confusion matrix visualization for top-k confused classes
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        modality: String indicating modality (A2V or V2A)
        output_dir: Directory to save the plot
        k: Number of top confused classes to show
    """
    # Find the most confused classes
    np.fill_diagonal(cm, 0)  # Ignore correct classifications
    
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
    plt.title(f'Top {k} Most Confused Classes - {modality} Model')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'top{k}_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_per_class_accuracy_plot(accuracies, class_names, modality, output_dir, k=10):
    """
    Create and save bar plots for top and bottom k classes by accuracy
    
    Args:
        accuracies: Array of per-class accuracies
        class_names: List of class names
        modality: String indicating modality (A2V or V2A)
        output_dir: Directory to save the plots
        k: Number of top/bottom classes to show
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'class': class_names, 'accuracy': accuracies})
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy')
    
    # Plot bottom k classes (worst performing)
    plt.figure(figsize=(14, 10))
    bottom_k = df_sorted.head(k)
    sns.barplot(x='accuracy', y='class', data=bottom_k, palette="YlOrRd_r")
    plt.title(f'Bottom {k} Classes by Accuracy - {modality} Model')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'bottom{k}_accuracies.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot top k classes (best performing)
    plt.figure(figsize=(14, 10))
    top_k = df_sorted.tail(k).iloc[::-1]  # Reverse to show highest on top
    sns.barplot(x='accuracy', y='class', data=top_k, palette="YlGn")
    plt.title(f'Top {k} Classes by Accuracy - {modality} Model')
    plt.xlabel('Accuracy')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top{k}_accuracies.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_most_misclassified_pairs_plot(cm, class_names, modality, output_dir, k=10):
    """
    Identify and visualize the most common misclassification pairs
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        modality: String indicating modality (A2V or V2A)
        output_dir: Directory to save the plot
        k: Number of top misclassification pairs to show
    """
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
    plt.title(f'Top {k} Misclassification Pairs - {modality} Model')
    plt.xlabel('Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'top{k}_misclassification_pairs.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def test_model(args, model_path, test_loader, categories, model_type, device, logger):
    """
    Test a model and generate visualizations
    
    Args:
        args: Command line arguments
        model_path: Path to model checkpoint
        test_loader: DataLoader for test dataset
        categories: List of category names
        model_type: String indicating model type (A2V or V2A)
        device: Device to run testing on
        logger: Logger object
        
    Returns:
        accuracy: Overall test accuracy
    """
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
    decoder = Semantic_Decoder(input_dim=embedding_dim, class_num=num_classes)
    
    # Load weights
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    decoder.load_state_dict(checkpoint['Decoder_parameters'])
    
    # Move to device and set to evaluation mode
    encoder = encoder.double().to(device)
    decoder = decoder.double().to(device)
    encoder.eval()
    decoder.eval()
    
    # Initialize metrics
    all_preds = []
    all_true_labels = []
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    sample_losses = []
    
    # Loss function
    criterion_event = nn.CrossEntropyLoss(reduction='none').cuda()
    
    # Test loop
    logger.info(f"Starting evaluation of {model_type} model...")
    
    with torch.no_grad():
        for batch_idx, (visual_feat, audio_feat, label) in enumerate(tqdm(test_loader, desc=f"Testing {model_type}")):
            # Move to device
            visual_feat = visual_feat.double().to(device)
            audio_feat = audio_feat.double().to(device)
            label = label.double().to(device)
            
            # Extract true labels (assuming AVE format is the same as in your training code)
            label_foreground = label[:, :, :-1]  # Remove background class
            _, labels_evn = label_foreground.max(-1)
            true_labels, _ = labels_evn.max(-1)  # Get most common class over time
            
            # Process according to model type
            if model_type == 'A2V':
                # Audio to Video model: encode audio, predict class
                # vq = encoder.Audio_VQ_Encoder(audio_feat)
                vq = encoder.Video_VQ_Encoder(visual_feat)
                logits = decoder(vq)
            else:  # V2A
                # Video to Audio model: encode video, predict class
                # vq = encoder.Video_VQ_Encoder(visual_feat)
                vq = encoder.Audio_VQ_Encoder(audio_feat)
                logits = decoder(vq)
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            
            # Calculate losses
            losses = criterion_event(logits, true_labels)
            
            # Update per-class statistics
            for i in range(true_labels.size(0)):
                true_class = true_labels[i].item()
                pred_class = preds[i].item()
                loss_val = losses[i].item()
                
                # Store sample info
                sample_losses.append({
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'loss': loss_val,
                    'category': categories[true_class]
                })
                
                # Update class statistics
                class_total[true_class] += 1
                if pred_class == true_class:
                    class_correct[true_class] += 1
            
            # Store predictions and true labels for overall metrics
            all_preds.extend(preds.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {(batch_idx + 1) * args.batch_size} samples")
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_true_labels, all_preds)
    logger.info(f"{model_type} Overall Accuracy: {accuracy:.4f}")
    
    # Calculate per-class accuracy
    class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i] / class_total[i]
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_preds, labels=range(num_classes))
    
    # Generate classification report
    report = classification_report(all_true_labels, all_preds, 
                                  target_names=categories, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(stats_dir, 'classification_report.csv'))
    
    # Save per-sample losses
    loss_df = pd.DataFrame(sample_losses)
    loss_df.to_csv(os.path.join(stats_dir, 'sample_losses.csv'), index=False)
    
    # Create class statistics
    class_stats = []
    for i in range(num_classes):
        stat = {
            'category': categories[i],
            'total_samples': int(class_total[i]),
            'correct_samples': int(class_correct[i]),
            'accuracy': class_accuracy[i]
        }
        class_stats.append(stat)
    
    class_stats_df = pd.DataFrame(class_stats)
    class_stats_df.to_csv(os.path.join(stats_dir, 'class_statistics.csv'), index=False)
    
    # Create visualizations
    logger.info(f"Generating visualizations for {model_type} model...")
    create_confusion_matrix_plot(cm, categories, model_type, vis_dir)
    create_per_class_accuracy_plot(class_accuracy, categories, model_type, vis_dir)
    create_most_misclassified_pairs_plot(cm, categories, model_type, vis_dir)
    
    # Create analysis of most ambiguous categories
    confusion_sum = np.sum(cm, axis=1) - np.diag(cm)
    confusion_rate = np.zeros(num_classes)
    
    for i in range(num_classes):
        if np.sum(cm[i]) > 0:
            confusion_rate[i] = confusion_sum[i] / np.sum(cm[i])
    
    ambiguous_df = pd.DataFrame({
        'category': categories,
        'confusion_rate': confusion_rate,
        'sample_count': np.sum(cm, axis=1)
    })
    
    ambiguous_df.sort_values('confusion_rate', ascending=False).to_csv(
        os.path.join(stats_dir, 'ambiguous_categories.csv'), index=False)
    
    # Create summary
    summary = {
        'accuracy': float(accuracy),
        'total_samples': len(all_true_labels),
        'total_classes': num_classes,
        'best_class': categories[np.argmax(class_accuracy)],
        'best_accuracy': float(np.max(class_accuracy)),
        'worst_class': categories[np.argmin(class_accuracy)],
        'worst_accuracy': float(np.min(class_accuracy))
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"{model_type} testing complete. Results saved to {args.output_dir}")
    
    return accuracy

def main():
    """Main function to run the testing script"""
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Starting testing with arguments: {args}")
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    logger.info(f"Loading test dataset from {args.data_root}")
    test_dataset = AVETestDataset(args.data_root, args.annotations_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    logger.info(f"Test dataset loaded with {len(test_dataset)} samples and {len(test_dataset.get_categories())} categories")
    
    # Get categories
    categories = test_dataset.get_categories()
    
    # Test the specified model
    accuracy = test_model(args, args.model_path, test_loader, categories, args.model_type, device, logger)
    
    logger.info(f"Testing complete! {args.model_type} accuracy: {accuracy:.4f}")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()