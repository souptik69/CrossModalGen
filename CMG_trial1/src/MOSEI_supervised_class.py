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
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser
from model.main_model_mosei import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# CHANGE 1: Import the new label conversion functions for classification
from label_conversion_mosei import (
    continuous_to_discrete_sentiment,
    predictions_to_continuous,
    discrete_to_continuous_sentiment
)

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =================================  evaluation metrics ============================
def multiclass_acc(preds, truths):
    """Compute the multiclass accuracy w.r.t. groundtruth."""
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

def weighted_accuracy(test_preds_emo, test_truth_emo):
    """Compute multiclass accuracy weighted by class occurence."""
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))
    return (tp * (n/p) + tn) / (2*n)

def eval_mosei_senti_return(results, truths, exclude_zero=False):
    """Evaluate MOSEI and return metric list."""
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    # Average L1 distance between preds and truths
    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    return mae, corr, mult_a7, mult_a5, f_score, accuracy_score(binary_truth, binary_preds)

# CHANGE 2: Add new hybrid evaluation functions for classification
def eval_classification_metrics(logits, discrete_labels):
    """
    Evaluate pure classification metrics.
    
    Args:
        logits: Tensor of shape [batch_size, 7] - model predictions
        discrete_labels: Tensor of shape [batch_size] - ground truth class indices
    
    Returns:
        Dictionary with classification metrics
    """
    # Convert logits to predicted classes
    predicted_classes = torch.argmax(logits, dim=1)
    
    # Convert to numpy for sklearn metrics
    pred_np = predicted_classes.cpu().numpy()
    true_np = discrete_labels.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(true_np, pred_np)
    f1_weighted = f1_score(true_np, pred_np, average='weighted')
    f1_macro = f1_score(true_np, pred_np, average='macro')
    
    return {
        'classification_accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'predicted_classes': pred_np,
        'true_classes': true_np
    }

def eval_sentiment_hybrid(logits, continuous_labels, discrete_labels=None, exclude_zero=False):
    """
    Comprehensive evaluation combining classification and regression metrics.
    
    Args:
        logits: Tensor of shape [batch_size, 7] - model predictions
        continuous_labels: Tensor of shape [batch_size, 1] - original continuous labels
        discrete_labels: Tensor of shape [batch_size] - discrete class labels (optional)
        exclude_zero: Whether to exclude neutral samples from binary classification
    
    Returns:
        Dictionary with all metrics
    """
    batch_size = logits.shape[0]
    
    # If discrete labels not provided, compute them from continuous labels
    if discrete_labels is None:
        discrete_labels = continuous_to_discrete_sentiment(continuous_labels)
    
    # 1. Pure Classification Metrics
    classification_metrics = eval_classification_metrics(logits, discrete_labels)
    
    # 2. Convert predictions back to continuous for regression metrics
    continuous_preds = predictions_to_continuous(logits)  # Shape: [batch_size, 1]
    
    # 3. Compute your existing regression metrics
    regression_metrics = eval_mosei_senti_return(continuous_preds, continuous_labels, exclude_zero)
    mae, corr, mult_a7, mult_a5, f_score, binary_acc = regression_metrics
    
    # 4. Additional analysis: per-class accuracy
    predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
    true_classes = discrete_labels.cpu().numpy()
    
    # Calculate per-class accuracy
    per_class_acc = {}
    class_names = ['Highly Neg', 'Negative', 'Weakly Neg', 'Neutral', 
                   'Weakly Pos', 'Positive', 'Highly Pos']
    
    for class_idx in range(7):
        mask = (true_classes == class_idx)
        if mask.sum() > 0:  # If this class exists in the batch
            class_acc = (predicted_classes[mask] == class_idx).mean()
            per_class_acc[f'{class_names[class_idx]}_acc'] = class_acc
        else:
            per_class_acc[f'{class_names[class_idx]}_acc'] = 0.0
    
    # Combine all metrics
    all_metrics = {
        # Classification metrics
        'classification_accuracy': classification_metrics['classification_accuracy'],
        'f1_weighted': classification_metrics['f1_weighted'],
        'f1_macro': classification_metrics['f1_macro'],
        
        # Regression-style metrics (computed from continuous predictions)
        'mae': mae,
        'correlation': corr,
        'mult_acc_7': mult_a7,
        'mult_acc_5': mult_a5,
        'f1_score_binary': f_score,
        'binary_accuracy': binary_acc,
        
        # Per-class accuracies
        **per_class_acc,
        
        # Raw predictions for further analysis
        'continuous_predictions': continuous_preds,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes
    }
    
    return all_metrics

def print_comprehensive_results(all_metrics, modality_name=""):
    """
    Print both classification and regression-style results in a clear format.
    """
    logger.info("=" * 80)
    logger.info(f"COMPREHENSIVE EVALUATION RESULTS - {modality_name}")
    logger.info("=" * 80)
    
    logger.info("CLASSIFICATION METRICS:")
    logger.info(f"  7-Class Accuracy: {all_metrics['classification_accuracy']:.4f}")
    logger.info(f"  F1-Score (Weighted): {all_metrics['f1_weighted']:.4f}")
    logger.info(f"  F1-Score (Macro): {all_metrics['f1_macro']:.4f}")
    
    logger.info("\nREGRESSION-STYLE METRICS (from continuous predictions):")
    logger.info(f"  MAE: {all_metrics['mae']:.4f}")
    logger.info(f"  Correlation: {all_metrics['correlation']:.4f}")
    logger.info(f"  Multi-class Acc (7-level): {all_metrics['mult_acc_7']:.4f}")
    logger.info(f"  Multi-class Acc (5-level): {all_metrics['mult_acc_5']:.4f}")
    logger.info(f"  Binary F1-Score: {all_metrics['f1_score_binary']:.4f}")
    logger.info(f"  Binary Accuracy: {all_metrics['binary_accuracy']:.4f}")
    
    logger.info("\nPER-CLASS ACCURACIES:")
    class_names = ['Highly Neg', 'Negative', 'Weakly Neg', 'Neutral', 
                   'Weakly Pos', 'Positive', 'Highly Pos']
    for i, class_name in enumerate(class_names):
        acc_key = f'{class_name}_acc'
        if acc_key in all_metrics:
            logger.info(f"  {class_name}: {all_metrics[acc_key]:.4f}")
    
    logger.info("=" * 80)

def eval_mosei_senti_print(results, truths, modality_name="", exclude_zero=False):
    """Print out MOSEI metrics given results and ground truth."""
    mae, corr, mult_a7, mult_a5, f_score, binary_acc = eval_mosei_senti_return(results, truths, exclude_zero)
    
    logger.info("=" * 50)
    logger.info(f"MOSEI Sentiment Evaluation Results - {modality_name}:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Correlation Coefficient: {corr:.4f}")
    logger.info(f"mult_acc_7 (7-class): {mult_a7:.4f}")
    logger.info(f"mult_acc_5 (5-class): {mult_a5:.4f}")
    logger.info(f"F1 score: {f_score:.4f}")
    logger.info(f"Binary Accuracy (2-class): {binary_acc:.4f}")
    logger.info("=" * 50)
    
    return mae, corr, mult_a7, mult_a5, f_score, binary_acc

def main():
    global args, logger, dataset_configs

    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    
    # select GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    logger = Prepare_logger(args, eval=True)  # Set eval=True since this is testing only
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))

    '''dataset selection'''
    if args.dataset_name == 'mosei':
        from dataset.MOSEI_MOSI import get_mosei_supervised_dataloaders
    else:
        raise NotImplementedError

    train_dataloader, val_loader, test_dataloader = get_mosei_supervised_dataloaders(batch_size=args.batch_size, max_seq_len=10, num_workers=8)

    '''model setting'''
    video_dim = 35
    text_dim = 300
    audio_dim = 74
    text_lstm_dim = 128
    n_embeddings = 400
    embedding_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, n_embeddings, embedding_dim)
    # CHANGE 3: Add num_classes=7 parameter to Decoder for classification heads
    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, num_classes=7)

    Text_ar_lstm.double()
    Encoder.double()
    Decoder.double()

    '''Load trained supervised model'''
    Text_ar_lstm.to(device)
    Encoder.to(device)
    Decoder.to(device)

    # Load supervised pretrained model
    path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models/mosei_supervised_AV/checkpoint/MOSEI-model-9.pt"
    logger.info(f"Loading supervised model from: {path_checkpoints}")
    checkpoints = torch.load(path_checkpoints)
    Encoder.load_state_dict(checkpoints['Encoder_parameters'])
    Decoder.load_state_dict(checkpoints['Decoder_parameters'])
    Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
    start_epoch = checkpoints['epoch']
    logger.info("Loaded supervised model from epoch {}".format(start_epoch))

    '''Testing'''
    logger.info(f"Starting testing in {args.test_mode} mode...")
    
    if args.test_mode == 'MSR':
        test_msr_mode(Encoder, Text_ar_lstm, Decoder, test_dataloader, args)
    else:  # CMG mode
        test_cmg_mode(Encoder, Text_ar_lstm, Decoder, test_dataloader, args)

@torch.no_grad()
def test_msr_mode(Encoder, Text_ar_lstm, Decoder, test_dataloader, args):
    """Test Multimodal Sentiment Regression (MSR) mode"""
    logger.info("=" * 80)
    logger.info("TESTING MSR MODE - MULTIMODAL SENTIMENT CLASSIFICATION")  # CHANGE 4: Updated description
    logger.info("=" * 80)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Text_ar_lstm.eval()
    Decoder.eval()
    Encoder.cuda()
    Text_ar_lstm.cuda()
    Decoder.cuda()

    # CHANGE 5: Updated collections for classification evaluation
    all_logits = []           # Store logits instead of direct predictions
    all_continuous_labels = []  # Original continuous labels
    all_discrete_labels = []    # Discrete class labels
    
    # CHANGE 6: Use CrossEntropyLoss instead of MSELoss
    criterion_sentiment = nn.CrossEntropyLoss().cuda()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # Feed input to model
        text_feature_raw, audio_feature, video_feature, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        labels = labels.double().cuda()

        # CHANGE 7: Convert continuous labels to discrete for loss computation
        discrete_labels = continuous_to_discrete_sentiment(labels)

        batch_dim = text_feature_raw.size()[0]
        hidden_dim = 128
        num_layers = 2
        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                      torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(text_feature_raw, text_hidden)

        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        # Get VQ representations
        out_vq_audio, audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
        out_vq_video, video_vq = Encoder.Video_VQ_Encoder(video_feature)
        out_vq_text, text_vq = Encoder.Text_VQ_Encoder(text_feature)

        # CHANGE 8: Get logits instead of continuous scores
        combined_logits = Decoder.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)
        # CHANGE 9: Use discrete labels with CrossEntropyLoss
        loss = criterion_sentiment(combined_logits, discrete_labels.long().cuda())

        # CHANGE 10: Store logits and both label types for hybrid evaluation
        all_logits.append(combined_logits)
        all_continuous_labels.append(labels)
        all_discrete_labels.append(discrete_labels)
        
        losses.update(loss.item(), batch_dim)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if n_iter % 50 == 0:
            logger.info(f'Test: [{n_iter}/{len(test_dataloader)}] Loss: {loss.item():.4f}')

    # CHANGE 11: Concatenate all predictions and labels
    all_logits = torch.cat(all_logits, dim=0)
    all_continuous_labels = torch.cat(all_continuous_labels, dim=0)
    all_discrete_labels = torch.cat(all_discrete_labels, dim=0)
    
    # CHANGE 12: Use hybrid evaluation instead of simple regression metrics
    logger.info(f"MSR Mode - Multimodal Classification Results on MOSEI Test Set:")
    metrics = eval_sentiment_hybrid(all_logits, all_continuous_labels, all_discrete_labels)
    print_comprehensive_results(metrics, "Multimodal")
    
    # CHANGE 13: Save comprehensive results including both classification and regression metrics
    results_dict = {
        'mode': 'MSR',
        'modality': 'multimodal',
        'classification_accuracy': metrics['classification_accuracy'],
        'f1_weighted': metrics['f1_weighted'],
        'f1_macro': metrics['f1_macro'],
        'mae': metrics['mae'],
        'correlation': metrics['correlation'],
        'acc_7': metrics['mult_acc_7'],
        'acc_5': metrics['mult_acc_5'],
        'f1_score': metrics['f1_score_binary'],
        'binary_acc': metrics['binary_accuracy'],
        'avg_loss': losses.avg
    }
    
    results_path = os.path.join(args.snapshot_pref, f"results_MSR_multimodal_classification.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Results saved to {results_path}")

@torch.no_grad()
def test_cmg_mode(Encoder, Text_ar_lstm, Decoder, test_dataloader, args):
    """Test Cross-Modal Generalization (CMG) mode"""
    logger.info("=" * 80)
    logger.info("TESTING CMG MODE - CROSS-MODAL GENERALIZATION WITH CLASSIFICATION")  # CHANGE 14: Updated description
    logger.info("=" * 80)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Text_ar_lstm.eval()
    Decoder.eval()
    Encoder.cuda()
    Text_ar_lstm.cuda()
    Decoder.cuda()

    # CHANGE 15: Updated collections for classification evaluation
    all_audio_logits = []
    all_video_logits = []
    all_text_logits = []
    all_continuous_labels = []
    all_discrete_labels = []
    
    # CHANGE 16: Use CrossEntropyLoss instead of MSELoss
    criterion_sentiment = nn.CrossEntropyLoss().cuda()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # Feed input to model
        text_feature_raw, audio_feature, video_feature, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        labels = labels.double().cuda()

        # CHANGE 17: Convert continuous labels to discrete
        discrete_labels = continuous_to_discrete_sentiment(labels)

        batch_dim = text_feature_raw.size()[0]
        hidden_dim = 128
        num_layers = 2
        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                      torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(text_feature_raw, text_hidden)

        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        # Get VQ representations
        out_vq_audio, audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
        out_vq_video, video_vq = Encoder.Video_VQ_Encoder(video_feature)
        out_vq_text, text_vq = Encoder.Text_VQ_Encoder(text_feature)

        # CHANGE 18: Get logits for all modalities instead of continuous scores
        audio_logits = Decoder.audio_sentiment_decoder(out_vq_audio)
        video_logits = Decoder.video_sentiment_decoder(out_vq_video)
        text_logits = Decoder.text_sentiment_decoder(out_vq_text)

        # CHANGE 19: Store logits and labels for hybrid evaluation
        all_audio_logits.append(audio_logits)
        all_video_logits.append(video_logits)
        all_text_logits.append(text_logits)
        all_continuous_labels.append(labels)
        all_discrete_labels.append(discrete_labels)
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if n_iter % 50 == 0:
            logger.info(f'Test: [{n_iter}/{len(test_dataloader)}]')

    # CHANGE 20: Concatenate all predictions and labels
    all_audio_logits = torch.cat(all_audio_logits, dim=0)
    all_video_logits = torch.cat(all_video_logits, dim=0)
    all_text_logits = torch.cat(all_text_logits, dim=0)
    all_continuous_labels = torch.cat(all_continuous_labels, dim=0)
    all_discrete_labels = torch.cat(all_discrete_labels, dim=0)
    
    # CHANGE 21: Use hybrid evaluation for each modality
    logger.info("CMG Mode - Individual Modality Classification Results on MOSEI Test Set:")
    
    # Audio results
    logger.info("Evaluating Audio Modality:")
    metrics_a = eval_sentiment_hybrid(all_audio_logits, all_continuous_labels, all_discrete_labels)
    print_comprehensive_results(metrics_a, "Audio")
    
    # Video results
    logger.info("Evaluating Video Modality:")
    metrics_v = eval_sentiment_hybrid(all_video_logits, all_continuous_labels, all_discrete_labels)
    print_comprehensive_results(metrics_v, "Video")
    
    # Text results
    logger.info("Evaluating Text Modality:")
    metrics_t = eval_sentiment_hybrid(all_text_logits, all_continuous_labels, all_discrete_labels)
    print_comprehensive_results(metrics_t, "Text")
    
    # CHANGE 22: Save comprehensive results for all modalities
    modalities_results = {
        'audio': {
            'classification_accuracy': metrics_a['classification_accuracy'],
            'f1_weighted': metrics_a['f1_weighted'],
            'f1_macro': metrics_a['f1_macro'],
            'mae': metrics_a['mae'], 
            'correlation': metrics_a['correlation'], 
            'acc_7': metrics_a['mult_acc_7'], 
            'acc_5': metrics_a['mult_acc_5'], 
            'f1_score': metrics_a['f1_score_binary'], 
            'binary_acc': metrics_a['binary_accuracy']
        },
        'video': {
            'classification_accuracy': metrics_v['classification_accuracy'],
            'f1_weighted': metrics_v['f1_weighted'],
            'f1_macro': metrics_v['f1_macro'],
            'mae': metrics_v['mae'], 
            'correlation': metrics_v['correlation'], 
            'acc_7': metrics_v['mult_acc_7'], 
            'acc_5': metrics_v['mult_acc_5'], 
            'f1_score': metrics_v['f1_score_binary'], 
            'binary_acc': metrics_v['binary_accuracy']
        },
        'text': {
            'classification_accuracy': metrics_t['classification_accuracy'],
            'f1_weighted': metrics_t['f1_weighted'],
            'f1_macro': metrics_t['f1_macro'],
            'mae': metrics_t['mae'], 
            'correlation': metrics_t['correlation'], 
            'acc_7': metrics_t['mult_acc_7'], 
            'acc_5': metrics_t['mult_acc_5'], 
            'f1_score': metrics_t['f1_score_binary'], 
            'binary_acc': metrics_t['binary_accuracy']
        }
    }
    
    results_dict = {
        'mode': 'CMG',
        'modalities': modalities_results
    }
    
    results_path = os.path.join(args.snapshot_pref, f"results_CMG_all_modalities_classification.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Results saved to {results_path}")
    
    # CHANGE 23: Updated summary with classification metrics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - CMG MODE CLASSIFICATION RESULTS:")
    logger.info("=" * 80)
    logger.info(f"Audio - ClassAcc: {metrics_a['classification_accuracy']:.4f}, MAE: {metrics_a['mae']:.4f}, Corr: {metrics_a['correlation']:.4f}, F1: {metrics_a['f1_weighted']:.4f}")
    logger.info(f"Video - ClassAcc: {metrics_v['classification_accuracy']:.4f}, MAE: {metrics_v['mae']:.4f}, Corr: {metrics_v['correlation']:.4f}, F1: {metrics_v['f1_weighted']:.4f}")
    logger.info(f"Text  - ClassAcc: {metrics_t['classification_accuracy']:.4f}, MAE: {metrics_t['mae']:.4f}, Corr: {metrics_t['correlation']:.4f}, F1: {metrics_t['f1_weighted']:.4f}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()