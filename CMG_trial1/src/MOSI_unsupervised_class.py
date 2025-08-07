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

def eval_mosi_senti_return(results, truths, exclude_zero=False):
    """Evaluate MOSI and return metric list."""
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
    regression_metrics = eval_mosi_senti_return(continuous_preds, continuous_labels, exclude_zero)
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

def eval_mosi_senti_print(results, truths, exclude_zero=False):
    """Print out MOSI metrics given results and ground truth."""
    mae, corr, mult_a7, mult_a5, f_score, binary_acc = eval_mosi_senti_return(results, truths, exclude_zero)
    
    logger.info("=" * 50)
    logger.info("MOSI Sentiment Evaluation Results:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Correlation Coefficient: {corr:.4f}")
    logger.info(f"mult_acc_7: {mult_a7:.4f}")
    logger.info(f"mult_acc_5: {mult_a5:.4f}")
    logger.info(f"F1 score: {f_score:.4f}")
    logger.info(f"Binary Accuracy: {binary_acc:.4f}")
    logger.info("=" * 50)
    
    return mae, corr, mult_a7, mult_a5, f_score, binary_acc

def main():
    global args, logger, dataset_configs
    global best_mae, best_mae_epoch
    best_mae, best_mae_epoch = float('inf'), 0

    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    
    # select GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    logger = Prepare_logger(args, eval=args.evaluate)
    logger.info(f'\nCreating folder: {args.snapshot_pref}')
    logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))

    '''dataset selection'''
    if args.dataset_name == 'mosi':
        from dataset.MOSEI_MOSI import get_mosi_dataloaders
    else:
        raise NotImplementedError

    train_dataloader, val_loader, test_dataloader = get_mosi_dataloaders(batch_size=args.batch_size, max_seq_len=10, num_workers=8)

    '''model setting'''
    video_dim = 35
    text_dim = 300
    audio_dim = 74
    text_lstm_dim = 128
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = True
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, n_embeddings, embedding_dim)
    # CHANGE 3: Add num_classes=7 parameter to Decoder for classification heads
    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, num_classes=7)

    Text_ar_lstm.double()
    Encoder.double()
    Decoder.double()

    '''optimizer setting'''
    Text_ar_lstm.to(device)
    Encoder.to(device)
    Decoder.to(device)
    
    if args.test_mode == 'MSR':
        # For MSR mode, train the combined sentiment decoder
        optimizer = torch.optim.Adam(Decoder.combined_sentiment_decoder.parameters(), lr=args.lr)
    else:  # CMG mode
        # For CMG mode, train individual sentiment decoder based on modality
        if args.modality == 'audio':
            optimizer = torch.optim.Adam(Decoder.audio_sentiment_decoder.parameters(), lr=args.lr)
        elif args.modality == 'video':
            optimizer = torch.optim.Adam(Decoder.video_sentiment_decoder.parameters(), lr=args.lr)
        else:  # text
            optimizer = torch.optim.Adam(Decoder.text_sentiment_decoder.parameters(), lr=args.lr)
    
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

    # CHANGE 4: Use CrossEntropyLoss instead of MSELoss
    criterion_sentiment = nn.CrossEntropyLoss().cuda()

    if model_resume:
        # Load unsupervised pretrained model
        path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models/unsupervised_class_AV/checkpoint/MOSEI-model-9.pt"
        logger.info(f"Loading unsupervised model from: {path_checkpoints}")
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
        start_epoch = checkpoints['epoch']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Training and Evaluation'''
    for epoch in range(start_epoch+1, args.n_epoch):
        loss, total_step = train_epoch(Encoder, Text_ar_lstm, Decoder, train_dataloader, criterion_sentiment,
                                       optimizer, epoch, total_step, args)
        
        # Validation
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            if args.test_mode == 'MSR':
                val_mae = validate_epoch(Encoder, Text_ar_lstm, Decoder, test_dataloader, criterion_sentiment, epoch, args)
            else:
                val_mae = validate_epoch(Encoder, Text_ar_lstm, Decoder, train_dataloader, criterion_sentiment, epoch, args)
            logger.info("-----------------------------")
            logger.info(f"Epoch {epoch} - Training Loss: {loss:.4f}, Validation MAE: {val_mae:.4f}")
            logger.info("-----------------------------")
            
            # Save best model
            if val_mae < best_mae:
                best_mae = val_mae
                best_mae_epoch = epoch
                save_best_model(Encoder, Text_ar_lstm, Decoder, optimizer, epoch, total_step, args)
                
        scheduler.step()

    logger.info(f"Best MAE: {best_mae:.4f} at epoch {best_mae_epoch}")

def _export_log(epoch, total_step, batch_idx, lr, loss_meter):
    msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, batch_idx, lr)
    for k, v in loss_meter.items():
        msg += '{} = {:.4f}, '.format(k, v)
    logger.info(msg)
    sys.stdout.flush()
    loss_meter.update({"batch": total_step})

def to_eval(all_models):
    for m in all_models:
        m.eval()

def to_train(all_models):
    for m in all_models:
        m.train()

def save_best_model(Encoder, Text_ar_lstm, Decoder, optimizer, epoch_num, total_step, args):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'Text_ar_lstm_parameters': Text_ar_lstm.state_dict(),
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step,
        'test_mode': args.test_mode,
        'modality': args.modality if args.test_mode == 'CMG' else 'multimodal'
    }
    
    model_path = os.path.join(args.snapshot_pref, f"MOSI_best_{args.test_mode}_{args.modality if args.test_mode == 'CMG' else 'multimodal'}_classification.pt")
    torch.save(state_dict, model_path)
    logger.info(f'Saved best model to {model_path}')

def train_epoch(Encoder, Text_ar_lstm, Decoder, train_dataloader, criterion_sentiment, optimizer, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()
    
    models = [Decoder]  # Only train decoder, freeze encoder
    to_train(models)
    Encoder.eval()  # Freeze encoder
    Text_ar_lstm.eval()  # Freeze text LSTM
    
    Encoder.cuda()
    Text_ar_lstm.cuda()
    Decoder.cuda()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        
        # Feed input to model
        text_feature_raw, audio_feature, video_feature, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        labels = labels.double().cuda()

        # CHANGE 5: Convert continuous labels to discrete class indices for classification loss
        discrete_labels = continuous_to_discrete_sentiment(labels)

        batch_dim = text_feature_raw.size()[0]
        hidden_dim = 128
        num_layers = 2
        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                      torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        
        with torch.no_grad():
            text_feature, text_hidden = Text_ar_lstm(text_feature_raw, text_hidden)

        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        with torch.no_grad():
            # Get VQ representations
            out_vq_audio, audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
            out_vq_video, video_vq = Encoder.Video_VQ_Encoder(video_feature)
            out_vq_text, text_vq = Encoder.Text_VQ_Encoder(text_feature)

        if args.test_mode == 'MSR':
            # CHANGE 6: Get logits instead of continuous scores for Multimodal Sentiment Classification
            combined_logits = Decoder.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)
            # CHANGE 7: Use discrete labels with CrossEntropyLoss
            sentiment_loss = criterion_sentiment(combined_logits, discrete_labels.long().cuda())
            
            loss_items = {
                "combined_sentiment_loss": sentiment_loss.item(),
            }
            loss = sentiment_loss
            
        else:  # CMG mode
            # CHANGE 8: Get logits for Cross-Modal Generalization - train on one modality
            if args.modality == 'audio':
                audio_logits = Decoder.audio_sentiment_decoder(out_vq_audio)
                # CHANGE 9: Use discrete labels with CrossEntropyLoss
                sentiment_loss = criterion_sentiment(audio_logits, discrete_labels.long().cuda())
                loss_items = {
                    "audio_sentiment_loss": sentiment_loss.item(),
                }
            elif args.modality == 'video':
                video_logits = Decoder.video_sentiment_decoder(out_vq_video)
                # CHANGE 10: Use discrete labels with CrossEntropyLoss
                sentiment_loss = criterion_sentiment(video_logits, discrete_labels.long().cuda())
                loss_items = {
                    "video_sentiment_loss": sentiment_loss.item(),
                }
            else:  # text
                text_logits = Decoder.text_sentiment_decoder(out_vq_text)
                # CHANGE 11: Use discrete labels with CrossEntropyLoss
                sentiment_loss = criterion_sentiment(text_logits, discrete_labels.long().cuda())
                loss_items = {
                    "text_sentiment_loss": sentiment_loss.item(),
                }
            loss = sentiment_loss

        metricsContainer.update("loss", loss_items)

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=optimizer.state_dict()['param_groups'][0]['lr'], loss_meter=metricsContainer.calculate_average("loss"))
        
        loss.backward()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), audio_feature.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    return losses.avg, n_iter + total_step

@torch.no_grad()
def validate_epoch(Encoder, Text_ar_lstm, Decoder, test_dataloader, criterion_sentiment, epoch, args):
    """
    CHANGE 12: Updated validation function that handles classification training and comprehensive evaluation.
    """
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

    # CHANGE 13: Updated collections for classification evaluation
    all_logits = []          # Store logits instead of direct predictions
    all_continuous_labels = []  # Original continuous labels
    all_discrete_labels = []    # Converted discrete labels
    
    # For CMG mode, collect predictions for other modalities
    if args.test_mode == 'CMG':
        all_logits_cross1 = []
        all_logits_cross2 = []

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # Feed input to model
        text_feature_raw, audio_feature, video_feature, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        labels = labels.double().cuda()

        # CHANGE 14: Convert to discrete for loss computation
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

        if args.test_mode == 'MSR':
            # CHANGE 15: Get logits for multimodal classification
            combined_logits = Decoder.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)
            loss = criterion_sentiment(combined_logits, discrete_labels.long().cuda())
            all_logits.append(combined_logits)
            
        else:  # CMG mode
            # CHANGE 16: Get logits for trained modality and cross-modal generalization
            if args.modality == 'audio':
                # Trained on audio, test on video and text
                audio_logits = Decoder.audio_sentiment_decoder(out_vq_audio)
                video_logits = Decoder.video_sentiment_decoder(out_vq_video)
                text_logits = Decoder.text_sentiment_decoder(out_vq_text)
                
                loss = criterion_sentiment(audio_logits, discrete_labels.long().cuda())
                all_logits.append(audio_logits)
                all_logits_cross1.append(video_logits)
                all_logits_cross2.append(text_logits)
                
            elif args.modality == 'video':
                # Trained on video, test on audio and text
                video_logits = Decoder.video_sentiment_decoder(out_vq_video)
                audio_logits = Decoder.audio_sentiment_decoder(out_vq_audio)
                text_logits = Decoder.text_sentiment_decoder(out_vq_text)
                
                loss = criterion_sentiment(video_logits, discrete_labels.long().cuda())
                all_logits.append(video_logits)
                all_logits_cross1.append(audio_logits)
                all_logits_cross2.append(text_logits)
                
            else:  # text
                # Trained on text, test on audio and video
                text_logits = Decoder.text_sentiment_decoder(out_vq_text)
                audio_logits = Decoder.audio_sentiment_decoder(out_vq_audio)
                video_logits = Decoder.video_sentiment_decoder(out_vq_video)
                
                loss = criterion_sentiment(text_logits, discrete_labels.long().cuda())
                all_logits.append(text_logits)
                all_logits_cross1.append(audio_logits)
                all_logits_cross2.append(video_logits)

        # CHANGE 17: Store labels for hybrid evaluation
        all_continuous_labels.append(labels)
        all_discrete_labels.append(discrete_labels)
        
        losses.update(loss.item(), batch_dim)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    # CHANGE 18: Concatenate all predictions and labels for comprehensive evaluation
    all_logits = torch.cat(all_logits, dim=0)
    all_continuous_labels = torch.cat(all_continuous_labels, dim=0)
    all_discrete_labels = torch.cat(all_discrete_labels, dim=0)
    
    if args.test_mode == 'MSR':
        # CHANGE 19: Use hybrid evaluation for MSR mode
        logger.info(f"MSR Mode - Multimodal Classification Results:")
        metrics = eval_sentiment_hybrid(all_logits, all_continuous_labels, all_discrete_labels)
        print_comprehensive_results(metrics, "Multimodal")
        return metrics['mae']  # Return MAE for compatibility with existing code
        
    else:  # CMG mode
        # CHANGE 20: Use hybrid evaluation for trained modality
        logger.info(f"CMG Mode - Trained on {args.modality}:")
        metrics_main = eval_sentiment_hybrid(all_logits, all_continuous_labels, all_discrete_labels)
        print_comprehensive_results(metrics_main, f"Trained on {args.modality}")
        
        # CHANGE 21: Evaluate cross-modal performance with hybrid metrics
        all_logits_cross1 = torch.cat(all_logits_cross1, dim=0)
        all_logits_cross2 = torch.cat(all_logits_cross2, dim=0)
        
        modalities = ['audio', 'video', 'text']
        other_modalities = [m for m in modalities if m != args.modality]
        
        logger.info(f"CMG Mode - Cross-modal to {other_modalities[0]}:")
        metrics_cross1 = eval_sentiment_hybrid(all_logits_cross1, all_continuous_labels, all_discrete_labels)
        print_comprehensive_results(metrics_cross1, f"Cross-modal to {other_modalities[0]}")
        
        logger.info(f"CMG Mode - Cross-modal to {other_modalities[1]}:")
        metrics_cross2 = eval_sentiment_hybrid(all_logits_cross2, all_continuous_labels, all_discrete_labels)
        print_comprehensive_results(metrics_cross2, f"Cross-modal to {other_modalities[1]}")
        
        return metrics_main['mae']  # Return main modality MAE

if __name__ == '__main__':
    main()