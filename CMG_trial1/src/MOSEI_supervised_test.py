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
    # Add custom arguments
    # parser.add_argument('--test_mode', type=str, default='MSR', choices=['MSR', 'CMG'], 
    #                    help='Testing mode: MSR (Multimodal Sentiment Regression) or CMG (Cross-Modal Generalization)')
    # parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'video', 'text'],
    #                    help='Modality for CMG mode testing')
    # args = parser.parse_args()
    
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
    Video_ar_lstm = nn.LSTM(video_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Audio_ar_lstm = nn.LSTM(audio_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Encoder = AVT_VQVAE_Encoder(text_lstm_dim*2, text_lstm_dim*2, text_lstm_dim*2, n_embeddings, embedding_dim)
    Decoder = AVT_VQVAE_Decoder(text_lstm_dim*2, text_lstm_dim*2, text_lstm_dim*2)

    # Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_dim, n_embeddings, embedding_dim)
    # Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_dim)

    Text_ar_lstm.double()
    Video_ar_lstm.double()
    Audio_ar_lstm.double()
    Encoder.double()
    Decoder.double()

    '''Load trained supervised model'''
    Text_ar_lstm.to(device)
    Video_ar_lstm.to(device)
    Audio_ar_lstm.to(device)
    Encoder.to(device)
    Decoder.to(device)

    # Load supervised pretrained model
    path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models2/LSTM_New_100_supervised_reg_TAV/checkpoint/MOSEI-model-99.pt"
    logger.info(f"Loading supervised model from: {path_checkpoints}")
    checkpoints = torch.load(path_checkpoints)
    Encoder.load_state_dict(checkpoints['Encoder_parameters'])
    Decoder.load_state_dict(checkpoints['Decoder_parameters'])
    Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
    Video_ar_lstm.load_state_dict(checkpoints['Video_ar_lstm_parameters'])
    Audio_ar_lstm.load_state_dict(checkpoints['Audio_ar_lstm_parameters'])
    start_epoch = checkpoints['epoch']
    logger.info("Loaded supervised model from epoch {}".format(start_epoch))

    '''Testing'''
    logger.info(f"Starting testing in {args.test_mode} mode...")
    
    if args.test_mode == 'MSR':
        test_msr_mode(Encoder, Text_ar_lstm, Video_ar_lstm, Audio_ar_lstm, Decoder, test_dataloader, args)
    else:  # CMG mode
        test_cmg_mode(Encoder, Text_ar_lstm, Video_ar_lstm, Audio_ar_lstm, Decoder, test_dataloader, args)

@torch.no_grad()
def test_msr_mode(Encoder, Text_ar_lstm, Video_ar_lstm, Audio_ar_lstm, Decoder, test_dataloader, args):
    """Test Multimodal Sentiment Regression (MSR) mode"""
    logger.info("=" * 80)
    logger.info("TESTING MSR MODE - MULTIMODAL SENTIMENT REGRESSION")
    logger.info("=" * 80)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Text_ar_lstm.eval()
    Video_ar_lstm.eval()
    Audio_ar_lstm.eval()
    Decoder.eval()
    Encoder.cuda()
    Text_ar_lstm.cuda()
    Video_ar_lstm.cuda()
    Audio_ar_lstm.cuda()
    Decoder.cuda()

    all_preds = []
    all_labels = []
    criterion_sentiment = nn.MSELoss().cuda()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # Feed input to model
        text_feature_raw, audio_feature_raw, video_feature_raw, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        video_feature_raw = video_feature_raw.double().cuda()
        audio_feature_raw = audio_feature_raw.double().cuda()
        labels = labels.double().cuda()

        batch_dim = text_feature_raw.size()[0]
        hidden_dim = 128
        num_layers = 2

        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                      torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(text_feature_raw, text_hidden)

        video_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        video_feature, video_hidden = Video_ar_lstm(video_feature_raw, video_hidden)

        audio_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        audio_feature, audio_hidden = Audio_ar_lstm(audio_feature_raw, audio_hidden)


        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        # Get VQ representations
        out_vq_audio, audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
        out_vq_video, video_vq = Encoder.Video_VQ_Encoder(video_feature)
        out_vq_text, text_vq = Encoder.Text_VQ_Encoder(text_feature)

        # Test multimodal prediction
        combined_score = Decoder.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)
        loss = criterion_sentiment(combined_score, labels)

        all_preds.append(combined_score)
        all_labels.append(labels)
        losses.update(loss.item(), batch_dim)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if n_iter % 50 == 0:
            logger.info(f'Test: [{n_iter}/{len(test_dataloader)}] Loss: {loss.item():.4f}')

    # Evaluate results
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    logger.info(f"MSR Mode - Multimodal Results on MOSEI Test Set:")
    mae, corr, mult_a7, mult_a5, f_score, binary_acc = eval_mosei_senti_print(all_preds, all_labels, "Multimodal")
    
    # Save results
    results_dict = {
        'mode': 'MSR',
        'modality': 'multimodal',
        'mae': mae,
        'correlation': corr,
        'acc_7': mult_a7,
        'acc_5': mult_a5,
        'f1_score': f_score,
        'binary_acc': binary_acc,
        'avg_loss': losses.avg
    }
    
    results_path = os.path.join(args.snapshot_pref, f"results_MSR_multimodal.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Results saved to {results_path}")

@torch.no_grad()
def test_cmg_mode(Encoder, Text_ar_lstm, Video_ar_lstm, Audio_ar_lstm, Decoder, test_dataloader, args):
    """Test Cross-Modal Generalization (CMG) mode"""
    logger.info("=" * 80)
    logger.info("TESTING CMG MODE - CROSS-MODAL GENERALIZATION")
    logger.info("=" * 80)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Text_ar_lstm.eval()
    Video_ar_lstm.eval()
    Audio_ar_lstm.eval()
    Decoder.eval()
    Encoder.cuda()
    Text_ar_lstm.cuda()
    Video_ar_lstm.cuda()
    Audio_ar_lstm.cuda()
    Decoder.cuda()

    # Collections for all three modalities
    all_audio_preds = []
    all_video_preds = []
    all_text_preds = []
    all_labels = []
    criterion_sentiment = nn.MSELoss().cuda()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # Feed input to model
        text_feature_raw, audio_feature_raw, video_feature_raw, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        video_feature_raw = video_feature_raw.double().cuda()
        audio_feature_raw = audio_feature_raw.double().cuda()
        labels = labels.double().cuda()

        batch_dim = text_feature_raw.size()[0]
        hidden_dim = 128
        num_layers = 2

        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                      torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(text_feature_raw, text_hidden)

        video_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        video_feature, video_hidden = Video_ar_lstm(video_feature_raw, video_hidden)

        audio_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        audio_feature, audio_hidden = Audio_ar_lstm(audio_feature_raw, audio_hidden)



        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        # Get VQ representations
        out_vq_audio, audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
        out_vq_video, video_vq = Encoder.Video_VQ_Encoder(video_feature)
        out_vq_text, text_vq = Encoder.Text_VQ_Encoder(text_feature)

        # Test individual modalities
        audio_score = Decoder.audio_sentiment_decoder(out_vq_audio)
        video_score = Decoder.video_sentiment_decoder(out_vq_video)
        text_score = Decoder.text_sentiment_decoder(out_vq_text)

        all_audio_preds.append(audio_score)
        all_video_preds.append(video_score)
        all_text_preds.append(text_score)
        all_labels.append(labels)
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if n_iter % 50 == 0:
            logger.info(f'Test: [{n_iter}/{len(test_dataloader)}]')

    # Evaluate results for all modalities
    all_audio_preds = torch.cat(all_audio_preds, dim=0)
    all_video_preds = torch.cat(all_video_preds, dim=0)
    all_text_preds = torch.cat(all_text_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Evaluate each modality individually
    logger.info("CMG Mode - Individual Modality Results on MOSEI Test Set:")
    
    # Audio results
    mae_a, corr_a, mult_a7_a, mult_a5_a, f_score_a, binary_acc_a = eval_mosei_senti_print(all_audio_preds, all_labels, "Audio")
    
    # Video results
    mae_v, corr_v, mult_a7_v, mult_a5_v, f_score_v, binary_acc_v = eval_mosei_senti_print(all_video_preds, all_labels, "Video")
    
    # Text results
    mae_t, corr_t, mult_a7_t, mult_a5_t, f_score_t, binary_acc_t = eval_mosei_senti_print(all_text_preds, all_labels, "Text")
    
    # Save results for each modality
    modalities_results = {
        'audio': {
            'mae': mae_a, 'correlation': corr_a, 'acc_7': mult_a7_a, 
            'acc_5': mult_a5_a, 'f1_score': f_score_a, 'binary_acc': binary_acc_a
        },
        'video': {
            'mae': mae_v, 'correlation': corr_v, 'acc_7': mult_a7_v, 
            'acc_5': mult_a5_v, 'f1_score': f_score_v, 'binary_acc': binary_acc_v
        },
        'text': {
            'mae': mae_t, 'correlation': corr_t, 'acc_7': mult_a7_t, 
            'acc_5': mult_a5_t, 'f1_score': f_score_t, 'binary_acc': binary_acc_t
        }
    }
    
    results_dict = {
        'mode': 'CMG',
        'modalities': modalities_results
    }
    
    results_path = os.path.join(args.snapshot_pref, f"results_CMG_all_modalities.json")
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Results saved to {results_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - CMG MODE RESULTS:")
    logger.info("=" * 80)
    logger.info(f"Audio - MAE: {mae_a:.4f}, Acc7: {mult_a7_a:.4f}, Acc5: {mult_a5_a:.4f}, F1: {f_score_a:.4f}, Binary: {binary_acc_a:.4f}")
    logger.info(f"Video - MAE: {mae_v:.4f}, Acc7: {mult_a7_v:.4f}, Acc5: {mult_a5_v:.4f}, F1: {f_score_v:.4f}, Binary: {binary_acc_v:.4f}")
    logger.info(f"Text  - MAE: {mae_t:.4f}, Acc7: {mult_a7_t:.4f}, Acc5: {mult_a5_t:.4f}, F1: {f_score_t:.4f}, Binary: {binary_acc_t:.4f}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()