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
from sklearn.metrics import accuracy_score, f1_score

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
    
    # Add custom arguments
    # parser.add_argument('--test_mode', type=str, default='MSR', choices=['MSR', 'CMG'], 
    #                    help='Testing mode: MSR (Multimodal Sentiment Regression) or CMG (Cross-Modal Generalization)')
    # parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'video', 'text'],
    #                    help='Modality for CMG mode training')
    # args = parser.parse_args()
    
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
    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2)

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

    '''loss'''
    criterion_sentiment = nn.MSELoss().cuda()

    if model_resume:
        # Load unsupervised pretrained model
        path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models/mosei_unsupervised_AV/checkpoint/MOSEI-model-9.pt"
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
            val_mae = validate_epoch(Encoder, Text_ar_lstm, Decoder, test_dataloader, criterion_sentiment, epoch, args)
            
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
    
    model_path = os.path.join(args.snapshot_pref, f"MOSI_best_{args.test_mode}_{args.modality if args.test_mode == 'CMG' else 'multimodal'}.pt")
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
            # Multimodal Sentiment Regression
            combined_score = Decoder.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)
            sentiment_loss = criterion_sentiment(combined_score, labels)
            
            loss_items = {
                "combined_sentiment_loss": sentiment_loss.item(),
            }
            loss = sentiment_loss
            
        else:  # CMG mode
            # Cross-Modal Generalization - train on one modality
            if args.modality == 'audio':
                audio_score = Decoder.audio_sentiment_decoder(out_vq_audio)
                sentiment_loss = criterion_sentiment(audio_score, labels)
                loss_items = {
                    "audio_sentiment_loss": sentiment_loss.item(),
                }
            elif args.modality == 'video':
                video_score = Decoder.video_sentiment_decoder(out_vq_video)
                sentiment_loss = criterion_sentiment(video_score, labels)
                loss_items = {
                    "video_sentiment_loss": sentiment_loss.item(),
                }
            else:  # text
                text_score = Decoder.text_sentiment_decoder(out_vq_text)
                sentiment_loss = criterion_sentiment(text_score, labels)
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

    all_preds = []
    all_labels = []
    
    # For CMG mode, collect predictions for other modalities
    if args.test_mode == 'CMG':
        all_preds_cross1 = []
        all_preds_cross2 = []

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        # Feed input to model
        text_feature_raw, audio_feature, video_feature, labels = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['labels']
        text_feature_raw = text_feature_raw.double().cuda()
        labels = labels.double().cuda()

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
            # Test multimodal
            combined_score = Decoder.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)
            loss = criterion_sentiment(combined_score, labels)
            all_preds.append(combined_score)
            
        else:  # CMG mode
            # Test trained modality and cross-modal generalization
            if args.modality == 'audio':
                # Trained on audio, test on video and text
                audio_score = Decoder.audio_sentiment_decoder(out_vq_audio)
                video_score = Decoder.video_sentiment_decoder(out_vq_video)
                text_score = Decoder.text_sentiment_decoder(out_vq_text)
                
                loss = criterion_sentiment(audio_score, labels)
                all_preds.append(audio_score)
                all_preds_cross1.append(video_score)
                all_preds_cross2.append(text_score)
                
            elif args.modality == 'video':
                # Trained on video, test on audio and text
                video_score = Decoder.video_sentiment_decoder(out_vq_video)
                audio_score = Decoder.audio_sentiment_decoder(out_vq_audio)
                text_score = Decoder.text_sentiment_decoder(out_vq_text)
                
                loss = criterion_sentiment(video_score, labels)
                all_preds.append(video_score)
                all_preds_cross1.append(audio_score)
                all_preds_cross2.append(text_score)
                
            else:  # text
                # Trained on text, test on audio and video
                text_score = Decoder.text_sentiment_decoder(out_vq_text)
                audio_score = Decoder.audio_sentiment_decoder(out_vq_audio)
                video_score = Decoder.video_sentiment_decoder(out_vq_video)
                
                loss = criterion_sentiment(text_score, labels)
                all_preds.append(text_score)
                all_preds_cross1.append(audio_score)
                all_preds_cross2.append(video_score)

        all_labels.append(labels)
        losses.update(loss.item(), batch_dim)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    # Evaluate results
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if args.test_mode == 'MSR':
        logger.info(f"MSR Mode - Multimodal Results:")
        mae, corr, mult_a7, mult_a5, f_score, binary_acc = eval_mosi_senti_print(all_preds, all_labels)
    else:  # CMG mode
        # Evaluate trained modality
        logger.info(f"CMG Mode - Trained on {args.modality}:")
        mae, corr, mult_a7, mult_a5, f_score, binary_acc = eval_mosi_senti_print(all_preds, all_labels)
        
        # Evaluate cross-modal generalization
        all_preds_cross1 = torch.cat(all_preds_cross1, dim=0)
        all_preds_cross2 = torch.cat(all_preds_cross2, dim=0)
        
        modalities = ['audio', 'video', 'text']
        other_modalities = [m for m in modalities if m != args.modality]
        
        logger.info(f"CMG Mode - Cross-modal to {other_modalities[0]}:")
        eval_mosi_senti_print(all_preds_cross1, all_labels)
        
        logger.info(f"CMG Mode - Cross-modal to {other_modalities[1]}:")
        eval_mosi_senti_print(all_preds_cross2, all_labels)

    return mae

if __name__ == '__main__':
    main()