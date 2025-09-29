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
from model.main_model_mosei import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
from model.CPC import Cross_CPC, Cross_CPC_AVT_pad
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
from label_conversion_mosei import (
    continuous_to_discrete_sentiment, 
    continuous_to_onehot_sentiment,
    predictions_to_continuous
)
from transformers import BertTokenizer, BertModel
import pickle
from collections import Counter
import csv
import os
torch.autograd.set_detect_anomaly(True)


# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # utils variable
    global args, logger, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''Csv Training loss Iniatialize'''
    csv_headers = ['epoch', 'total_loss', 'audio_recon_loss', 'video_recon_loss', 'text_recon_loss', 
               'audio_embedding_loss', 'video_embedding_loss', 'text_embedding_loss', 
               'cpc_loss', 'cmcm_loss']

    csv_file_path = os.path.join(args.snapshot_pref, args.loss_csv_path)
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
    logger.info(f"Loss CSV will be saved to: {csv_file_path}")

    '''dataset selection'''
    if args.dataset_name == 'mosei':
        from dataset.MOSEI_MOSI import get_mosei_unsupervised_split_dataloaders
    else:
        raise NotImplementedError 

    train_dataloader, test_train_loader, test_val_loader = get_mosei_unsupervised_split_dataloaders(batch_size=args.batch_size, max_seq_len=50, num_workers=8)

    '''model setting'''
    video_dim = 35
    text_dim = 300
    audio_dim = 74
    # text_lstm_dim = 128
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = False
    # model_resume = True
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    # Video_ar_lstm = nn.LSTM(video_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    # Audio_ar_lstm = nn.LSTM(audio_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)

    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_dim, n_embeddings, embedding_dim)
    # Encoder = AVT_VQVAE_Encoder(text_lstm_dim*2, text_lstm_dim*2, text_lstm_dim*2, n_embeddings, embedding_dim)

    CPC = Cross_CPC_AVT_pad(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)

    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_dim, embedding_dim)
    # Decoder = AVT_VQVAE_Decoder(text_lstm_dim*2, text_lstm_dim*2, text_lstm_dim*2)

    # Text_ar_lstm.double()
    # Video_ar_lstm.double()
    # Audio_ar_lstm.double()
    Encoder.double()
    CPC.double()
    Decoder.double()


    '''optimizer setting'''
    # Text_ar_lstm.to(device)
    # Video_ar_lstm.to(device)
    # Audio_ar_lstm.to(device)
    Encoder.to(device)
    CPC.to(device)
    Decoder.to(device)
    # optimizer = torch.optim.Adam(chain(Text_ar_lstm.parameters(),Video_ar_lstm.parameters(),Audio_ar_lstm.parameters(), \
    #                                    Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=args.lr)

    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    

    if model_resume is True:
        path_checkpoints = ""
        print(path_checkpoints)
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        CPC.load_state_dict(checkpoints['CPC_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        # Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
        # Video_ar_lstm.load_state_dict(checkpoints['Video_ar_lstm_parameters'])
        # Audio_ar_lstm.load_state_dict(checkpoints['Audio_ar_lstm_parameters'])
        start_epoch = checkpoints['epoch']
        total_step = checkpoints['total_step']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Training and Evaluation'''
    for epoch in range(start_epoch+1, args.n_epoch):
        # loss, total_step, epoch_losses = train_epoch(CPC, Encoder, Text_ar_lstm, Video_ar_lstm, Audio_ar_lstm, Decoder, train_dataloader,
        #                                optimizer, epoch, total_step, args)

        loss, total_step, epoch_losses = train_epoch(CPC, Encoder, Decoder, train_dataloader,
                                       optimizer, epoch, total_step, args)

        csv_file_path = os.path.join(args.snapshot_pref, args.loss_csv_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch] + epoch_losses)
        
        
        save_path = os.path.join(args.model_save_path, 'MOSEI-model-unsupervised-{}.pt'.format(epoch))
        # save_models(CPC, Encoder, Text_ar_lstm, Video_ar_lstm, Audio_ar_lstm, Decoder, optimizer, epoch, total_step, save_path)
        save_models(CPC, Encoder, Decoder, optimizer, epoch, total_step, save_path)
        logger.info(f"epoch: ******************************************* {epoch}")
        logger.info(f"loss: {loss}")
        scheduler.step()


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


def save_models(CPC, Encoder, Decoder, optimizer, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'CPC_parameters': CPC.state_dict(),
        # 'Text_ar_lstm_parameters': Text_ar_lstm.state_dict(),
        # 'Video_ar_lstm_parameters': Video_ar_lstm.state_dict(),
        # 'Audio_ar_lstm_parameters': Audio_ar_lstm.state_dict(),
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))



def train_epoch(CPC,Encoder, Decoder,train_dataloader, optimizer, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()

    epoch_total_loss = AverageMeter()
    epoch_audio_recon = AverageMeter()
    epoch_video_recon = AverageMeter()
    epoch_text_recon = AverageMeter()
    epoch_audio_embed = AverageMeter()
    epoch_video_embed = AverageMeter()
    epoch_text_embed = AverageMeter()
    epoch_cpc_loss = AverageMeter()
    epoch_cmcm_loss = AverageMeter()


    end_time = time.time()
    # models = [CPC,Encoder,Text_ar_lstm,Video_ar_lstm, Audio_ar_lstm,Decoder]
    models = [CPC, Encoder, Decoder]
    to_train(models)

    Encoder.cuda()
    # Text_ar_lstm.cuda()
    # Video_ar_lstm.cuda()
    # Audio_ar_lstm.cuda()
    Decoder.cuda()
    CPC.cuda()
    optimizer.zero_grad()

    quantizer = Encoder.Cross_quantizer
    with torch.no_grad():
        logger.info(f"INITIALIZATION_1 - Vector 200 (first 5 video dims): {quantizer.embedding[200, :5]}")
        logger.info(f"INITIALIZATION_1 - Vector 200 (first 5 audio dims): {quantizer.embedding[200, 256:261]}")
        logger.info(f"INITIALIZATION_1 - Vector 200 (first 5 text dims): {quantizer.embedding[200, 512:517]}")
        logger.info(f"INITIALIZATION_1 - Vector 120 (first 5 video dims): {quantizer.embedding[120, :5]}")
        logger.info(f"INITIALIZATION_1 - Vector 120 (first 5 audio dims): {quantizer.embedding[120, 256:261]}")
        logger.info(f"INITIALIZATION_1 - Vector 120 (first 5 text dims): {quantizer.embedding[120, 512:517]}")
        logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 video dims): {quantizer.embedding[45, :5]}")
        logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 audio dims): {quantizer.embedding[45, 256:261]}")
        logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 text dims): {quantizer.embedding[45, 512:517]}")
   
        logger.info(f"Init Codebook min: {quantizer.embedding.min().item()}, max: {quantizer.embedding.max().item()}")

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        # text_feature_raw, audio_feature_raw, video_feature_raw = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea']
        # text_feature_raw = text_feature_raw.double().cuda()
        # video_feature_raw = video_feature_raw.double().cuda()
        # audio_feature_raw = audio_feature_raw.double().cuda()

        text_feature, audio_feature, video_feature, attention_mask, lengths = batch_data['text_fea'], batch_data['audio_fea'], batch_data['video_fea'], batch_data['attention_mask'], batch_data['lengths']

        batch_dim = text_feature.size()[0]
        # hidden_dim = 128
        # num_layers = 2
        # text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
        #           torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        # text_feature, text_hidden = Text_ar_lstm(text_feature_raw, text_hidden)

        # video_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
        #           torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        # video_feature, video_hidden = Video_ar_lstm(video_feature_raw, video_hidden)

        # audio_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
        #           torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        # audio_feature, audio_hidden = Audio_ar_lstm(audio_feature_raw, audio_hidden)

        # text_feature = text_feature_raw
        # audio_feature = audio_feature_raw
        # video_feature = video_feature_raw

        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)
        attention_mask = attention_mask.cuda()
        lengths = lengths.cuda()

       

        audio_semantic_result, audio_encoder_result, video_semantic_result, video_encoder_result, \
        text_semantic_result, text_encoder_result, \
        out_vq_video, video_vq, out_vq_audio, audio_vq,\
        out_vq_text, text_vq, video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, equal_num, cmcm_loss, segment_loss \
        = Encoder(audio_feature, video_feature, text_feature, epoch, attention_mask=attention_mask)

        
        if n_iter == 0:  # Log every 10 iterations to match semantic logging frequency
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - RAW FEATURES (Iter {n_iter}) =====")
                logger.info(f"Batch size: {text_feature.size(0)}")
                logger.info(f"Original lengths from dataloader: {lengths}")
                logger.info(f"Audio feature shape: {audio_feature.shape} (timesteps: {audio_feature.size(1)})")
                logger.info(f"Video feature shape: {video_feature.shape} (timesteps: {video_feature.size(1)})")
                logger.info(f"Text feature shape: {text_feature.shape} (timesteps: {text_feature.size(1)})")
                
                # Count effective (non-padded) timesteps using attention mask
                effective_lengths = attention_mask.sum(dim=1)  # Sum along sequence dimension
                logger.info(f"Effective lengths from attention mask: {effective_lengths}")

                # Log attention mask information
                logger.info(f"\n===== ATTENTION MASK INFO (Iter {n_iter}) =====")
                logger.info(f"Attention mask shape: {attention_mask.shape}")
                logger.info(f"Full attention masks for all batch samples:\n{attention_mask}")
                
                # Show last 5 timesteps of attention mask to match feature logging
                attention_mask_last5 = attention_mask[:, -5:]
                logger.info(f"Attention mask (last 5 timesteps, shape {attention_mask_last5.shape}):\n{attention_mask_last5}")
                
                # Show attention mask statistics
                total_positions = attention_mask.numel()
                active_positions = attention_mask.sum().item()
                padding_ratio = 1.0 - (active_positions / total_positions)
                logger.info(f"Attention mask stats - Active positions: {active_positions}/{total_positions} ({padding_ratio:.2%} padding)")

                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                raw_audio_subset = audio_feature[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                raw_video_subset = video_feature[:, -5:, :5]    # All samples, last 5 steps, first 5 features  
                raw_text_subset = text_feature[:, -5:, :5]      # All samples, last 5 steps, first 5 features
        
                logger.info(f"\n===== RAW INPUT FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Raw Audio features (shape {raw_audio_subset.shape}):\n{raw_audio_subset}")
                logger.info(f"Raw Video features (shape {raw_video_subset.shape}):\n{raw_video_subset}")  
                logger.info(f"Raw Text features (shape {raw_text_subset.shape}):\n{raw_text_subset}")
                logger.info("=" * 70)

        if n_iter == 1:  # Log every 10 iterations to match semantic logging frequency
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - RAW FEATURES (Iter {n_iter}) =====")
                logger.info(f"Batch size: {text_feature.size(0)}")
                logger.info(f"Original lengths from dataloader: {lengths}")
                logger.info(f"Audio feature shape: {audio_feature.shape} (timesteps: {audio_feature.size(1)})")
                logger.info(f"Video feature shape: {video_feature.shape} (timesteps: {video_feature.size(1)})")
                logger.info(f"Text feature shape: {text_feature.shape} (timesteps: {text_feature.size(1)})")
                
                # Count effective (non-padded) timesteps using attention mask
                effective_lengths = attention_mask.sum(dim=1)  # Sum along sequence dimension
                logger.info(f"Effective lengths from attention mask: {effective_lengths}")

                # Log attention mask information
                logger.info(f"\n===== ATTENTION MASK INFO (Iter {n_iter}) =====")
                logger.info(f"Attention mask shape: {attention_mask.shape}")
                logger.info(f"Full attention masks for all batch samples:\n{attention_mask}")
                
                # Show last 5 timesteps of attention mask to match feature logging
                attention_mask_last5 = attention_mask[:, -5:]
                logger.info(f"Attention mask (last 5 timesteps, shape {attention_mask_last5.shape}):\n{attention_mask_last5}")
                
                # Show attention mask statistics
                total_positions = attention_mask.numel()
                active_positions = attention_mask.sum().item()
                padding_ratio = 1.0 - (active_positions / total_positions)
                logger.info(f"Attention mask stats - Active positions: {active_positions}/{total_positions} ({padding_ratio:.2%} padding)")

                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                raw_audio_subset = audio_feature[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                raw_video_subset = video_feature[:, -5:, :5]    # All samples, last 5 steps, first 5 features  
                raw_text_subset = text_feature[:, -5:, :5]      # All samples, last 5 steps, first 5 features
        
                logger.info(f"\n===== RAW INPUT FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Raw Audio features (shape {raw_audio_subset.shape}):\n{raw_audio_subset}")
                logger.info(f"Raw Video features (shape {raw_video_subset.shape}):\n{raw_video_subset}")  
                logger.info(f"Raw Text features (shape {raw_text_subset.shape}):\n{raw_text_subset}")
                logger.info("=" * 70)


        if n_iter % 100 == 0:  # Log every 10 iterations to match semantic logging frequency
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - RAW FEATURES (Iter {n_iter}) =====")
                logger.info(f"Batch size: {text_feature.size(0)}")
                logger.info(f"Original lengths from dataloader: {lengths}")
                logger.info(f"Audio feature shape: {audio_feature.shape} (timesteps: {audio_feature.size(1)})")
                logger.info(f"Video feature shape: {video_feature.shape} (timesteps: {video_feature.size(1)})")
                logger.info(f"Text feature shape: {text_feature.shape} (timesteps: {text_feature.size(1)})")
                
                # Count effective (non-padded) timesteps using attention mask
                effective_lengths = attention_mask.sum(dim=1)  # Sum along sequence dimension
                logger.info(f"Effective lengths from attention mask: {effective_lengths}")

                # Log attention mask information
                logger.info(f"\n===== ATTENTION MASK INFO (Iter {n_iter}) =====")
                logger.info(f"Attention mask shape: {attention_mask.shape}")
                logger.info(f"Full attention masks for all batch samples:\n{attention_mask}")
                
                # Show last 5 timesteps of attention mask to match feature logging
                attention_mask_last5 = attention_mask[:, -5:]
                logger.info(f"Attention mask (last 5 timesteps, shape {attention_mask_last5.shape}):\n{attention_mask_last5}")
                
                # Show attention mask statistics
                total_positions = attention_mask.numel()
                active_positions = attention_mask.sum().item()
                padding_ratio = 1.0 - (active_positions / total_positions)
                logger.info(f"Attention mask stats - Active positions: {active_positions}/{total_positions} ({padding_ratio:.2%} padding)")

                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                raw_audio_subset = audio_feature[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                raw_video_subset = video_feature[:, -5:, :5]    # All samples, last 5 steps, first 5 features  
                raw_text_subset = text_feature[:, -5:, :5]      # All samples, last 5 steps, first 5 features
        
                logger.info(f"\n===== RAW INPUT FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Raw Audio features (shape {raw_audio_subset.shape}):\n{raw_audio_subset}")
                logger.info(f"Raw Video features (shape {raw_video_subset.shape}):\n{raw_video_subset}")  
                logger.info(f"Raw Text features (shape {raw_text_subset.shape}):\n{raw_text_subset}")
                logger.info("=" * 70)






        if n_iter == 0:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - ENCODER RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio encoder shape: {audio_encoder_result.shape} (timesteps: {audio_encoder_result.size(1)})")
                logger.info(f"Video encoder shape: {video_encoder_result.shape} (timesteps: {video_encoder_result.size(1)})")
                logger.info(f"Text encoder shape: {text_encoder_result.shape} (timesteps: {text_encoder_result.size(1)})")
                
                # Check if sequence length changed after encoding
                length_change_audio = audio_feature.size(1) - audio_encoder_result.size(1)
                length_change_video = video_feature.size(1) - video_encoder_result.size(1)
                length_change_text = text_feature.size(1) - text_encoder_result.size(1)
                logger.info(f"Length changes after encoding - Audio: {length_change_audio}, Video: {length_change_video}, Text: {length_change_text}")
        
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_encoder = audio_encoder_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_encoder = video_encoder_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_encoder = text_encoder_result[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== ENOCDER FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio Encoder (shape {audio_encoder.shape}):\n{audio_encoder}")
                logger.info(f"Video Encoder (shape {video_encoder.shape}):\n{video_encoder}")
                logger.info(f"Text Encoder (shape {text_encoder.shape}):\n{text_encoder}")
                logger.info("=" * 70)

        if n_iter == 1:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - ENCODER RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio encoder shape: {audio_encoder_result.shape} (timesteps: {audio_encoder_result.size(1)})")
                logger.info(f"Video encoder shape: {video_encoder_result.shape} (timesteps: {video_encoder_result.size(1)})")
                logger.info(f"Text encoder shape: {text_encoder_result.shape} (timesteps: {text_encoder_result.size(1)})")
                
                # Check if sequence length changed after encoding
                length_change_audio = audio_feature.size(1) - audio_encoder_result.size(1)
                length_change_video = video_feature.size(1) - video_encoder_result.size(1)
                length_change_text = text_feature.size(1) - text_encoder_result.size(1)
                logger.info(f"Length changes after encoding - Audio: {length_change_audio}, Video: {length_change_video}, Text: {length_change_text}")
        
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_encoder = audio_encoder_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_encoder = video_encoder_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_encoder = text_encoder_result[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== ENOCDER FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio Encoder (shape {audio_encoder.shape}):\n{audio_encoder}")
                logger.info(f"Video Encoder (shape {video_encoder.shape}):\n{video_encoder}")
                logger.info(f"Text Encoder (shape {text_encoder.shape}):\n{text_encoder}")
                logger.info("=" * 70)

        if n_iter % 100 == 0:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - ENCODER RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio encoder shape: {audio_encoder_result.shape} (timesteps: {audio_encoder_result.size(1)})")
                logger.info(f"Video encoder shape: {video_encoder_result.shape} (timesteps: {video_encoder_result.size(1)})")
                logger.info(f"Text encoder shape: {text_encoder_result.shape} (timesteps: {text_encoder_result.size(1)})")
                
                # Check if sequence length changed after encoding
                length_change_audio = audio_feature.size(1) - audio_encoder_result.size(1)
                length_change_video = video_feature.size(1) - video_encoder_result.size(1)
                length_change_text = text_feature.size(1) - text_encoder_result.size(1)
                logger.info(f"Length changes after encoding - Audio: {length_change_audio}, Video: {length_change_video}, Text: {length_change_text}")
        
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_encoder = audio_encoder_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_encoder = video_encoder_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_encoder = text_encoder_result[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== ENOCDER FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio Encoder (shape {audio_encoder.shape}):\n{audio_encoder}")
                logger.info(f"Video Encoder (shape {video_encoder.shape}):\n{video_encoder}")
                logger.info(f"Text Encoder (shape {text_encoder.shape}):\n{text_encoder}")
                logger.info("=" * 70)

        
        
        if n_iter == 0:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - SEMANTIC RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio semantic shape: {audio_semantic_result.shape} (timesteps: {audio_semantic_result.size(1)})")
                logger.info(f"Video semantic shape: {video_semantic_result.shape} (timesteps: {video_semantic_result.size(1)})")
                logger.info(f"Text semantic shape: {text_semantic_result.shape} (timesteps: {text_semantic_result.size(1)})")
                
                # Check if sequence length changed from encoder to semantic
                semantic_length_change_audio = audio_encoder_result.size(1) - audio_semantic_result.size(1)
                semantic_length_change_video = video_encoder_result.size(1) - video_semantic_result.size(1)
                semantic_length_change_text = text_encoder_result.size(1) - text_semantic_result.size(1)
                logger.info(f"Length changes from encoder to semantic - Audio: {semantic_length_change_audio}, Video: {semantic_length_change_video}, Text: {semantic_length_change_text}")
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_subset = audio_semantic_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_subset = video_semantic_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_subset = text_semantic_result[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== SEMANTIC FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio semantic (shape {audio_subset.shape}):\n{audio_subset}")
                logger.info(f"Video semantic (shape {video_subset.shape}):\n{video_subset}")
                logger.info(f"Text semantic (shape {text_subset.shape}):\n{text_subset}")
                logger.info("=" * 70)

        
        if n_iter == 1:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - SEMANTIC RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio semantic shape: {audio_semantic_result.shape} (timesteps: {audio_semantic_result.size(1)})")
                logger.info(f"Video semantic shape: {video_semantic_result.shape} (timesteps: {video_semantic_result.size(1)})")
                logger.info(f"Text semantic shape: {text_semantic_result.shape} (timesteps: {text_semantic_result.size(1)})")
                
                # Check if sequence length changed from encoder to semantic
                semantic_length_change_audio = audio_encoder_result.size(1) - audio_semantic_result.size(1)
                semantic_length_change_video = video_encoder_result.size(1) - video_semantic_result.size(1)
                semantic_length_change_text = text_encoder_result.size(1) - text_semantic_result.size(1)
                logger.info(f"Length changes from encoder to semantic - Audio: {semantic_length_change_audio}, Video: {semantic_length_change_video}, Text: {semantic_length_change_text}")
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_subset = audio_semantic_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_subset = video_semantic_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_subset = text_semantic_result[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== SEMANTIC FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio semantic (shape {audio_subset.shape}):\n{audio_subset}")
                logger.info(f"Video semantic (shape {video_subset.shape}):\n{video_subset}")
                logger.info(f"Text semantic (shape {text_subset.shape}):\n{text_subset}")
                logger.info("=" * 70)


        if n_iter % 100 == 0:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - SEMANTIC RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio semantic shape: {audio_semantic_result.shape} (timesteps: {audio_semantic_result.size(1)})")
                logger.info(f"Video semantic shape: {video_semantic_result.shape} (timesteps: {video_semantic_result.size(1)})")
                logger.info(f"Text semantic shape: {text_semantic_result.shape} (timesteps: {text_semantic_result.size(1)})")
                
                # Check if sequence length changed from encoder to semantic
                semantic_length_change_audio = audio_encoder_result.size(1) - audio_semantic_result.size(1)
                semantic_length_change_video = video_encoder_result.size(1) - video_semantic_result.size(1)
                semantic_length_change_text = text_encoder_result.size(1) - text_semantic_result.size(1)
                logger.info(f"Length changes from encoder to semantic - Audio: {semantic_length_change_audio}, Video: {semantic_length_change_video}, Text: {semantic_length_change_text}")
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_subset = audio_semantic_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_subset = video_semantic_result[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_subset = text_semantic_result[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== SEMANTIC FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio semantic (shape {audio_subset.shape}):\n{audio_subset}")
                logger.info(f"Video semantic (shape {video_subset.shape}):\n{video_subset}")
                logger.info(f"Text semantic (shape {text_subset.shape}):\n{text_subset}")
                logger.info("=" * 70)
        
        if n_iter == 0:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - VQ RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio VQ shape: {out_vq_audio.shape} (timesteps: {out_vq_audio.size(1)})")
                logger.info(f"Video VQ shape: {out_vq_video.shape} (timesteps: {out_vq_video.size(1)})")
                logger.info(f"Text VQ shape: {out_vq_text.shape} (timesteps: {out_vq_text.size(1)})")
                
                # # Check if sequence length changed from semantic to VQ
                # vq_length_change_audio = audio_semantic_result.size(1) - out_vq_audio.size(1)
                # vq_length_change_video = video_semantic_result.size(1) - out_vq_video.size(1)
                # vq_length_change_text = text_semantic_result.size(1) - out_vq_text.size(1)
                # logger.info(f"Length changes from semantic to VQ - Audio: {vq_length_change_audio}, Video: {vq_length_change_video}, Text: {vq_length_change_text}")
        
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_vq_subset = out_vq_audio[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_vq_subset = out_vq_video[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_vq_subset = out_vq_text[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== VQ FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio VQ (shape {audio_vq_subset.shape}):\n{audio_vq_subset}")
                logger.info(f"Video VQ (shape {video_vq_subset.shape}):\n{video_vq_subset}")
                logger.info(f"Text VQ (shape {text_vq_subset.shape}):\n{text_vq_subset}")
                logger.info("=" * 70)

        if n_iter == 1:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - VQ RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio VQ shape: {out_vq_audio.shape} (timesteps: {out_vq_audio.size(1)})")
                logger.info(f"Video VQ shape: {out_vq_video.shape} (timesteps: {out_vq_video.size(1)})")
                logger.info(f"Text VQ shape: {out_vq_text.shape} (timesteps: {out_vq_text.size(1)})")
                
                # # Check if sequence length changed from semantic to VQ
                # vq_length_change_audio = audio_semantic_result.size(1) - out_vq_audio.size(1)
                # vq_length_change_video = video_semantic_result.size(1) - out_vq_video.size(1)
                # vq_length_change_text = text_semantic_result.size(1) - out_vq_text.size(1)
                # logger.info(f"Length changes from semantic to VQ - Audio: {vq_length_change_audio}, Video: {vq_length_change_video}, Text: {vq_length_change_text}")
        
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_vq_subset = out_vq_audio[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_vq_subset = out_vq_video[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_vq_subset = out_vq_text[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== VQ FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio VQ (shape {audio_vq_subset.shape}):\n{audio_vq_subset}")
                logger.info(f"Video VQ (shape {video_vq_subset.shape}):\n{video_vq_subset}")
                logger.info(f"Text VQ (shape {text_vq_subset.shape}):\n{text_vq_subset}")
                logger.info("=" * 70)

        if n_iter % 100 == 0:  # Log every 10 iterations for frequent monitoring
            with torch.no_grad():
                logger.info(f"\n===== SEQUENCE LENGTH INFO - VQ RESULTS (Iter {n_iter}) =====")
                logger.info(f"Audio VQ shape: {out_vq_audio.shape} (timesteps: {out_vq_audio.size(1)})")
                logger.info(f"Video VQ shape: {out_vq_video.shape} (timesteps: {out_vq_video.size(1)})")
                logger.info(f"Text VQ shape: {out_vq_text.shape} (timesteps: {out_vq_text.size(1)})")
                
                # # Check if sequence length changed from semantic to VQ
                # vq_length_change_audio = audio_semantic_result.size(1) - out_vq_audio.size(1)
                # vq_length_change_video = video_semantic_result.size(1) - out_vq_video.size(1)
                # vq_length_change_text = text_semantic_result.size(1) - out_vq_text.size(1)
                # logger.info(f"Length changes from semantic to VQ - Audio: {vq_length_change_audio}, Video: {vq_length_change_video}, Text: {vq_length_change_text}")
        
                # Extract last 5 timesteps and first 5 feature dimensions: [batch_size, 5_timesteps, 5_features]
                audio_vq_subset = out_vq_audio[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                video_vq_subset = out_vq_video[:, -5:, :5]  # All samples, last 5 steps, first 5 features
                text_vq_subset = out_vq_text[:, -5:, :5]    # All samples, last 5 steps, first 5 features
                
                logger.info(f"\n===== VQ FEATURES - Last 5 Steps, First 5 Dims (Iter {n_iter}) =====")
                logger.info(f"Audio VQ (shape {audio_vq_subset.shape}):\n{audio_vq_subset}")
                logger.info(f"Video VQ (shape {video_vq_subset.shape}):\n{video_vq_subset}")
                logger.info(f"Text VQ (shape {text_vq_subset.shape}):\n{text_vq_subset}")
                logger.info("=" * 70)



        if n_iter == 0:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                logger.info(f"INITIALIZATION_1 - Vector 200 (first 5 video dims): {quantizer.embedding[200, :5]}")
                logger.info(f"INITIALIZATION_1 - Vector 200 (first 5 audio dims): {quantizer.embedding[200, 256:261]}")
                logger.info(f"INITIALIZATION_1 - Vector 200 (first 5 text dims): {quantizer.embedding[200, 512:517]}")
                logger.info(f"INITIALIZATION_1 - Vector 120 (first 5 video dims): {quantizer.embedding[120, :5]}")
                logger.info(f"INITIALIZATION_1 - Vector 120 (first 5 audio dims): {quantizer.embedding[120, 256:261]}")
                logger.info(f"INITIALIZATION_1 - Vector 120 (first 5 text dims): {quantizer.embedding[120, 512:517]}")
                logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 video dims): {quantizer.embedding[45, :5]}")
                logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 audio dims): {quantizer.embedding[45, 256:261]}")
                logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 text dims): {quantizer.embedding[45, 512:517]}")
                logger.info(f"INITIALIZATION Epoch 0 - Count for vector 200: {quantizer.ema_count[200]}")
                logger.info(f"INITIALIZATION Epoch 0 - Count for vector 120: {quantizer.ema_count[120]}")
                logger.info(f"INITIALIZATION Epoch 0 - Count for vector 45: {quantizer.ema_count[45]}")
                logger.info(f"Init Codebook min: {quantizer.embedding.min().item()}, max: {quantizer.embedding.max().item()}")

        if n_iter == 60:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                most_used_idx = torch.argmax(quantizer.ema_count)
                random_idx1 = (most_used_idx + 1) % 400
                random_idx2 = (most_used_idx + 100) % 400
                logger.info(f"\n===== BATCH ({n_iter}) DEBUG =====")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 video dims): {quantizer.embedding[most_used_idx, :5]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 audio dims): {quantizer.embedding[most_used_idx, 256:261]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 text dims): {quantizer.embedding[most_used_idx, 512:517]}")
                logger.info(f"Random ({random_idx1}) vector value (first 5 video dims): {quantizer.embedding[random_idx1, :5]}")
                logger.info(f"Random {random_idx1} vector value (first 5 audio dims): {quantizer.embedding[random_idx1, 256:261]}")
                logger.info(f"Random {random_idx1} vector value (first 5 text dims): {quantizer.embedding[random_idx1, 512:517]}")
                logger.info(f"Random {random_idx2} vector value (first 5 video dims): {quantizer.embedding[random_idx2, :5]}")
                logger.info(f"Random {random_idx2} vector value (first 5 audio dims): {quantizer.embedding[random_idx2, 256:261]}")
                logger.info(f"Random {random_idx2} vector value (first 5 text dims): {quantizer.embedding[random_idx2, 512:517]}")
                logger.info(f"Count statistics - min: {quantizer.ema_count.min()}, max: {quantizer.ema_count.max()}, mean: {quantizer.ema_count.mean()}")
                logger.info(f"Count histogram: {torch.histc(quantizer.ema_count, bins=10, min=0, max=quantizer.ema_count.max())}")
                logger.info(f"Number of dead vectors (count < 0.01): {(quantizer.ema_count < 0.01).sum()}")
                logger.info(f"Embedding stats - mean: {quantizer.embedding.mean()}, std: {quantizer.embedding.std()}, min: {quantizer.embedding.min()}, max: {quantizer.embedding.max()}")

        if n_iter == 220:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                most_used_idx = torch.argmax(quantizer.ema_count)
                random_idx1 = (most_used_idx + 1) % 400
                random_idx2 = (most_used_idx + 100) % 400
                logger.info(f"\n===== BATCH ({n_iter}) DEBUG =====")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 video dims): {quantizer.embedding[most_used_idx, :5]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 audio dims): {quantizer.embedding[most_used_idx, 256:261]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 text dims): {quantizer.embedding[most_used_idx, 512:517]}")
                logger.info(f"Random ({random_idx1}) vector value (first 5 video dims): {quantizer.embedding[random_idx1, :5]}")
                logger.info(f"Random {random_idx1} vector value (first 5 audio dims): {quantizer.embedding[random_idx1, 256:261]}")
                logger.info(f"Random {random_idx1} vector value (first 5 text dims): {quantizer.embedding[random_idx1, 512:517]}")
                logger.info(f"Random {random_idx2} vector value (first 5 video dims): {quantizer.embedding[random_idx2, :5]}")
                logger.info(f"Random {random_idx2} vector value (first 5 audio dims): {quantizer.embedding[random_idx2, 256:261]}")
                logger.info(f"Random {random_idx2} vector value (first 5 text dims): {quantizer.embedding[random_idx2, 512:517]}")
                logger.info(f"Count statistics - min: {quantizer.ema_count.min()}, max: {quantizer.ema_count.max()}, mean: {quantizer.ema_count.mean()}")
                logger.info(f"Count histogram: {torch.histc(quantizer.ema_count, bins=10, min=0, max=quantizer.ema_count.max())}")
                logger.info(f"Number of dead vectors (count < 0.01): {(quantizer.ema_count < 0.01).sum()}")
                logger.info(f"Embedding stats - mean: {quantizer.embedding.mean()}, std: {quantizer.embedding.std()}, min: {quantizer.embedding.min()}, max: {quantizer.embedding.max()}")

        if n_iter == 340:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                most_used_idx = torch.argmax(quantizer.ema_count)
                random_idx1 = (most_used_idx + 1) % 400
                random_idx2 = (most_used_idx + 100) % 400
                logger.info(f"\n===== BATCH ({n_iter}) DEBUG =====")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 video dims): {quantizer.embedding[most_used_idx, :5]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 audio dims): {quantizer.embedding[most_used_idx, 256:261]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 text dims): {quantizer.embedding[most_used_idx, 512:517]}")
                logger.info(f"Random ({random_idx1}) vector value (first 5 video dims): {quantizer.embedding[random_idx1, :5]}")
                logger.info(f"Random {random_idx1} vector value (first 5 audio dims): {quantizer.embedding[random_idx1, 256:261]}")
                logger.info(f"Random {random_idx1} vector value (first 5 text dims): {quantizer.embedding[random_idx1, 512:517]}")
                logger.info(f"Random {random_idx2} vector value (first 5 video dims): {quantizer.embedding[random_idx2, :5]}")
                logger.info(f"Random {random_idx2} vector value (first 5 audio dims): {quantizer.embedding[random_idx2, 256:261]}")
                logger.info(f"Random {random_idx2} vector value (first 5 text dims): {quantizer.embedding[random_idx2, 512:517]}")
                logger.info(f"Count statistics - min: {quantizer.ema_count.min()}, max: {quantizer.ema_count.max()}, mean: {quantizer.ema_count.mean()}")
                logger.info(f"Count histogram: {torch.histc(quantizer.ema_count, bins=10, min=0, max=quantizer.ema_count.max())}")
                logger.info(f"Number of dead vectors (count < 0.01): {(quantizer.ema_count < 0.01).sum()}")
                logger.info(f"Embedding stats - mean: {quantizer.embedding.mean()}, std: {quantizer.embedding.std()}, min: {quantizer.embedding.min()}, max: {quantizer.embedding.max()}")

        if n_iter == 640:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                most_used_idx = torch.argmax(quantizer.ema_count)
                random_idx1 = (most_used_idx + 1) % 400
                random_idx2 = (most_used_idx + 100) % 400
                logger.info(f"\n===== BATCH ({n_iter}) DEBUG =====")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 video dims): {quantizer.embedding[most_used_idx, :5]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 audio dims): {quantizer.embedding[most_used_idx, 256:261]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 text dims): {quantizer.embedding[most_used_idx, 512:517]}")
                logger.info(f"Random ({random_idx1}) vector value (first 5 video dims): {quantizer.embedding[random_idx1, :5]}")
                logger.info(f"Random {random_idx1} vector value (first 5 audio dims): {quantizer.embedding[random_idx1, 256:261]}")
                logger.info(f"Random {random_idx1} vector value (first 5 text dims): {quantizer.embedding[random_idx1, 512:517]}")
                logger.info(f"Random {random_idx2} vector value (first 5 video dims): {quantizer.embedding[random_idx2, :5]}")
                logger.info(f"Random {random_idx2} vector value (first 5 audio dims): {quantizer.embedding[random_idx2, 256:261]}")
                logger.info(f"Random {random_idx2} vector value (first 5 text dims): {quantizer.embedding[random_idx2, 512:517]}")
                logger.info(f"Count statistics - min: {quantizer.ema_count.min()}, max: {quantizer.ema_count.max()}, mean: {quantizer.ema_count.mean()}")
                logger.info(f"Count histogram: {torch.histc(quantizer.ema_count, bins=10, min=0, max=quantizer.ema_count.max())}")
                logger.info(f"Number of dead vectors (count < 0.01): {(quantizer.ema_count < 0.01).sum()}")
                logger.info(f"Embedding stats - mean: {quantizer.embedding.mean()}, std: {quantizer.embedding.std()}, min: {quantizer.embedding.min()}, max: {quantizer.embedding.max()}")

        if n_iter == 920:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                most_used_idx = torch.argmax(quantizer.ema_count)
                random_idx1 = (most_used_idx + 1) % 400
                random_idx2 = (most_used_idx + 100) % 400
                logger.info(f"\n===== BATCH ({n_iter}) DEBUG =====")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 video dims): {quantizer.embedding[most_used_idx, :5]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 audio dims): {quantizer.embedding[most_used_idx, 256:261]}")
                logger.info(f"Most used vector ({most_used_idx}) value (first 5 text dims): {quantizer.embedding[most_used_idx, 512:517]}")
                logger.info(f"Random ({random_idx1}) vector value (first 5 video dims): {quantizer.embedding[random_idx1, :5]}")
                logger.info(f"Random {random_idx1} vector value (first 5 audio dims): {quantizer.embedding[random_idx1, 256:261]}")
                logger.info(f"Random {random_idx1} vector value (first 5 text dims): {quantizer.embedding[random_idx1, 512:517]}")
                logger.info(f"Random {random_idx2} vector value (first 5 video dims): {quantizer.embedding[random_idx2, :5]}")
                logger.info(f"Random {random_idx2} vector value (first 5 audio dims): {quantizer.embedding[random_idx2, 256:261]}")
                logger.info(f"Random {random_idx2} vector value (first 5 text dims): {quantizer.embedding[random_idx2, 512:517]}")
                logger.info(f"Count statistics - min: {quantizer.ema_count.min()}, max: {quantizer.ema_count.max()}, mean: {quantizer.ema_count.mean()}")
                logger.info(f"Count histogram: {torch.histc(quantizer.ema_count, bins=10, min=0, max=quantizer.ema_count.max())}")
                logger.info(f"Number of dead vectors (count < 0.01): {(quantizer.ema_count < 0.01).sum()}")
                logger.info(f"Embedding stats - mean: {quantizer.embedding.mean()}, std: {quantizer.embedding.std()}, min: {quantizer.embedding.min()}, max: {quantizer.embedding.max()}")


        accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, cpc_loss, \
        audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score, combined_score \
        = mi_first_forward(CPC, audio_feature, video_feature, text_feature, Decoder,epoch,
                        audio_encoder_result, audio_semantic_result, video_encoder_result, video_semantic_result,
                        text_encoder_result, text_semantic_result,
                        out_vq_audio, audio_vq, out_vq_video, video_vq,
                        out_vq_text, text_vq, lengths=lengths, attention_mask=attention_mask)


        if n_iter % 20 == 0:
            logger.info("equal_num is {} in {}-th iteration.".format(equal_num, n_iter))
        
        loss_items = {
            "audio_recon_loss": audio_recon_loss.item(),
            "audio_embed_loss": audio_embedding_loss.item(),
            "text_recon_loss": text_recon_loss.item(),
            "text_embed_loss": text_embedding_loss.item(),
            "video_recon_loss": video_recon_loss.item(),
            "video_embed_loss": video_embedding_loss.item(),
            "acc_av": accuracy1.item(),
            "acc_at": accuracy2.item(),
            "acc_vt": accuracy3.item(),
            "acc_va": accuracy4.item(),
            "acc_ta": accuracy5.item(),
            "acc_tv": accuracy6.item(),
            "acc_aa": accuracy7.item(),
            "acc_vv": accuracy8.item(),
            "acc_tt": accuracy9.item(),
            "cpc_loss": cpc_loss.item(),
            "cmcm_loss": cmcm_loss.item(),
            "segment_loss": segment_loss.item(),
            "audio_perplexity": audio_perplexity.item(),
            "video_perplexity": video_perplexity.item(),
            "text_perplexity": text_perplexity.item()
        }

        metricsContainer.update("loss", loss_items)

        # VGG downstream
        loss =  audio_recon_loss + video_recon_loss + text_recon_loss + audio_embedding_loss +  video_embedding_loss + text_embedding_loss + cmcm_loss + cpc_loss

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=0.0004, loss_meter=metricsContainer.calculate_average("loss"))
        
        # Update epoch loss meters
        batch_size = audio_feature.size(0)
        epoch_total_loss.update(loss.item(), batch_size * audio_feature.size(1))
        epoch_audio_recon.update(audio_recon_loss.item(), batch_size * audio_feature.size(1))
        epoch_video_recon.update(video_recon_loss.item(), batch_size * audio_feature.size(1))
        epoch_text_recon.update(text_recon_loss.item(), batch_size * audio_feature.size(1))
        epoch_audio_embed.update(audio_embedding_loss.item(), batch_size * audio_feature.size(1))
        epoch_video_embed.update(video_embedding_loss.item(), batch_size * audio_feature.size(1))
        epoch_text_embed.update(text_embedding_loss.item(), batch_size * audio_feature.size(1))
        epoch_cpc_loss.update(cpc_loss.item(), batch_size * audio_feature.size(1))
        epoch_cmcm_loss.update(cmcm_loss.item(), batch_size * audio_feature.size(1))
       
        loss.backward()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), audio_feature.size(0) * audio_feature.size(1))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    epoch_losses = [epoch_total_loss.avg, epoch_audio_recon.avg, epoch_video_recon.avg, 
                    epoch_text_recon.avg, epoch_audio_embed.avg, epoch_video_embed.avg, 
                    epoch_text_embed.avg, epoch_cpc_loss.avg, epoch_cmcm_loss.avg]


    return losses.avg, n_iter + total_step, epoch_losses

        


def mi_first_forward(CPC, audio_feature, video_feature, text_feature, Decoder,epoch,
                      audio_encoder_result, audio_semantic_result, video_encoder_result, 
                      video_semantic_result, text_encoder_result, text_semantic_result, out_vq_audio,
                      audio_vq, out_vq_video, video_vq, out_vq_text, text_vq, lengths=None, attention_mask=None):

    
    """Cross_CPC"""

    accuracy1, accuracy2, accuracy3, accuracy4, \
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, \
    cpc_loss = CPC(audio_semantic_result, video_semantic_result, text_semantic_result, lengths=lengths, attention_mask=attention_mask )


    audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score, combined_score \
        = Decoder(audio_feature, video_feature, text_feature, audio_encoder_result, video_encoder_result, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq, attention_mask=attention_mask)
    


    
    return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, cpc_loss,\
           audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score, combined_score
    
   

if __name__ == '__main__':
    main()