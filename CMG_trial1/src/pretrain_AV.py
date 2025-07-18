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
from model.main_model_2 import AV_VQVAE_Encoder, AV_VQVAE_Decoder
from model.CLUB import CLUBSample_group
from model.CPC import Cross_CPC
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
# from transformers import BertTokenizer, BertModel
import pickle

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
def transpose(x):
    return x.transpose(-2, -1)


def collate_func_AV(samples):
    bsz = len(samples)
    return {
        'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
    }


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



    '''dataset selection'''
    
    if args.dataset_name =='vggsound':
        from dataset.VGGSOUND_dataset import VGGSoundDataset_AV_1 as AVEDataset 
    else:
        raise NotImplementedError

    '''Dataloader selection'''
    
    ########## 90 k #############
    meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv'
    audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/audio80k_features_new'
    video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/video80k_features_keras'
    train_dataloader = DataLoader(
        AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_func_AV
    )
    ########## 90 k #############

    ########## 40 k #############
    # meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv'
    # audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip'
    # video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip'
    # train_dataloader = DataLoader(
    #     AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=8,
    #     pin_memory=False,
    #     collate_fn=collate_func_AV
    # )
    ########## 40 k #############

    '''model setting'''
    video_dim = 512
    audio_dim = 128
    video_output_dim = 2048
    audio_output_dim = 256
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = False
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Encoder = AV_VQVAE_Encoder(video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, embedding_dim)
    CPC = Cross_CPC(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    
    Video_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=video_dim, hidden_size=256)
    Audio_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=audio_output_dim, hidden_size=256)

    
    Decoder = AV_VQVAE_Decoder(video_dim, audio_dim, video_output_dim, audio_output_dim)

    Encoder.double()
    CPC.double()
    Video_mi_net.double()
    Audio_mi_net.double()
    Decoder.double()

    '''optimizer setting'''
    Encoder.to(device)
    CPC.to(device)
    Video_mi_net.to(device)
    Audio_mi_net.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=args.lr)
    optimizer_video_mi_net = torch.optim.Adam(Video_mi_net.parameters(), lr=args.mi_lr)
    optimizer_audio_mi_net = torch.optim.Adam(Audio_mi_net.parameters(), lr=args.mi_lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        path_checkpoints = ""
        print(path_checkpoints)
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        CPC.load_state_dict(checkpoints['CPC_parameters'])
        Video_mi_net.load_state_dict(checkpoints['Video_mi_net_parameters'])
        Audio_mi_net.load_state_dict(checkpoints['Audio_mi_net_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        optimizer_audio_mi_net.load_state_dict(checkpoints['optimizer_audio_mi_net'])
        optimizer_video_mi_net.load_state_dict(checkpoints['optimizer_video_mi_net'])
        start_epoch = checkpoints['epoch']
        total_step = checkpoints['total_step']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    
    '''Training and Evaluation'''
    for epoch in range(start_epoch+1, args.n_epoch):
        loss, total_step = train_epoch(CPC, Encoder, Audio_mi_net, Video_mi_net, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch, total_step, args)
        
        
        save_path = os.path.join(args.model_save_path, 'DCID-model-{}.pt'.format(epoch))
        save_models(CPC, Encoder, Audio_mi_net, Video_mi_net, Decoder, optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch, total_step, save_path)
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

# If resuming training is not required, downstream tasks only need to save the encoder & epoch & Text_ar_lstm, as these are the only components needed for inference.
def save_models(CPC, Encoder, Audio_mi_net, Video_mi_net, Decoder, optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'CPC_parameters': CPC.state_dict(),
        'Video_mi_net_parameters': Video_mi_net.state_dict(),
        'Audio_mi_net_parameters': Audio_mi_net.state_dict(),
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_video_mi_net': optimizer_video_mi_net.state_dict(),
        'optimizer_audio_mi_net': optimizer_audio_mi_net.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))

def train_epoch_check(train_dataloader, epoch, total_step, args):
    train_dataloader = tqdm(train_dataloader)
    for n_iter, batch_data in enumerate(train_dataloader):
        
        '''Feed input to model'''
        visual_feature, audio_feature = batch_data
        visual_feature.cuda()
        audio_feature.cuda()
        
    return torch.zeros(1)   


def train_epoch(CPC,Encoder, Audio_mi_net, Video_mi_net, Decoder,train_dataloader, criterion, criterion_event, optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    models = [CPC,Encoder, Decoder]
    to_train(models)
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.

    Encoder.cuda()
    Audio_mi_net.cuda()
    Video_mi_net.cuda()
    Decoder.cuda()
    CPC.cuda()
    optimizer.zero_grad()
    mi_iters = 5


    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        
        # vggsound_AV
        audio_feature, video_feature = batch_data['audio_fea'], batch_data['video_fea']

        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)
        

        video_semantic_result, audio_semantic_result, video_encoder_result, video_club_feature, audio_encoder_result, \
        video_vq, audio_vq, audio_embedding_loss, video_embedding_loss, cmcm_loss, equal_num, audio_perplexity, video_perplexity\
        = Encoder(audio_feature, video_feature, epoch)

        optimizer_audio_mi_net, lld_audio_loss, optimizer_video_mi_net, lld_video_loss = \
        mi_first_forward(Audio_mi_net, Video_mi_net, optimizer_audio_mi_net,optimizer_video_mi_net, epoch, 
                         audio_vq.detach(), video_vq.detach(),audio_encoder_result.detach(), 
                         video_encoder_result.detach(), video_club_feature.detach(),mi_iters)
        
        mi_audio_loss, mi_video_loss, \
        accuracy1, accuracy2, accuracy3, accuracy4,\
        cpc_loss, audio_recon_loss, video_recon_loss,\
        audio_class, video_class\
        = mi_second_forward(CPC, audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net, Decoder,epoch,
                audio_semantic_result, video_semantic_result,
                audio_encoder_result, video_encoder_result, video_club_feature,
                audio_vq, video_vq)

        if n_iter % 20 == 0:
            logger.info("equal_num is {} in {}-th iteration.".format(equal_num, n_iter))

        loss_items = {
            "audio_recon_loss": audio_recon_loss.item(),
            "lld_audio_loss": lld_audio_loss.item(),
            "audio_embed_loss": audio_embedding_loss.item(),
            "audio_mine_loss": mi_audio_loss.item(),
            "video_recon_loss": video_recon_loss.item(),
            "lld_video_loss": lld_video_loss.item(),
            "video_embed_loss": video_embedding_loss.item(),
            "video_mine_loss": mi_video_loss.item(),
            "acc_va": accuracy1.item(),
            "acc_av": accuracy2.item(),
            "acc_vv": accuracy3.item(),
            "acc_aa": accuracy4.item(),
            "cpc_loss": cpc_loss.item(),
            "cmcm_loss": cmcm_loss.item(),
            "audio_perplexity": audio_perplexity.item(),
            "video_perplexity": video_perplexity.item()
        }

        metricsContainer.update("loss", loss_items)
        loss = audio_recon_loss + video_recon_loss + audio_embedding_loss +  video_embedding_loss\
                 + mi_audio_loss + mi_video_loss + cpc_loss + cmcm_loss

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=0.0004, loss_meter=metricsContainer.calculate_average("loss"))
        
        loss.backward()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), audio_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # '''Add loss of a iteration in Tensorboard'''
        # writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        # '''Add loss of an epoch in Tensorboard'''
        # writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg, n_iter + total_step



def mi_first_forward(Audio_mi_net, Video_mi_net, optimizer_audio_mi_net,optimizer_video_mi_net, epoch,
                        audio_vq, video_vq,
                        audio_encoder_result, video_encoder_result, 
                        video_club_feature, mi_iters):

    for i in range(mi_iters):
        optimizer_video_mi_net.zero_grad()
        optimizer_audio_mi_net.zero_grad()

        # video processing is different from audio and text modalities because video is more complex and feature extraction is more difficult.
        lld_video_loss = -Video_mi_net.loglikeli(video_vq, video_club_feature)
        lld_video_loss.backward()
        optimizer_video_mi_net.step()

        lld_audio_loss = -Audio_mi_net.loglikeli(audio_vq, audio_encoder_result)
        lld_audio_loss.backward()
        optimizer_audio_mi_net.step()

    return optimizer_audio_mi_net, lld_audio_loss, optimizer_video_mi_net, lld_video_loss


def mi_second_forward(CPC, audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net, Decoder,epoch,
                      audio_semantic_result, video_semantic_result,
                      audio_encoder_result, video_encoder_result, video_club_feature,
                      audio_vq, video_vq):
    # audio_semantic_result, video_semantic_result, text_semantic_result, \
    # audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
    # audio_vq, video_vq, text_vq, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num \
    # = Encoder(audio_feature, video_feature, text_feature, epoch)
    
    mi_video_loss = Video_mi_net.mi_est(video_vq, video_club_feature)
    mi_audio_loss = Audio_mi_net.mi_est(audio_vq, audio_encoder_result)
    
    """Cross_CPC"""
    accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss = CPC(video_semantic_result, audio_semantic_result)

    video_recon_loss, audio_recon_loss, video_class, audio_class \
        = Decoder(video_feature, audio_feature, video_encoder_result, audio_encoder_result, video_vq, audio_vq)
    
    return mi_audio_loss, mi_video_loss, \
           accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss,  \
           audio_recon_loss, video_recon_loss, audio_class, video_class

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)
    

if __name__ == '__main__':
    main()
