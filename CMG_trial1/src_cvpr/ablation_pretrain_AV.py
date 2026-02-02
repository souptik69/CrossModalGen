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
from model.main_model_ablation import AV_VQVAE_Encoder, AV_VQVAE_Decoder
from model.CPC import Cross_CPC
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
import pickle
from collections import Counter
torch.autograd.set_detect_anomaly(True)
import torch.cuda as cuda

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
    if args.dataset_name == 'vggsound_AVT_90k' or args.dataset_name == 'vggsound_AVT_40k':
        from dataset.VGG_dataset_novel import VGGSoundDataset_AV_novel as AVEDataset
    else:
        raise NotImplementedError 

    '''Dataloader Loading'''
    
    if args.dataset_name == 'vggsound_AVT_90k':
        meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv'
        audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/audio80k_features_new'
        video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/video80k_features_keras'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_func_AV
        )

    elif args.dataset_name == 'vggsound_AVT_40k':
        meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv'
        audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip'
        video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_func_AV
        )
    
    else:
        raise NotImplementedError

   

    '''model setting'''
    video_dim = 512
    audio_dim = 128
    video_output_dim = 2048
    n_embeddings = args.n_embeddings
    embedding_dim = args.embedding_dim
    start_epoch = -1
    model_resume = False
    # model_resume = True
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Encoder = AV_VQVAE_Encoder( audio_dim, video_dim, video_output_dim, n_embeddings, embedding_dim, ablation=args.ablation)
    CPC = Cross_CPC(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    Decoder = AV_VQVAE_Decoder(audio_dim, video_dim, video_output_dim, embedding_dim, ablation=args.ablation)

    Encoder.double()
    CPC.double()
    Decoder.double()

    '''optimizer setting'''
    Encoder.to(device)
    CPC.to(device)
    Decoder.to(device)


    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        path_checkpoints = ""
        # path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/Hier/40k/checkpoint/DCID-model-5.pt"
        print(path_checkpoints)
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        CPC.load_state_dict(checkpoints['CPC_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        start_epoch = checkpoints['epoch']
        total_step = checkpoints['total_step']
        logger.info("Resume from number {}-th model.".format(start_epoch))
    
    '''Training and Evaluation'''
    for epoch in range(start_epoch+1, args.n_epoch):
        epoch_start_time = time.time()
        loss, total_step = train_epoch(CPC, Encoder, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, epoch, total_step, args)
        epoch_time = time.time() - epoch_start_time
        save_path = os.path.join(args.model_save_path, 'HierVQ-model-AV-{}.pt'.format(epoch))
        save_models(CPC, Encoder, Decoder, optimizer, epoch, total_step, save_path)
        logger.info(f"epoch: ******************************************* {epoch}")
        logger.info(f"loss: {loss}")
        logger.info(f"Epoch time: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)") 
        scheduler.step()
        
    if torch.cuda.is_available():
        logger.info("=" * 80)
        logger.info("GPU Memory Summary:")
        logger.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
        logger.info(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info("=" * 80)

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
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch(CPC, Encoder, Decoder, train_dataloader, criterion, criterion_event, optimizer, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    models = [CPC, Encoder, Decoder]
    to_train(models)

    Encoder.cuda()
    Decoder.cuda()
    CPC.cuda()
    optimizer.zero_grad()

    
    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''

        audio_feature, video_feature = batch_data['audio_fea'], batch_data['video_fea']

        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)


        audio_semantic_result, audio_encoder_result, video_semantic_result, video_spatial, out_vq_video, video_vq, out_vq_audio, \
        audio_vq, video_embedding_loss, audio_embedding_loss, video_perplexity, audio_perplexity,\
        equal_num, cmcm_loss, segment_loss\
        = Encoder(audio_feature, video_feature, epoch)



        accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss,  \
        audio_recon_loss, video_recon_loss, audio_class, video_class\
        = mi_first_forward(CPC, audio_feature, video_feature, Decoder,epoch,
                        audio_encoder_result, audio_semantic_result, video_spatial, video_semantic_result,
                        out_vq_audio, audio_vq, out_vq_video, video_vq, criterion_event)
        
       
        if n_iter % 20 == 0:
            logger.info("equal_num is {} in {}-th iteration.".format(equal_num, n_iter))


        loss_items = {
            "audio_recon_loss": audio_recon_loss.item(),
            "audio_embed_loss": audio_embedding_loss.item(),
            "video_recon_loss": video_recon_loss.item(),
            "video_embed_loss": video_embedding_loss.item(),
            "acc_va": accuracy1.item(),
            "acc_av": accuracy2.item(),
            "acc_vv": accuracy3.item(),
            "acc_aa": accuracy4.item(),
            "cpc_loss": cpc_loss.item(),
            "cmcm_loss": cmcm_loss.item(),
            "segment_loss": segment_loss.item(),
            "audio_perplexity": audio_perplexity.item(),
            "video_perplexity": video_perplexity.item()
        }


        metricsContainer.update("loss", loss_items)

        if args.ablation == 'AlignEMA' or args.ablation == 'HIER' or args.ablation == 'Reset' or args.ablation == 'EqualHIER' or args.ablation == 'UniRecon' or args.ablation == 'Order':
            loss =  audio_recon_loss + video_recon_loss + audio_embedding_loss +  video_embedding_loss\
                + cpc_loss + cmcm_loss

        elif args.ablation == 'CMCM':
            loss =  audio_recon_loss + video_recon_loss + audio_embedding_loss +  video_embedding_loss\
                + cpc_loss
        
        elif args.ablation == 'CPC':
            loss =  audio_recon_loss + video_recon_loss + audio_embedding_loss +  video_embedding_loss\
                 + cmcm_loss

        else:
            raise NotImplementedError


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


    return losses.avg, n_iter + total_step


def mi_first_forward(CPC, audio_feature, video_feature, Decoder,epoch,
                      audio_encoder_result, audio_semantic_result, video_spatial, 
                      video_semantic_result, out_vq_audio,
                      audio_vq, out_vq_video, video_vq, criterion_event):
    
    """Cross_CPC"""
    accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss = CPC(video_semantic_result, audio_semantic_result)
   
    audio_recon_loss, video_recon_loss, audio_class, video_class, \
        = Decoder(audio_feature, video_feature, audio_encoder_result, video_spatial, out_vq_audio, audio_vq, out_vq_video, video_vq)
    
   
    return accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss,\
           audio_recon_loss, video_recon_loss, audio_class, video_class

    
if __name__ == '__main__':
    main()

