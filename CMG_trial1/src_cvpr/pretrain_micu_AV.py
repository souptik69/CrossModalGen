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
from model.main_model_2 import AV_VQVAE_Encoder, AV_VQVAE_Decoder
from model.CLUB import CLUBSample_group
from model.CPC import Cross_CPC
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
import pickle
import itertools
from info_nce import InfoNCE

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

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss


def collate_func_AV(samples):
    return {
        'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
    }

class EncoderJigsaw(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8, hidden=512):
        super(EncoderJigsaw, self).__init__()
        self.enc_net = nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.ReLU(),
          nn.Linear(hidden, out_dim)
        )
        
    def forward(self, feat):
        return self.enc_net(feat)


def main():
    # utils variable
    global args, logger, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch, global_steps
    best_accuracy, best_accuracy_epoch, global_steps = 0, 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    
    parser.add_argument('--jigsaw_ratio', type=float, default=1.0,
                        help='jigsaw_ratio')
    parser.add_argument("--jigsaw_num_splits", type=int, default=2)
    parser.add_argument("--jigsaw_samples", type=int, default=2)
    parser.add_argument("--jigsaw_hidden", type=int, default=512)
    
    args = parser.parse_args()
    # select GPUs
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
    if args.dataset_name == 'ave':
        from dataset.AVE_dataset import AVEDataset as AVEDataset
    elif args.dataset_name =='vggsound':
        from dataset.VGGSOUND_dataset import VGGSoundDataset as AVEDataset 
    elif args.dataset_name =='vggsound_AVT_40k' or args.dataset_name == 'vggsound_AVT_90k':
        from dataset.VGGSOUND_dataset import VGGSoundDataset_AV_1 as AVEDataset 
    # elif args.dataset_name =='vggsound179k' or args.dataset_name =='vggsound81k':
    #     from dataset.VGGSOUND_dataset179k import VGGSoundDataset as AVEDataset     
    else:
        raise NotImplementedError
    
    
    '''Dataloader selection'''
    if args.dataset_name == 'ave':
        data_root = 'data'
        train_dataloader = DataLoader(
            AVEDataset(data_root, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            AVEDataset(data_root, split='val'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            AVEDataset(data_root, split='test'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound':
        meta_csv_path = 'vggsound-avel40k.csv'
        audio_fea_base_path = 'audio/zip'
        video_fea_base_path = 'video/zip'
        avc_label_base_path = 'label/zip'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound_AVT_90k':
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
    elif args.dataset_name == 'vggsound81k':
        meta_csv_path = 'video_name_vggsound81k_checked.csv'
        audio_fea_base_path = 'audio/zip'
        video_fea_base_path = 'video/zip'
        avc_label_base_path = '...'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound179k':
        meta_csv_path = 'video_name_vggsound179k_checked.csv'
        audio_fea_base_path = 'audio/zip'
        video_fea_base_path = 'video/zip'
        avc_label_base_path = '...'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    else:
        raise NotImplementedError

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
    
    infonce = InfoNCE().double().to(device)
    
    """MMJP"""
    modal_num = 1  # cross-modal unified
    jigsaw_indices = random.sample(range(np.math.factorial(modal_num * args.jigsaw_num_splits)), args.jigsaw_samples)

    jigsaw_cls = EncoderJigsaw(input_dim=embedding_dim*modal_num, out_dim=args.jigsaw_samples, hidden=args.jigsaw_hidden)
    jigsaw_cls = jigsaw_cls.to(device).double()
    
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters(), jigsaw_cls.parameters()), lr=args.lr)
    optimizer_video_mi_net = torch.optim.Adam(Video_mi_net.parameters(), lr=args.mi_lr)
    optimizer_audio_mi_net = torch.optim.Adam(Audio_mi_net.parameters(), lr=args.mi_lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        path_checkpoints = ""
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
        epoch_start_time = time.time()
        loss, total_step = train_epoch(CPC, Encoder, Audio_mi_net, Video_mi_net, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch, total_step, args, jigsaw_indices, jigsaw_cls, infonce)
        epoch_time = time.time() - epoch_start_time
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

def save_models(CPC, Encoder, epoch_num, total_step, path):
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

def all_apply_mask(mask_prob, inputs1, inputs2):
    mask1 = (torch.rand(inputs1.shape[:-1], device=inputs1.device) > mask_prob).double().cuda()
    mask1 = mask1.unsqueeze(-1).expand_as(inputs1)
    mask2 = (torch.rand(inputs2.shape[:-1], device=inputs2.device) > mask_prob).double().cuda()
    mask2 = mask2.unsqueeze(-1).expand_as(inputs2)
    return inputs1 * mask1, inputs2 * mask2, mask1, mask2

def single_apply_mask(mask_prob, inputs):
    mask = (torch.rand(inputs.shape[:-1], device=inputs.device) > mask_prob).double().cuda()
    mask = mask.unsqueeze(-1).expand_as(inputs)
    return inputs * mask, mask

def tanh_based_mapping(v, mid_ratio=0.3):
    return 0.01 + (mid_ratio-0.01) * (1 + np.tanh(v*v - 1))

def compute_params(x, y):
    # Find ratio between x and y
    if x >= y:
        a = 0.3
        b = tanh_based_mapping(y / x)
    else:
        a = tanh_based_mapping(x / y)
        b = 0.3
    return a, b

def train_epoch(CPC, Encoder, Audio_mi_net, Video_mi_net, Decoder, train_dataloader, criterion, criterion_event, 
                optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch, total_step, args, jigsaw_indices, jigsaw_cls, infonce):
    global global_steps
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    models = [CPC, Encoder, Audio_mi_net, Video_mi_net, Decoder]
    to_train(models)

    Encoder.cuda()
    Audio_mi_net.cuda()
    Video_mi_net.cuda()
    Decoder.cuda()
    CPC.cuda()
    optimizer.zero_grad()
    
    mi_iters = 5
    
    m_a_ratio, m_v_ratio = 0.3, 0.3
    last_score = 0

    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        
        '''Feed input to model'''
        audio_feature, video_feature = batch_data['audio_fea'], batch_data['video_fea']
        
        audio_feature = audio_feature.to(torch.float64)
        video_feature = video_feature.to(torch.float64)
        
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)
        
        video_semantic_result, audio_semantic_result, \
        video_encoder_result, video_club_feature, audio_encoder_result, \
        video_vq, audio_vq, audio_embedding_loss, video_embedding_loss, cmcm_loss, equal_num, \
        audio_perplexity, video_perplexity \
        = Encoder(audio_feature, video_feature, epoch)
        
        # w/o mi_loss in micu(oscmg)
        lld_audio_loss, lld_video_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        
        audio_embedding_loss, video_embedding_loss, mi_audio_loss, mi_video_loss, \
        accuracy1, accuracy2, accuracy3, accuracy4, \
        cpc_loss, audio_recon_loss, video_recon_loss, \
        audio_class, video_class, cmcm_loss, equal_num = mi_second_forward(
            CPC, audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net, Decoder, epoch)
        
        """Mask FC InfoNCE"""
        infonce_fine_loss, infonce_coarse_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        
        batch_dim = audio_feature.size()[0]
        m_a_s, m_v_s, _, _ = all_apply_mask(m_a_ratio, audio_semantic_result, video_semantic_result)
        
        for i in range(batch_dim):
            infonce_fine_loss += infonce(m_a_s[i], audio_semantic_result[i]) + infonce(m_a_s[i], video_semantic_result[i]) + \
                                infonce(m_v_s[i], audio_semantic_result[i]) + infonce(m_v_s[i], video_semantic_result[i])
        
        infonce_fine_loss /= batch_dim
        
        infonce_coarse_loss = infonce(torch.mean(m_a_s, dim=1), torch.mean(audio_semantic_result, dim=1)) + \
                            infonce(torch.mean(m_a_s, dim=1), torch.mean(video_semantic_result, dim=1)) + \
                            infonce(torch.mean(m_v_s, dim=1), torch.mean(audio_semantic_result, dim=1)) + \
                            infonce(torch.mean(m_v_s, dim=1), torch.mean(video_semantic_result, dim=1))
        
        # Adjust mask_ratio
        score_a_f, score_v_f, score_a_c, score_v_c = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
        for i in range(batch_dim):
            score_a_f += torch.sum(torch.diag(torch.matmul(m_a_s[i], audio_semantic_result[i].T))) + \
                         torch.sum(torch.diag(torch.matmul(m_a_s[i], video_semantic_result[i].T)))
            
            score_v_f += torch.sum(torch.diag(torch.matmul(m_v_s[i], audio_semantic_result[i].T))) + \
                         torch.sum(torch.diag(torch.matmul(m_v_s[i], video_semantic_result[i].T)))
        
        seq_len = audio_semantic_result.shape[1]
        score_a_f /= (batch_dim * seq_len)
        score_v_f /= (batch_dim * seq_len)
        
        score_a_c = torch.sum(torch.diag(torch.matmul(torch.mean(m_a_s, dim=1), torch.mean(audio_semantic_result, dim=1).T))) + \
                   torch.sum(torch.diag(torch.matmul(torch.mean(m_a_s, dim=1), torch.mean(video_semantic_result, dim=1).T)))
        
        score_v_c = torch.sum(torch.diag(torch.matmul(torch.mean(m_v_s, dim=1), torch.mean(audio_semantic_result, dim=1).T))) + \
                   torch.sum(torch.diag(torch.matmul(torch.mean(m_v_s, dim=1), torch.mean(video_semantic_result, dim=1).T)))
        
        score_a_c /= batch_dim
        score_v_c /= batch_dim
        
        """CUJP"""
        emd_dim = audio_semantic_result.shape[-1]
        audio_emd, video_emd = audio_vq.reshape(-1, emd_dim), video_vq.reshape(-1, emd_dim)
        audio_parts = torch.split(audio_emd, audio_emd.shape[1] // args.jigsaw_num_splits, dim=1)
        video_parts = torch.split(video_emd, video_emd.shape[1] // args.jigsaw_num_splits, dim=1)

        parts = ()
        for i in range(args.jigsaw_num_splits):
            random_idx = random.randint(0, 1)
            if random_idx == 0:
                parts += (audio_parts[i],)
            else:
                parts += (video_parts[i],)
        
        all_combinations = list(itertools.permutations(parts, len(parts)))
        all_combinations = [all_combinations[ji] for ji in jigsaw_indices]
        
        jigsaw_labels = []
        combinations = []
        for label, all_parts in enumerate(all_combinations):
            concatenated = torch.cat(all_parts, dim=1)
            jigsaw_labels.append(torch.tensor([label]).repeat(concatenated.shape[0], 1))
            combinations.append(concatenated)
        
        combinations = torch.cat(combinations, dim=0)
        jigsaw_labels = torch.cat(jigsaw_labels, dim=0).squeeze(1).type(torch.LongTensor).cuda()
        
        predict_jigsaw = jigsaw_cls(combinations)
        jigsaw_loss = nn.CrossEntropyLoss()(predict_jigsaw, jigsaw_labels)
        
        if n_iter % 40 == 0:
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
            "acc_av": accuracy1.item(),
            "acc_va": accuracy2.item(),
            "acc_aa": accuracy3.item(),
            "acc_vv": accuracy4.item(),
            "cpc_loss": cpc_loss.item(),
            "cmcm_loss": cmcm_loss.item(),
            'jigsaw_loss': jigsaw_loss.item(),
            'infonce_fine_loss': infonce_fine_loss.item(),
            'infonce_coarse_loss': infonce_coarse_loss.item(),
        }

        metricsContainer.update("loss", loss_items)
        
        """Best model without [cpc, cmcm, mi]"""
        loss = audio_recon_loss + video_recon_loss + audio_embedding_loss + video_embedding_loss \
                + infonce_fine_loss + infonce_coarse_loss + jigsaw_loss
        
        if n_iter % 5 == 0:
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

        global_steps = global_steps + 1
        if global_steps % 200 == 0 and global_steps > 500:
            save_path = os.path.join(args.model_save_path, 'AV-pretrain-mfnce[0.3all]-wo[cpc,jp,cmcm,mi]-step{}.pt'.format(global_steps))
            save_models(CPC, Encoder, epoch, total_step, save_path)

    return losses.avg, n_iter + total_step


def mi_first_forward(Audio_mi_net, Video_mi_net, optimizer_audio_mi_net, optimizer_video_mi_net,
                     audio_encoder_result, video_club_feature,
                     audio_vq, video_vq, mi_iters):
    
    for i in range(mi_iters):
        optimizer_video_mi_net.zero_grad()
        optimizer_audio_mi_net.zero_grad()
        
        lld_video_loss = -Video_mi_net.loglikeli(video_vq, video_club_feature)
        lld_video_loss.backward()
        optimizer_video_mi_net.step()
        
        lld_audio_loss = -Audio_mi_net.loglikeli(audio_vq, audio_encoder_result)
        lld_audio_loss.backward()
        optimizer_audio_mi_net.step()

    return optimizer_audio_mi_net, lld_audio_loss, optimizer_video_mi_net, lld_video_loss


def VQ_audio_forward(audio_feature, visual_feature, Encoder, optimizer, epoch):
    audio_vq_forward_loss = Encoder.Audio_vq_forward(audio_feature, visual_feature, epoch)
    audio_vq_forward_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return audio_vq_forward_loss, optimizer

def VQ_video_forward(audio_feature, visual_feature, Encoder, optimizer, epoch):
    optimizer.zero_grad()
    video_vq_forard_loss = Encoder.Video_vq_forward(audio_feature, visual_feature, epoch)
    video_vq_forard_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return video_vq_forard_loss, optimizer

def mi_second_forward(CPC, audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net, Decoder, epoch):
    video_semantic_result, audio_semantic_result, \
    video_encoder_result, video_club_feature, audio_encoder_result, \
    video_vq, audio_vq, audio_embedding_loss, video_embedding_loss, cmcm_loss, equal_num, \
    audio_perplexity, video_perplexity \
    = Encoder(audio_feature, video_feature, epoch)
    
    mi_video_loss = Video_mi_net.mi_est(video_vq, video_club_feature)
    mi_audio_loss = Audio_mi_net.mi_est(audio_vq, audio_encoder_result)
    
    """Cross_CPC"""
    accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss = CPC(audio_semantic_result, video_semantic_result)
    
    video_recon_loss, audio_recon_loss, video_class, audio_class \
        = Decoder(video_feature, audio_feature, video_encoder_result, audio_encoder_result, video_vq, audio_vq)
    
    return audio_embedding_loss, video_embedding_loss, mi_audio_loss, mi_video_loss, \
           accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss, \
           audio_recon_loss, video_recon_loss, audio_class, video_class, cmcm_loss, equal_num


def compute_accuracy_supervised(event_scores, labels):
    labels_foreground = labels[:, :, :-1]
    labels_BCE, labels_evn = labels_foreground.max(-1)
    labels_event, _ = labels_evn.max(-1)
    _, event_class = event_scores.max(-1)
    correct = event_class.eq(labels_event)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)
    

if __name__ == '__main__':
    main()