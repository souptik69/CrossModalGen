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
# from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
# from model.main_model_2 import AV_VQVAE_Encoder
# from model.main_model_2 import AV_VQVAE_Decoder
from model.main_model_novel import Semantic_Decoder, AVT_VQVAE_Encoder
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from utils.draw import Draw_Heatmap
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from model.retrieval_loss import BiDirectionalRankingLoss
from model.retrieval_utils import t2a, a2t
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pickle
import torch.nn.utils.rnn as rnn_utils
# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================

class Retrieval_Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Retrieval_Decoder, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
    def forward(self, fea1, fea2):
        embed1 = self.linear1(fea1)
        embed2 = self.linear2(fea2)
        return embed1, embed2


with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
    
def collate_func_AT(samples):
    """Audio-Text collate function using transformers library"""
    bsz = len(samples)
    
    # Caption preprocess
    text_prompts = [sample[1] for sample in samples]
    query = []
    
    for text in text_prompts:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Remove [CLS] and [SEP] tokens
        non_special_embeddings = embeddings[1:-1]
        query.append(non_special_embeddings)
    
    max_query_len = 30
    query_len = []
    query1 = np.zeros([bsz, max_query_len, 768]).astype(np.float32)
    
    for i, sample in enumerate(query):
        keep = min(sample.shape[0], query1.shape[1])
        query_len.append(keep)
        if keep > 0:
            query1[i, :keep] = sample[:keep]
    
    query_len = np.asarray(query_len)
    query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
    
    # Audio preprocess
    max_audio_length = max([i[3] for i in samples])
    audio_tensor = []
    
    for audio_fea, _, _, _, _ in samples:
        if max_audio_length > audio_fea.shape[0]:
            padding = torch.zeros(max_audio_length - audio_fea.shape[0], 128).float()
            temp_audio = torch.cat([torch.from_numpy(audio_fea).float(), padding])
        else:
            temp_audio = torch.from_numpy(audio_fea[:max_audio_length]).float()
        audio_tensor.append(temp_audio)
    
    audio_ids = torch.Tensor([i[2] for i in samples])
    audio_len = torch.Tensor([i[3] for i in samples])
    indexs = np.array([i[4] for i in samples])
    
    return torch.stack(audio_tensor), audio_len, query, query_len, audio_ids, indexs


def collate_func_VT(samples):
    """Video-Text collate function using transformers library"""
    bsz = len(samples)
    
    # Caption preprocess
    text_prompts = [sample[1] for sample in samples]
    query = []
    
    for text in text_prompts:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Remove [CLS] and [SEP] tokens
        non_special_embeddings = embeddings[1:-1]
        query.append(non_special_embeddings)
    
    max_query_len = 30
    query_len = []
    query1 = np.zeros([bsz, max_query_len, 768]).astype(np.float32)
    
    for i, sample in enumerate(query):
        keep = min(sample.shape[0], query1.shape[1])
        query_len.append(keep)
        if keep > 0:
            query1[i, :keep] = sample[:keep]
    
    query_len = np.asarray(query_len)
    query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
    
    # Video preprocess
    max_video_length = max([i[3] for i in samples])
    video_tensor = []
    
    for video_fea, _, _, _, _ in samples:
        if max_video_length > video_fea.shape[0]:
            padding = torch.zeros(
                max_video_length - video_fea.shape[0], 
                video_fea.shape[1], 
                video_fea.shape[2], 
                video_fea.shape[3]
            ).float()
            temp_video = torch.cat([torch.from_numpy(video_fea).float(), padding])
        else:
            temp_video = torch.from_numpy(video_fea[:max_video_length]).float()
        video_tensor.append(temp_video)
    
    video_ids = torch.Tensor([i[2] for i in samples])
    video_len = torch.Tensor([i[3] for i in samples])
    indexs = np.array([i[4] for i in samples])
    
    return torch.stack(video_tensor), video_len, query, query_len, video_ids, indexs

def main():
    print('begin')
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    global best_acc,best_rec,best_f1
    global best1_recall, best2_recall
    best1_recall = [0,0,0]
    best2_recall = [0,0,0]
    best_acc=0
    best_rec=0
    best_f1=0
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
    from dataset.Clotho_dataset import ClothoDataset
    from dataset.MSCOCO_dataset import MSCOCODataset
    
    # import pdb
    # pdb.set_trace()
    # print('232')
 

    '''Dataloader selection'''
    if args.dataset_name == 'clotho':
        test_dataloader = DataLoader(
            ClothoDataset('/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/clotho_captions_evaluation.csv'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_func_AT
        ) 
    elif args.dataset_name == 'mscoco':
        test_dataloader = DataLoader(
            MSCOCODataset(),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_func_VT
        ) 
    '''model setting'''
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    n_embeddings = 800
    embedding_dim = 256
    start_epoch = -1
    model_resume = True
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim, n_embeddings, embedding_dim)
    choose_channel = args.choose_channel
    Decoder = Retrieval_Decoder(embedding_dim).double().to(device)
    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)

    Encoder.double()
    Decoder.double()
    Encoder.to(device)
    Decoder.to(device)
    Text_ar_lstm = Text_ar_lstm.double().to(device)
    optimizer = torch.optim.Adam(Decoder.parameters(), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = BiDirectionalRankingLoss(margin=0.2).cuda()# 0.2沿用自audio-text_retrieval
    criterion_event = nn.CrossEntropyLoss().cuda()
    
    # import pdb
    # pdb.set_trace()
    # print('309')


    if model_resume is True:
        path_checkpoints = args.checkpoint_path
        print(path_checkpoints)
        print('dataset:',args.dataset_name)
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
        start_epoch = checkpoints['epoch']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Tensorboard and Code backup'''
    # writer = SummaryWriter(args.snapshot_pref)
    # recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    # recorder.writeopt(args)

    '''Training and Evaluation'''


    """选择部分channel"""
    
    # indices = cal_criterion(Encoder.Cross_quantizer.embedding.cuda(), choose_channel, args.toc_max_num, args.toc_min_num)
    
    # indices = range(256)
    # print(indices)

    for epoch in range(start_epoch+1, args.n_epoch):
        
        # loss, total_step = train_epoch(Encoder, Decoder, Text_ar_lstm, train_dataloader, criterion,
        #                                optimizer, epoch, total_step, args, stage)
        # logger.info(f"epoch: *******************************************{epoch}")

        # if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
        #     new_best = validate_epoch(Encoder, Decoder, Text_ar_lstm,  val_dataloader, criterion, epoch, args, stage, val_test='val')
            # if new_best:
            #     validate_epoch(Encoder, Decoder, Text_ar_lstm,  test_dataloader, criterion, epoch, args, stage, val_test='test')


        if args.dataset_name == 'clotho':
            num = len(test_dataloader.dataset)//5
            fea_cap_num = [5 for _ in range(num)]
        elif args.dataset_name == 'mscoco':
            num = len(test_dataloader.dataset)//5
            fea_cap_num = [5 for _ in range(num)]
        validate_epoch(Encoder, Decoder, Text_ar_lstm,  test_dataloader, criterion, epoch, args, fea_cap_num, val_test='test')
        return 
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

def save_models(Encoder, optimizer, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step,
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch_check(train_dataloader, epoch, total_step, args):
    # train_dataloader = tqdm(train_dataloader)
    for n_iter, batch_data in enumerate(train_dataloader):
        
        '''Feed input to model'''
        feature, labels, mask = batch_data['feature'],batch_data['label'],batch_data['mask']
    return torch.zeros(1),torch.zeros(1)

"""先量化编码后取mean"""
@torch.no_grad()
def validate_epoch(Encoder,Decoder, Text_ar_lstm, val_dataloader, criterion, epoch, args, fea_cap_num,val_test, eval_only=False):
    Encoder.eval()
    Decoder.eval()
    Text_ar_lstm.eval()

    global best1_recall, best2_recall
    new_best = False

    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        fea_embs, cap_embs = None, None

        if (args.dataset_name in ['mscoco']):
            # print('vt_ret_begin')
            for n_iter, batch_data in enumerate(val_dataloader):
                
                # if n_iter % 10 == 0:
                #     print('n_iter:', n_iter)

                v_feature, fea_len, query, query_len, sample_ids, indexs = batch_data
                query = query.double().cuda()
                batch_dim = query.size()[0]
                hidden_dim = 128
                num_layers = 2
                # text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                #         torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
                # t_feature, text_hidden = Text_ar_lstm(query, text_hidden)

                """new"""
                packed_input = rnn_utils.pack_padded_sequence(query, query_len, batch_first=True, enforce_sorted=False).cuda().double()
                packed_output, _ = Text_ar_lstm(packed_input)
                lstm_out, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
                B, L, embed_dim = lstm_out.shape
                t_feature = torch.zeros(B, 1, embed_dim).cuda()
                for i in range(B):
                    t_feature[i,0,:] = torch.mean(lstm_out[i,:query_len[i]], dim=0, keepdim = False)
                t_feature = t_feature.to(torch.float64).cuda()


                v_feature = v_feature.to(torch.float64).cuda()

                
                with torch.no_grad():
                    video_vq, vq_v = Encoder.Video_VQ_Encoder(v_feature)
                    # video_vq = video_vq[:,:,indices]

                    new_video_vq = torch.zeros(video_vq.shape[0],video_vq.shape[2]).cuda().to(torch.float64)

                    for i in range(fea_len.size(0)):
                        l = int(fea_len[i])
                        new_video_vq[i] = torch.mean(video_vq[i, :l, :], dim = 0, keepdim =False)
                        # new_video_vq[i] = video_vq[i, l-1, :]

                    text_vq, vq_t = Encoder.Text_VQ_Encoder(t_feature)
                    # text_vq = text_vq[:,:,indices]
                    text_vq = torch.mean(text_vq, dim=1, keepdim = False)
                    # text_vq = text_vq[:,-1,:]

                # embeds1, embeds2 = Decoder(new_video_vq, text_vq)
                embeds1, embeds2 = new_video_vq, text_vq

                if fea_embs is None:
                    fea_embs = np.zeros((len(val_dataloader.dataset), embeds1.size(1)))
                    cap_embs = np.zeros((len(val_dataloader.dataset), embeds2.size(1)))

                fea_embs[indexs] = embeds1.cpu().numpy()
                cap_embs[indexs] = embeds2.cpu().numpy()

            #
            # video_cap_num = get_MSVD_cap_num()

            # evaluate text to audio retrieval
            r1, r5, r10, r50, medr, meanr = t2a(fea_embs, cap_embs, fea_cap_num)

            logger.info('Caption to video: r1: {:.2f}, r5: {:.2f}, '
                            'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                            r1, r5, r10, r50, medr, meanr))
            


            # evaluate audio to text retrieval
            r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(fea_embs, cap_embs, fea_cap_num)

            logger.info('Video to caption: r1: {:.2f}, r5: {:.2f}, '
                            'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                            r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a))
            

        elif args.dataset_name in ['clotho']:
            for n_iter, batch_data in enumerate(val_dataloader):
                a_feature, fea_len, query, query_len, sample_ids, indexs = batch_data
                
                query = query.double().cuda()
                batch_dim = query.size()[0]
                hidden_dim = 128
                num_layers = 2
                # text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                #         torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
                # t_feature, text_hidden = Text_ar_lstm(query, text_hidden)
                """new"""
                packed_input = rnn_utils.pack_padded_sequence(query, query_len, batch_first=True, enforce_sorted=False).cuda().double()
                packed_output, _ = Text_ar_lstm(packed_input)
                lstm_out, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
                B, L, embed_dim = lstm_out.shape
                t_feature = torch.zeros(B, 1, embed_dim).cuda()
                for i in range(B):
                    t_feature[i,0,:] = torch.mean(lstm_out[i,:query_len[i]], dim=0, keepdim = False)
                t_feature = t_feature.to(torch.float64).cuda()


                a_feature = a_feature.to(torch.float64).cuda()

                
                with torch.no_grad():
                    audio_vq, vq_a = Encoder.Audio_VQ_Encoder(a_feature)
                    # audio_vq = audio_vq[:,:,indices]
                    new_audio_vq = torch.zeros(audio_vq.shape[0],audio_vq.shape[2]).cuda().to(torch.float64)

                    for i in range(fea_len.size(0)):
                        l = int(fea_len[i])
                        new_audio_vq[i] = torch.mean(audio_vq[i, :l, :], dim = 0, keepdim =False)

                    text_vq, vq_t_1 = Encoder.Text_VQ_Encoder(t_feature)
                    # text_vq = text_vq[:,:,indices]
                    text_vq = torch.mean(text_vq, dim=1, keepdim = False)

                # embeds1, embeds2 = Decoder(new_audio_vq, text_vq)
                embeds1, embeds2 = new_audio_vq, text_vq

                if fea_embs is None:
                    fea_embs = np.zeros((len(val_dataloader.dataset), embeds1.size(1)))
                    cap_embs = np.zeros((len(val_dataloader.dataset), embeds2.size(1)))

                fea_embs[indexs] = embeds1.cpu().numpy()
                cap_embs[indexs] = embeds2.cpu().numpy()
                # print(indexs)


            r1, r5, r10, r50, medr, meanr = t2a(fea_embs, cap_embs, fea_cap_num)
            logging.info('current state: ', val_test)
            logger.info('Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                            'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                            r1, r5, r10, r50, medr, meanr))
            
            # evaluate audio to text retrieval
            r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a = a2t(fea_embs, cap_embs,  fea_cap_num)

            logger.info('Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                            'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                            r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a))
            

            if val_test=='val' and r1 + r5 + r10 + r1_a + r5_a + r10_a > best1_recall[0] + best1_recall[1] + best1_recall[2] + best2_recall[0] + best2_recall[1] + best2_recall[2]:
                new_best = True
                best1_recall[0] = r1
                best1_recall[1] = r5
                best1_recall[2] = r10
                best2_recall[0] = r1_a
                best2_recall[1] = r5_a
                best2_recall[2] = r10_a
                logger.info('best t2a: r1: {:.2f}, r5: {:.2f}, '
                            'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                            r1, r5, r10, r50, medr, meanr))
                logger.info('best a2t: r1: {:.2f}, r5: {:.2f}, '
                            'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                            r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a))


# def cal_criterion(feats, choose_channel, max_num, min_num):
#     import time
#     start_time = time.time()
    
    
#     code_num, code_dim = feats.shape
    
#     sim_sum = torch.zeros((code_dim)).cuda()
#     count = 0
#     for i in range(code_num):
#         for j in range(code_num):
#             if i != j:
#                 sim_sum += feats[i, :] * feats[j, :]
#                 count += 1
#     sim = sim_sum / count
    
#     criterion = (-0.7) * sim + 0.3 * torch.var(feats, dim=0)
#     # criterion = (-0.7) * sim
#     # criterion = 0.3 * torch.var(feats, dim=0)
    
#     end_time = time.time()
#     print('TOC消耗时间: %s Seconds'%(end_time-start_time))



#     _, max_indices = torch.topk(criterion, k=choose_channel//int(max_num+min_num)*int(max_num))
#     print(max_indices)
#     _, min_indices = torch.topk(criterion, k=choose_channel//int(max_num+min_num)*int(min_num), largest=False)
#     print(min_indices)
#     indices = torch.cat((max_indices, min_indices),dim=0)
#     # print(indices)
#     return indices


# def get_MSVD_cap_num():
#     import pandas as pd
#     df = pd.read_csv('../MSVD/MSVD/test.csv')
            
#     # 计算每个video_name的caption数量
#     counts = df.groupby('video_name').size()
#     caption_counts = []
#     # 添加到列表
#     caption_counts.extend(counts.values.tolist())
#     # print(caption_counts[0], caption_counts[1])
#     return caption_counts


if __name__ == '__main__':
    main()
    # get_MSVD_cap_num()
