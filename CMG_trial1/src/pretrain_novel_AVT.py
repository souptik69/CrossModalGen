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
from model.main_model_novel import AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
from model.CPC import Cross_CPC, Cross_CPC_AVT
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import pickle
from collections import Counter
torch.autograd.set_detect_anomaly(True)

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def collate_func_AVT(samples):
    bsz = len(samples)
    text_prompts = [sample['text_fea'] for sample in samples]
    query = []
    query_words = []
    
    for text in text_prompts:
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        token_ids = inputs.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        non_special_tokens = tokens[1:-1]
        non_special_embeddings = embeddings[1:-1]
        words = []
        words_emb = [] 
        for token, emb in zip(non_special_tokens, non_special_embeddings):
            idx = tokenizer.convert_tokens_to_ids(token)
            if idx in id2idx and idx != 0:
                words_emb.append(emb)
                words.append(id2idx[idx])

        query.append(np.asarray(words_emb))
        query_words.append(words)

    query_len = []
    for i, sample in enumerate(query):
        query_len.append(10)  # max_num_words:10
    
    query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
    query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
    for i, sample in enumerate(query):
        keep = min(sample.shape[0], query1.shape[1])
        """
        There may be cases where the sample length is 0, 
        for example if your text happens to not be seen before in this BERT model. 
        If that happens, you can 
        1) clean the text before it enters BERT, 
        2) add an if statement here, 
        3) discard idx and directly import all embeddings after.
        """
        query1[i, :keep] = sample[:keep]
        query_idx[i, :keep] = query_words[i][:keep]
    query_len = np.asarray(query_len)
    query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
    query_idx = torch.from_numpy(query_idx).long()

    return {
        'query': query,
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
    if args.dataset_name == 'vggsound':
        from dataset.VGG_dataset_novel import VGGSoundDataset_AVT as AVEDataset
    else:
        raise NotImplementedError 

    '''Dataloader Loading'''
    
    meta_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv'
    audio_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/audio80k_features_new'
    video_fea_base_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/video80k_features_keras'
    train_dataloader = DataLoader(
        AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        collate_fn=collate_func_AVT
    )

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
    #     collate_fn=collate_func_AVT
    # )

    ########## 40 k #############

    '''model setting'''
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = False
    # model_resume = True
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim, n_embeddings, embedding_dim)
    # CPC = Cross_CPC(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    CPC = Cross_CPC_AVT(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, video_output_dim)

    Text_ar_lstm.double()
    Encoder.double()
    CPC.double()
    Decoder.double()

    '''optimizer setting'''
    Text_ar_lstm.to(device)
    Encoder.to(device)
    CPC.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(chain(Text_ar_lstm.parameters(), \
                                       Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        # path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVT_model/AudioNoise/90k/checkpoint/DCID-model-3.pt"
        path_checkpoints = ""
        print(path_checkpoints)
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        CPC.load_state_dict(checkpoints['CPC_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
        start_epoch = checkpoints['epoch']
        total_step = checkpoints['total_step']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Training and Evaluation'''
    for epoch in range(start_epoch+1, args.n_epoch):
        loss, total_step = train_epoch(CPC, Encoder, Text_ar_lstm, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, epoch, total_step, args)
        
        
        save_path = os.path.join(args.model_save_path, 'DCID-model-{}.pt'.format(epoch))
        save_models(CPC, Encoder, Text_ar_lstm, Decoder, optimizer, epoch, total_step, save_path)
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


def save_models(CPC, Encoder, Text_ar_lstm, Decoder, optimizer, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'CPC_parameters': CPC.state_dict(),
        'Text_ar_lstm_parameters': Text_ar_lstm.state_dict(),
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch(CPC,Encoder,Text_ar_lstm, Decoder,train_dataloader, criterion, criterion_event, optimizer, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    models = [CPC,Encoder,Text_ar_lstm,Decoder]
    to_train(models)

    Encoder.cuda()
    Text_ar_lstm.cuda()
    Decoder.cuda()
    CPC.cuda()
    optimizer.zero_grad()

    quantizer = Encoder.Cross_quantizer
    with torch.no_grad():
        logger.info(f"INITIALIZATION_1 - Vector 345 (first 5 video dims): {quantizer.embedding[345, :5]}")
        logger.info(f"INITIALIZATION_1 - Vector 345 (first 5 audio dims): {quantizer.embedding[345, 256:261]}")
        logger.info(f"INITIALIZATION_1 - Vector 345 (first 5 text dims): {quantizer.embedding[345, 512:517]}")
        logger.info(f"INITIALIZATION_1 - Vector 346 (first 5 video dims): {quantizer.embedding[346, :5]}")
        logger.info(f"INITIALIZATION_1 - Vector 346 (first 5 audio dims): {quantizer.embedding[346, 256:261]}")
        logger.info(f"INITIALIZATION_1 - Vector 346 (first 5 text dims): {quantizer.embedding[346, 512:517]}")
        logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 video dims): {quantizer.embedding[45, :5]}")
        logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 audio dims): {quantizer.embedding[45, 256:261]}")
        logger.info(f"INITIALIZATION_1 - Vector 45 (first 5 text dims): {quantizer.embedding[45, 512:517]}")
        logger.info(f"Init Codebook min: {quantizer.embedding.min().item()}, max: {quantizer.embedding.max().item()}")
  
    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        # vggsound_AVT
        query, audio_feature, video_feature = batch_data['query'], batch_data['audio_fea'], batch_data['video_fea']
        query = query.double().cuda()

        batch_dim = query.size()[0]
        hidden_dim = 128
        num_layers = 2
        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(query, text_hidden)

        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)

        audio_semantic_result, audio_encoder_result, video_semantic_result, video_spatial, \
        text_semantic_result, text_encoder_result, \
        out_vq_video, video_vq, out_vq_audio, audio_vq,\
        out_vq_text, text_vq, video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, equal_num, cmcm_loss, segment_loss \
        = Encoder(audio_feature, video_feature, text_feature, epoch)

        if n_iter == 0:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                logger.info(f"INITIALIZATION Epoch 0 - Vector 345 (first 5 video dims): {quantizer.embedding[345, :5]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 345 (first 5 audio dims): {quantizer.embedding[345, 256:261]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 345 (first 5 text dims): {quantizer.embedding[345, 512:517]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 346 (first 5 video dims): {quantizer.embedding[346, :5]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 346 (first 5 audio dims): {quantizer.embedding[346, 256:261]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 346 (first 5 text dims): {quantizer.embedding[346, 512:517]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 45 (first 5 video dims): {quantizer.embedding[45, :5]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 45 (first 5 audio dims): {quantizer.embedding[45, 256:261]}")
                logger.info(f"INITIALIZATION Epoch 0 - Vector 45 (first 5 text dims): {quantizer.embedding[45, 512:517]}")
                logger.info(f"INITIALIZATION Epoch 0 - Count for vector 345: {quantizer.ema_count[345]}")
                logger.info(f"INITIALIZATION Epoch 0 - Count for vector 346: {quantizer.ema_count[346]}")
                logger.info(f"INITIALIZATION Epoch 0 - Count for vector 45: {quantizer.ema_count[45]}")
                logger.info(f"Init Codebook min: {quantizer.embedding.min().item()}, max: {quantizer.embedding.max().item()}")

        if n_iter == 20:
            quantizer = Encoder.Cross_quantizer
            with torch.no_grad():
                most_used_idx = torch.argmax(quantizer.ema_count)
                random_idx1 = (most_used_idx + 1) % 400
                random_idx2 = (most_used_idx + 100) % 400
                logger.info("\n===== BATCH 20 DEBUG =====")
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
                logger.info(f"Number of dead vectors (count < 0.01): {(quantizer.ema_count < 0.01).sum()}")


        if n_iter == 40:
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


        if n_iter == 80:
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

        if n_iter == 100:
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

        if n_iter == 120:
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

        if n_iter == 130:
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

        if n_iter == 140:
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

        if n_iter == 320:
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

        if n_iter == 620:
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



        # accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss,  \
        # audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class \
        # = mi_first_forward(CPC, audio_feature, video_feature, text_feature, Decoder,epoch,
        #                 audio_encoder_result, audio_semantic_result, video_spatial, video_semantic_result,
        #                 text_encoder_result, text_semantic_result,
        #                 out_vq_audio, audio_vq, out_vq_video, video_vq,
        #                 out_vq_text, text_vq, criterion_event)

        accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, cpc_loss, \
        audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class \
        = mi_first_forward(CPC, audio_feature, video_feature, text_feature, Decoder,epoch,
                        audio_encoder_result, audio_semantic_result, video_spatial, video_semantic_result,
                        text_encoder_result, text_semantic_result,
                        out_vq_audio, audio_vq, out_vq_video, video_vq,
                        out_vq_text, text_vq, criterion_event)

        if n_iter % 20 == 0:
            logger.info("equal_num is {} in {}-th iteration.".format(equal_num, n_iter))
        
        # loss_items = {
        #     "audio_recon_loss": audio_recon_loss.item(),
        #     "audio_embed_loss": audio_embedding_loss.item(),
        #     "text_recon_loss": text_recon_loss.item(),
        #     "text_embed_loss": text_embedding_loss.item(),
        #     "video_recon_loss": video_recon_loss.item(),
        #     "video_embed_loss": video_embedding_loss.item(),
        #     "acc_va": accuracy1.item(),
        #     "acc_av": accuracy2.item(),
        #     "acc_vv": accuracy3.item(),
        #     "acc_aa": accuracy4.item(),
        #     "cpc_loss": cpc_loss.item(),
        #     "cmcm_loss": cmcm_loss.item(),
        #     "segment_loss": segment_loss.item(),
        #     "audio_perplexity": audio_perplexity.item(),
        #     "video_perplexity": video_perplexity.item(),
        #     "text_perplexity": text_perplexity.item()
        # }

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
        loss =  audio_recon_loss + video_recon_loss + text_recon_loss + audio_embedding_loss +  video_embedding_loss + text_embedding_loss\
                + cpc_loss + cmcm_loss

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

def mi_first_forward(CPC, audio_feature, video_feature, text_feature, Decoder,epoch,
                      audio_encoder_result, audio_semantic_result, video_spatial, 
                      video_semantic_result, text_encoder_result, text_semantic_result, out_vq_audio,
                      audio_vq, out_vq_video, video_vq, out_vq_text, text_vq, criterion_event):

    
    """Cross_CPC"""
    # accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss = CPC(video_semantic_result, audio_semantic_result)

    accuracy1, accuracy2, accuracy3, accuracy4, \
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, \
    cpc_loss = CPC(audio_semantic_result, video_semantic_result, text_semantic_result)


    audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class \
        = Decoder(audio_feature, video_feature, text_feature, audio_encoder_result, video_spatial, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq)
    

    # return accuracy1, accuracy2, accuracy3, accuracy4, cpc_loss,\
    #        audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class

    
    return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, cpc_loss,\
           audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class

   
if __name__ == '__main__':
    main()