import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np
# from preprocess import mulaw_decode
import math
from torch.nn import MultiheadAttention
from model.models import EncoderLayer, Encoder, DecoderLayer
from torch import Tensor
# The model is testing
from model.mine import MINE
from info_nce import InfoNCE
import random
random.seed(123)
import logging
logger = logging.getLogger()  # Get the root logger

class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)
        return feature


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x



class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Video_Encoder(nn.Module):
    def __init__(self, video_dim, hidden_dim):
        super(Video_Encoder, self).__init__()
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.video_linear = nn.Linear(video_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, video_feat):
        return self.relu(self.video_linear(video_feat))

class Audio_Encoder(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Audio_Encoder, self).__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feat):
        return self.relu(self.audio_linear(audio_feat))


class Text_Encoder(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super(Text_Encoder, self).__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text_feat):
        return self.relu(self.text_linear(text_feat))


class AVT_VQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, n_embeddings, embedding_dim):
        super(AVT_VQVAE_Encoder, self).__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = 256
        self.Cross_quantizer = Cross_VQEmbeddingEMA_AVT_hierarchical(n_embeddings, self.hidden_dim)
        self.Video_encoder = Video_Encoder(video_dim, self.hidden_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, self.hidden_dim)
        self.Text_encoder = Text_Encoder(text_dim, self.hidden_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)
        self.text_self_att = InternalTemporalRelationModule(input_dim=text_dim, d_model=self.hidden_dim)


    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()  
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result) 
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous() # [batch, 10, 256]
        out_vq, audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        return out_vq, audio_vq  # [batch, 10, 768], [batch, 10, 256]


    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = video_feat.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result) 
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous() # [batch, 10, 256]
        out_vq, video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        return out_vq, video_vq  # [batch,10, 768], [batch, 10, 256]


    def Text_VQ_Encoder(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, 10, 256]
        out_vq, text_vq = self.Cross_quantizer.Text_vq_embedding(text_semantic_result)
        return out_vq, text_vq  # [batch,10, 768], [batch, 10, 256]



    def forward(self, audio_feat, video_feat, text_feat, epoch):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        
        video_semantic_result = video_feat.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]
        
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        video_encoder_result = self.Video_encoder(video_feat)
        audio_encoder_result = self.Audio_encoder(audio_feat)
        text_encoder_result = self.Text_encoder(text_feat)

        out_vq_video, video_vq, out_vq_audio, audio_vq, \
        out_vq_text, text_vq, \
        video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, \
        equal_num, cmcm_loss, segment_loss = self.Cross_quantizer(audio_semantic_result, video_semantic_result, text_semantic_result, epoch)

        return audio_semantic_result, audio_encoder_result, video_semantic_result, video_encoder_result, \
               text_semantic_result, text_encoder_result, \
               out_vq_video, video_vq, out_vq_audio, audio_vq,\
               out_vq_text, text_vq, video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
               video_perplexity, audio_perplexity, text_perplexity, equal_num, cmcm_loss, segment_loss


""" Sentiment Downstream Decoder for Sentiment Label regression """
class Sentiment_Decoder(nn.Module):
    def __init__(self, input_dim):
        super(Sentiment_Decoder, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sentiment_regressor = nn.Linear(input_dim, 1)  # Single continuous output
        
    def forward(self, input_vq):
        input_feat = self.linear(input_vq)
        input_feat, _ = input_feat.max(1)  # Temporal pooling for sequence
        sentiment_score = self.sentiment_regressor(input_feat)  # Continuous sentiment
        return sentiment_score
    

""" Sentiment Downstream Decoder for Sentiment Label regression """
class Sentiment_Decoder_class(nn.Module):
    def __init__(self, input_dim, num_classes= 7):
        super(Sentiment_Decoder_class, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sentiment_regressor = nn.Linear(input_dim, num_classes)  # Single continuous output
        
    def forward(self, input_vq):
        input_feat = self.linear(input_vq)
        input_feat, _ = input_feat.max(1)  # Temporal pooling for sequence
        sentiment_score = self.sentiment_regressor(input_feat)  # Continuous sentiment
        return sentiment_score


""" Sentiment Downstream Decoder for Sentiment Label regression combined """
class Sentiment_Decoder_Combined(nn.Module):
    def __init__(self, input_dim):
        super(Sentiment_Decoder_Combined, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sentiment_regressor = nn.Linear(input_dim, 1)  # Single continuous output
        
    def forward(self, video_vq, audio_vq, text_vq):
        # input_vq =  video_vq + audio_vq + text_vq
        input_vq =  (video_vq + audio_vq + text_vq)/3
        input_feat = self.linear(input_vq)
        input_feat, _ = input_feat.max(1)  # Temporal pooling for sequence
        sentiment_score = self.sentiment_regressor(input_feat)  # Continuous sentiment
        return sentiment_score
    
""" Sentiment Downstream Decoder for Sentiment Label regression combined """
class Sentiment_Decoder_Combined_class(nn.Module):
    def __init__(self, input_dim, num_classes=7):
        super(Sentiment_Decoder_Combined_class, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.sentiment_regressor = nn.Linear(input_dim, num_classes)  # Single continuous output
        
    def forward(self, video_vq, audio_vq, text_vq):
        # input_vq =  video_vq + audio_vq + text_vq
        input_vq =  (video_vq + audio_vq + text_vq)/3
        input_feat = self.linear(input_vq)
        input_feat, _ = input_feat.max(1)  # Temporal pooling for sequence
        sentiment_score = self.sentiment_regressor(input_feat)  # Continuous sentiment
        return sentiment_score



class Audio_Decoder(nn.Module):
    def __init__(self, output_dim , vq_dim):
        super(Audio_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.audio_rec = nn.Linear(vq_dim * 2, output_dim)
        self.audio_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.audio_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, audio_encoder_result, audio_vq):
        audio_vq_1 = self.audio_linear_1(audio_vq)
        audio_vq_result = self.audio_linear(audio_vq_1)
        audio_encoder_result = torch.cat([audio_vq_result, audio_encoder_result], dim=2)
        audio_decoder_result = self.audio_rec(audio_encoder_result)
        return audio_decoder_result


class Text_Decoder(nn.Module):
    def __init__(self, output_dim, vq_dim):
        super(Text_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.text_rec = nn.Linear(vq_dim * 2, output_dim)
        self.text_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.text_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, text_encoder_result, text_vq):
        text_vq_1 = self.text_linear_1(text_vq)
        text_vq_result = self.text_linear(text_vq_1)
        text_encoder_result = torch.cat([text_vq_result, text_encoder_result], dim=2)
        text_decoder_result = self.text_rec(text_encoder_result)
        return text_decoder_result


class Video_Decoder(nn.Module):
    def __init__(self, output_dim, vq_dim):
        super(Video_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.video_rec = nn.Linear(vq_dim * 2, output_dim)
        self.video_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.video_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, video_encoder_result, video_vq):
        video_vq_1 = self.video_linear_1(video_vq)
        video_vq_result = self.video_linear(video_vq_1)
        video_encoder_result = torch.cat([video_vq_result, video_encoder_result], dim=2)
        video_decoder_result = self.video_rec(video_encoder_result)
        return video_decoder_result

'''Decoder for both uni-modal feature reconstruction. To be used for unsupervised pre-training'''
class AVT_VQVAE_Decoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim):
        super(AVT_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 256 #embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.Video_decoder = Video_Decoder(video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder(audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_dim, self.hidden_dim)

        self.video_sentiment_decoder = Sentiment_Decoder(self.hidden_dim * 3)
        self.audio_sentiment_decoder = Sentiment_Decoder(self.hidden_dim * 3)
        self.text_sentiment_decoder = Sentiment_Decoder(self.hidden_dim * 3)
        self.combined_sentiment_decoder = Sentiment_Decoder_Combined(self.hidden_dim * 3)


    def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_encoder_result, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, out_vq_video)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio)
        text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)

        video_score = self.video_sentiment_decoder(out_vq_video)
        audio_score = self.audio_sentiment_decoder(out_vq_audio)
        text_score = self.text_sentiment_decoder(out_vq_text)
        combined_score = self.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)


        return audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score, combined_score


'''Decoder for uni-modal sentiment regression and reconstruction. To be used for supervised training for uni-modal use case'''
class AVT_VQVAE_Decoder_modal(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim):
        super(AVT_VQVAE_Decoder_modal, self).__init__()
        self.hidden_dim = 256 #embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.Video_decoder = Video_Decoder(video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder(audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_dim, self.hidden_dim)
        
        self.video_sentiment_decoder = Sentiment_Decoder(self.hidden_dim * 3)
        self.audio_sentiment_decoder = Sentiment_Decoder(self.hidden_dim * 3)
        self.text_sentiment_decoder = Sentiment_Decoder(self.hidden_dim * 3)


    def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_encoder_result, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, out_vq_video)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio)
        text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)

        video_score = self.video_sentiment_decoder(out_vq_video)
        audio_score = self.audio_sentiment_decoder(out_vq_audio)
        text_score = self.text_sentiment_decoder(out_vq_text)

        return audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score


'''Decoder for combined sentiment regression and reconstruction. To be used for supervised training for combined/multi-modal use case'''
class AVT_VQVAE_Decoder_combined(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim):
        super(AVT_VQVAE_Decoder_combined, self).__init__()
        self.hidden_dim = 256 #embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.Video_decoder = Video_Decoder(video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder(audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_dim, self.hidden_dim)
        self.combined_sentiment_decoder = Sentiment_Decoder_Combined(self.hidden_dim * 3)


    def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_encoder_result, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, out_vq_video)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio)
        text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)

        combined_score = self.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)


        return audio_recon_loss, video_recon_loss, text_recon_loss, combined_score


class Cross_VQEmbeddingEMA_AVT_hierarchical(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AVT_hierarchical, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim * 3)
        
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_v", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_a", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_t", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))# unactivated:-1



    def Video_vq_embedding(self, video_semantic):
        B, T, D = video_semantic.size()  # D is 256
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]

        video_embedding = self.embedding[:, :D]  # [n_embeddings, 256]

        v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                v_flat, video_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT]

        v_quantized = F.embedding(v_indices, video_embedding)  # [BxT, 256]
        v_quantized = v_quantized.view_as(video_semantic)  # [B, T, 256]

        v_quantized = video_semantic + (v_quantized - video_semantic).detach()

        out_vq = F.embedding(v_indices, self.embedding)  # [BxT, 768]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 768]
        
        return out_vq, v_quantized   # [batch,10, 768], [batch, 10, 256]

    def Audio_vq_embedding(self, audio_semantic):
        B, T, D = audio_semantic.size()  # D = 256
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
     
        audio_embedding = self.embedding[:, D:2*D]  # [n_embeddings, 256]

        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT]

        a_quantized = F.embedding(a_indices, audio_embedding)  # [BxT, 256]
        a_quantized = a_quantized.view_as(audio_semantic)  # [B, T, 256]

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()

        out_vq = F.embedding(a_indices, self.embedding)  # [BxT, 768]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 768]
        
        return out_vq, a_quantized   # [batch,10, 768], [batch, 10, 256]


    def Text_vq_embedding(self, text_semantic):
        B, T, D = text_semantic.size()
        t_flat = text_semantic.detach().reshape(-1, D) # [B, T, D] -> [BxT, D]

        text_embedding = self.embedding[:, 2*D:]  # [n_embeddings, 256]

        t_distances = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                t_flat, text_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        t_indices = torch.argmin(t_distances.double(), dim=-1)  # [BxT]

        t_quantized = F.embedding(t_indices, text_embedding)  # [BxT, 256]
        t_quantized = t_quantized.view_as(text_semantic)  # [B, T, 256]

        t_quantized = text_semantic + (t_quantized - text_semantic).detach()

        out_vq = F.embedding(t_indices, self.embedding)  # [BxT, 768]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 768]
        
        return out_vq, t_quantized   # [batch,10, 768], [batch, 10, 256]




    def forward(self, audio_semantic, video_semantic, text_semantic, epoch):
        M, D_total = self.embedding.size()  # M = num codebook vectors, D_total = 768
        B, T, D = audio_semantic.size()     # B = batch size, T = timesteps, D = 256



        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        t_flat = text_semantic.detach().reshape(-1, D) # [B, T, D] -> [BxT, D]
        
        # CODEBOOK DIMENSION SPLIT FOR MODALITY
        video_embedding = self.embedding[:, :D]         # First 256 dims for video
        audio_embedding = self.embedding[:, D:2*D]      # Second 256 dims for audio
        text_embedding = self.embedding[:, 2*D:]        # Last 256 dims for text

        v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                v_flat, video_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]
        
        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        t_distances = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                t_flat, text_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                        torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        video_semantic.reshape(-1, D), video_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]
        
        a_distances_gradient = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                        torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        audio_semantic.reshape(-1, D), audio_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]

        t_distances_gradient = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                        torch.sum(text_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        text_semantic.reshape(-1, D), text_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]

        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M]
        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M]
        t_ph = F.softmax(-torch.sqrt(t_distances_gradient), dim=1)  # [BxT, M]

        v_ph = torch.reshape(v_ph, (B, T, M))  # [BxT, M] -> [B, T, M]
        a_ph = torch.reshape(a_ph, (B, T, M))  # [BxT, M] -> [B, T, M]
        t_ph = torch.reshape(t_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]

        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        t_pH = torch.mean(t_ph, dim=1)  # [B, T, M] -> [B, M]

       

        Scode_av = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)
        Scode_at = a_pH @ torch.log(t_pH.t() + 1e-10) + t_pH @ torch.log(a_pH.t() + 1e-10)
        Scode_tv = t_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(t_pH.t() + 1e-10)


        MaxScode_av = torch.max(-Scode_av)
        EScode_av = torch.exp(Scode_av + MaxScode_av)

        MaxScode_at = torch.max(-Scode_at)
        EScode_at = torch.exp(Scode_at + MaxScode_at)
        
        MaxScode_tv = torch.max(-Scode_tv)
        EScode_tv = torch.exp(Scode_tv + MaxScode_tv)

        EScode_sumdim1_av = torch.sum(EScode_av, dim=1)
        Lcmcm_av = 0

        EScode_sumdim1_at = torch.sum(EScode_at, dim=1)
        Lcmcm_at = 0
        
        EScode_sumdim1_tv = torch.sum(EScode_tv, dim=1)
        Lcmcm_tv = 0

        for i in range(B):
            Lcmcm_av -= torch.log(EScode_av[i, i] / (EScode_sumdim1_av[i] + self.epsilon))
            Lcmcm_at -= torch.log(EScode_at[i, i] / (EScode_sumdim1_at[i] + self.epsilon))
            Lcmcm_tv -= torch.log(EScode_tv[i, i] / (EScode_sumdim1_tv[i] + self.epsilon))

        Lcmcm_av /= B
        Lcmcm_at /= B
        Lcmcm_tv /= B

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT]
        t_indices = torch.argmin(t_distances.double(), dim=-1)  # [BxT,1]

        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        t_encodings = F.one_hot(t_indices, M).double()  # [BxT, M]

        v_quantized_segment = F.embedding(v_indices, video_embedding)  # [BxT, 256]
        a_quantized_segment = F.embedding(a_indices, audio_embedding)  # [BxT, 256]
        t_quantized_segment = F.embedding(t_indices, text_embedding)  # [BxT, 256]

        v_quantized_segment = v_quantized_segment.view_as(video_semantic)  # [B, T, 256]
        a_quantized_segment = a_quantized_segment.view_as(audio_semantic)  # [B, T, 256]
        t_quantized_segment = t_quantized_segment.view_as(text_semantic)  # [B, T, 256]

        v_full_vectors = F.embedding(v_indices, self.embedding)  # [BxT, 768]
        a_full_vectors = F.embedding(a_indices, self.embedding)  # [BxT, 768]
        t_full_vectors = F.embedding(t_indices, self.embedding)  # [BxT, 768]

        v_full_vectors = v_full_vectors.view(B, T, D_total)  # [B, T, 768]
        a_full_vectors = a_full_vectors.view(B, T, D_total)  # [B, T, 768]
        t_full_vectors = t_full_vectors.view(B, T, D_total)  # [B, T, 768]

        if True:
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_reshape = a_indices.reshape(B, T)
            t_indices_reshape = t_indices.reshape(B, T)
            
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            t_indices_mode = torch.mode(t_indices_reshape, dim=-1, keepdim=False)
            
            equal_item = (a_indices_mode.values == v_indices_mode.values) & (a_indices_mode.values == t_indices_mode.values)
            equal_num = equal_item.sum()
        
        if self.training:

            # STEP 1: FEATURE-LEVEL CROSS-MODAL INFLUENCE
            
            # Video segment update (first third of embedding)
            self.ema_count_v = self.decay * self.ema_count_v + (1 - self.decay) * (torch.sum(v_encodings, dim=0))
            n_v = torch.sum(self.ema_count_v)
            self.ema_count_v = (self.ema_count_v + self.epsilon) / (n_v + M * self.epsilon) * n_v

            v_dw = torch.matmul(v_encodings.t(), v_flat)    # Video encodings with Video features → video segment
            a_dw_v = torch.matmul(v_encodings.t(), a_flat)  # Video encodings with Audio features → video segment
            t_dw_v = torch.matmul(v_encodings.t(), t_flat)  # Video encodings with Text features → video segment

            v_segment_update = (0.6 * v_dw) +  (0.2* a_dw_v) + (0.2* t_dw_v)
            with torch.no_grad():
                new_embedding_v = self.embedding.clone()
                self.ema_weight[:, :D] = self.decay * self.ema_weight[:, :D] + (1 - self.decay) * v_segment_update
                new_embedding_v[:, :D] = self.ema_weight[:, :D] / (self.ema_count_v.unsqueeze(-1))
                self.embedding = new_embedding_v

            
            # Audio segment update (second third of embedding)
            self.ema_count_a = self.decay * self.ema_count_a + (1 - self.decay) * (torch.sum(a_encodings, dim=0))
            n_a = torch.sum(self.ema_count_a)
            self.ema_count_a = (self.ema_count_a + self.epsilon) / (n_a + M * self.epsilon) * n_a

            a_dw = torch.matmul(a_encodings.t(), a_flat)   # Audio encodings with Audio features → audio segment
            v_dw_a = torch.matmul(a_encodings.t(), v_flat) # Audio encodings with Video features → audio segment
            t_dw_a = torch.matmul(a_encodings.t(), t_flat) # Audio encodings with Text features → audio segment

            a_segment_update = (0.2 * v_dw_a) + (0.2 * t_dw_a) + (0.6 * a_dw)
            with torch.no_grad():
                new_embedding_a = self.embedding.clone()
                self.ema_weight[:, D:2*D] = self.decay * self.ema_weight[:, D:2*D] + (1 - self.decay) * a_segment_update
                new_embedding_a[:, D:2*D] = self.ema_weight[:, D:2*D] / (self.ema_count_a.unsqueeze(-1))
                self.embedding = new_embedding_a


            # Text segment update (final third of embedding)
            self.ema_count_t = self.decay * self.ema_count_t + (1 - self.decay) * (torch.sum(t_encodings, dim=0))
            n_t = torch.sum(self.ema_count_t)
            self.ema_count_t = (self.ema_count_t + self.epsilon) / (n_t + M * self.epsilon) * n_t

            t_dw = torch.matmul(t_encodings.t(), t_flat)   # Text encodings with Text features → text segment
            v_dw_t = torch.matmul(t_encodings.t(), v_flat) # Text encodings with Video features → text segment
            a_dw_t = torch.matmul(t_encodings.t(), a_flat) # Text encodings with Audio features → text segment

            t_segment_update = (0.2 * v_dw_t) + (0.2 * a_dw_t) + (0.6 * t_dw)
            with torch.no_grad():
                new_embedding_t = self.embedding.clone()
                self.ema_weight[:, 2*D:] = self.decay * self.ema_weight[:, 2*D:] + (1 - self.decay) * t_segment_update
                new_embedding_t[:, 2*D:] = self.ema_weight[:, 2*D:] / (self.ema_count_t.unsqueeze(-1))
                self.embedding = new_embedding_t


            # STEP 2: HIERARCHICAL INFLUENCE BETWEEN SEGMENTS

            with torch.no_grad():
                new_embedding = self.embedding.clone()

                '''T -> A -> V''' 
                # new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, 2*D:])  #Audio = (4/9)Video + (4/9)Audio + (1/9)Text
                # new_embedding[:, :D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])     #Video = (16/27)Video + (7/27)Audio + (4/27)Text

                '''T -> V -> A''' 
                # new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])  #Audio = (4/9)Video + (4/9)Audio + (1/9)Text
                # new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])     #Video = (16/27)Video + (7/27)Audio + (4/27)Text

                '''V -> A -> T'''
                new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])

                '''A -> V -> T'''
                # new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                # new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                # new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])

                '''Equal Distribution A=V=T, Same vector repreated thrice for D dimensions'''
                # video_embedding_new = new_embedding[:, :D]         
                # audio_embedding_new = new_embedding[:, D:2*D] 
                # text_embedding_new = new_embedding[:, 2*D:]

                # text_emb = ((1/3) * video_embedding_new) + ((1/3) * audio_embedding_new) + ((1/3) * text_embedding_new)   
                # audio_emb = ((1/3) * video_embedding_new) + ((1/3) * audio_embedding_new) + ((1/3) * text_embedding_new)   
                # video_emb = ((1/3) * video_embedding_new) + ((1/3) * audio_embedding_new) + ((1/3) * text_embedding_new)    

                # new_embedding[:, 2*D:] = text_emb    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, D:2*D] = audio_emb  #Audio = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, :D] = video_emb     #Video = (1/3)Video + (1/3)Audio + (1/3)Text



                self.embedding = new_embedding


            # SEGMENT ALIGNMENT Metric
            new_embedding = self.embedding.clone()
            video_embedding_new = new_embedding[:, :D]         
            audio_embedding_new = new_embedding[:, D:2*D] 
            text_embedding_new = new_embedding[:, 2*D:]
            video_segments_norm = F.normalize(video_embedding_new, p=2, dim=1)  # [M, D]
            audio_segments_norm = F.normalize(audio_embedding_new, p=2, dim=1)  # [M, D]
            text_segments_norm = F.normalize(text_embedding_new, p=2, dim=1)    # [M, D]
            temperature = 0.1
            # Video vs Audio
            va_similarity = torch.matmul(video_segments_norm, audio_segments_norm.t()) / temperature
            va_positive = torch.diag(va_similarity)
            va_logsumexp = torch.logsumexp(va_similarity, dim=1)
            va_loss = torch.mean(-va_positive + va_logsumexp)
            # Video vs Text
            vt_similarity = torch.matmul(video_segments_norm, text_segments_norm.t()) / temperature
            vt_positive = torch.diag(vt_similarity)
            vt_logsumexp = torch.logsumexp(vt_similarity, dim=1)
            vt_loss = torch.mean(-vt_positive + vt_logsumexp)
            # Audio vs Text
            at_similarity = torch.matmul(audio_segments_norm, text_segments_norm.t()) / temperature
            at_positive = torch.diag(at_similarity)
            at_logsumexp = torch.logsumexp(at_similarity, dim=1)
            at_loss = torch.mean(-at_positive + at_logsumexp)
            segment_loss_raw = (va_loss + vt_loss + at_loss) / 3
            segment_loss = 0.5 * segment_loss_raw


            self.ema_count = self.ema_count_v + self.ema_count_a + self.ema_count_t

            # Dead Codebook vectors Alleviation 
            self.unactivated_count = self.unactivated_count + 1
            for indice in a_indices:
                self.unactivated_count[indice.item()] = 0
            for indice in v_indices:
                self.unactivated_count[indice.item()] = 0
            for indice in t_indices:
                self.unactivated_count[indice.item()] = 0
            activated_indices = []
            unactivated_indices = []
            for i, x in enumerate(self.unactivated_count):
                if x > 160:  # Dead for too long
                    unactivated_indices.append(i)
                    self.unactivated_count[i] = 0
                elif x >= 0 and x < 100:  # Recently active
                    activated_indices.append(i)
            if activated_indices and unactivated_indices:
                activated_quantized = F.embedding(
                    torch.tensor(activated_indices, dtype=torch.int32, device=self.embedding.device), 
                    self.embedding
                )
                for i in unactivated_indices:
                    random_idx = random.randint(0, len(activated_indices)-1)
                    self.embedding[i] = activated_quantized[random_idx] + torch.randn_like(self.embedding[i]) * 0.001

        cmcm_loss = 0.5 * (Lcmcm_av + Lcmcm_at + Lcmcm_tv)

        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized_segment.detach())
        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized_segment.detach())
        vt_e_latent_loss = F.mse_loss(video_semantic, t_quantized_segment.detach())
        v_loss = (self.commitment_cost * 2.0 * v_e_latent_loss) + (0.5*self.commitment_cost * va_e_latent_loss) + (0.5*self.commitment_cost * vt_e_latent_loss)

        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized_segment.detach())
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized_segment.detach())
        at_e_latent_loss = F.mse_loss(audio_semantic, t_quantized_segment.detach())
        a_loss = (self.commitment_cost * 2.0 * a_e_latent_loss) + (0.5*self.commitment_cost * av_e_latent_loss) + (0.5*self.commitment_cost * at_e_latent_loss)


        t_e_latent_loss = F.mse_loss(text_semantic, t_quantized_segment.detach())
        ta_e_latent_loss = F.mse_loss(text_semantic, a_quantized_segment.detach())
        tv_e_latent_loss = F.mse_loss(text_semantic, v_quantized_segment.detach())
        t_loss = (self.commitment_cost * 2.0 * t_e_latent_loss) + (0.5*self.commitment_cost * ta_e_latent_loss) + (0.5*self.commitment_cost * tv_e_latent_loss)
        

        v_quantized_segment = video_semantic + (v_quantized_segment - video_semantic).detach()    #[B,T,D = 256]
        a_quantized_segment = audio_semantic + (a_quantized_segment - audio_semantic).detach()    #[B,T,D = 256]
        t_quantized_segment = text_semantic + (t_quantized_segment - text_semantic).detach()    #[B,T,D = 256]
        
        ################### Use v_full_vectors for cross modal reconstruction gradients #################

        v_full_continuous = torch.cat([video_semantic, audio_semantic, text_semantic], dim=-1)
        a_full_continuous = torch.cat([video_semantic, audio_semantic, text_semantic], dim=-1)
        t_full_continuous = torch.cat([video_semantic, audio_semantic, text_semantic], dim=-1)
        v_full_vectors = v_full_continuous + (v_full_vectors - v_full_continuous).detach()  #[B,T,D = 768]
        a_full_vectors = a_full_continuous + (a_full_vectors - a_full_continuous).detach()  #[B,T,D = 768]
        t_full_vectors = t_full_continuous + (t_full_vectors - t_full_continuous).detach()  #[B,T,D = 768]

        ################## Use v_full_vectors for cross modal reconstruction gradients ###################


        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        t_avg_probs = torch.mean(t_encodings, dim=0)
        t_perplexity = torch.exp(-torch.sum(t_avg_probs * torch.log(t_avg_probs + 1e-10)))

        return  v_full_vectors, v_quantized_segment,\
                a_full_vectors, a_quantized_segment,\
                t_full_vectors, t_quantized_segment,\
                v_loss, a_loss, t_loss, v_perplexity, a_perplexity, t_perplexity,\
                equal_num, cmcm_loss, segment_loss  


