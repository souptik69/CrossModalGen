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
    def __init__(self, audio_dim, video_dim, text_dim,  video_output_dim, n_embeddings, embedding_dim):
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


class AVT_VQVAE_Decoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, video_output_dim):
        super(AVT_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 256 #embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.video_output_dim = video_output_dim
        self.Video_decoder = Video_Decoder_1(video_output_dim, video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder_1(audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_dim, self.hidden_dim)
        self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim * 3, class_num=142)
        self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim * 3, class_num=142)
        self.text_semantic_decoder = Semantic_Decoder(self.hidden_dim * 3, class_num=142)

        ### Equal distribution ###
        # self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim , class_num=142)
        # self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim , class_num=142)
        # self.text_semantic_decoder = Semantic_Decoder(self.hidden_dim , class_num=142)
        ### Equal distribution ###

    def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_spatial, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        video_recon_result = self.Video_decoder(video_spatial, out_vq_video)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio)
        text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)

        video_class = self.video_semantic_decoder(out_vq_video)
        audio_class = self.audio_semantic_decoder(out_vq_audio)
        text_class = self.text_semantic_decoder(out_vq_text)

        ### Equal distribution ###
        # video_class = self.video_semantic_decoder(video_vq)
        # audio_class = self.audio_semantic_decoder(audio_vq)
        # text_class = self.text_semantic_decoder(text_vq)
        ### Equal distribution ###

        return audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class
