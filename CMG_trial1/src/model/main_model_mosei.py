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
from model.models_transformer import TransformerEncoder
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

    def forward(self, feature, attention_mask=None):
        feature = self.affine_matrix(feature)
        if attention_mask is not None:
            mask_transposed = attention_mask.transpose(0, 1).unsqueeze(-1)
            feature = feature * mask_transposed
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask  
        else:
            src_key_padding_mask = None
        feature = self.encoder(feature, src_key_padding_mask) 
        return feature



class InternalTemporalRelationModule_New(nn.Module):
    def __init__(self, input_dim, d_model, num_heads=6, num_layers=6, 
                 dropout=0.1):
        super(InternalTemporalRelationModule_New, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.encoder = TransformerEncoder(
            embed_dim=d_model,
            num_heads=num_heads,
            layers=num_layers,
            attn_dropout=dropout,
            relu_dropout=dropout,
            res_dropout=dropout,
            embed_dropout=0.25,
            attn_mask=False
        )
        self.proj = nn.Conv1d(input_dim, d_model, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature, attention_mask=None):
        batch_size, seq_len, input_dim = feature.size()
        feature = feature.transpose(1, 2).contiguous()  # [batch, input_dim, seq_len]
        feature = self.proj(feature)  # [batch, d_model, seq_len]
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).float()  # [batch, 1, seq_len]
            feature = feature * mask_expanded  # Zero out padding
        feature = feature.permute(2, 0, 1)  # [seq_len, batch, d_model]
        if attention_mask is not None:
            key_padding_mask = ~attention_mask
        else:
            key_padding_mask = None
        feature = self.encoder(feature, key_padding_mask=key_padding_mask)
        feature = feature.transpose(0, 1).contiguous()  # [batch, seq_len, d_model]
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

    def forward(self, video_feat, attention_mask=None):
        encoded = self.relu(self.video_linear(video_feat))
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded * mask_expanded
        return encoded


class Audio_Encoder(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Audio_Encoder, self).__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feat, attention_mask=None):
        encoded = self.relu(self.audio_linear(audio_feat))
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded * mask_expanded
        return encoded


class Text_Encoder(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super(Text_Encoder, self).__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text_feat, attention_mask=None):
        encoded = self.relu(self.text_linear(text_feat))
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            encoded = encoded * mask_expanded
        return encoded


class AVT_VQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, n_embeddings, embedding_dim):
        super(AVT_VQVAE_Encoder, self).__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = embedding_dim
        self.Cross_quantizer = Cross_VQEmbeddingEMA_AVT_hierarchical_pad(n_embeddings, self.hidden_dim)
        self.Video_encoder = Video_Encoder(video_dim, self.hidden_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, self.hidden_dim)
        self.Text_encoder = Text_Encoder(text_dim, self.hidden_dim)
        # self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        # self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)
        # self.text_self_att = InternalTemporalRelationModule(input_dim=text_dim, d_model=self.hidden_dim)
        self.video_self_att = InternalTemporalRelationModule_New(input_dim=video_dim, d_model=self.hidden_dim, num_heads=6, num_layers=6, dropout=0.1)
        self.audio_self_att = InternalTemporalRelationModule_New(input_dim=audio_dim, d_model=self.hidden_dim, num_heads=6, num_layers=6, dropout=0.1)
        self.text_self_att = InternalTemporalRelationModule_New(input_dim=text_dim, d_model=self.hidden_dim, num_heads=6, num_layers=6, dropout=0.1)



    def Audio_VQ_Encoder(self, audio_feat, attention_mask=None):
        audio_feat = audio_feat.cuda()  
        # audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        # audio_semantic_result = self.audio_self_att(audio_semantic_result, attention_mask) 
        # audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous() # [batch, 10, 256]
        audio_semantic_result = self.audio_self_att(audio_feat, attention_mask)
        out_vq, audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result, attention_mask)
        return out_vq, audio_vq  # [batch, 10, 768], [batch, 10, 256]


    def Video_VQ_Encoder(self, video_feat, attention_mask=None):
        video_feat = video_feat.cuda()
        # video_semantic_result = video_feat.transpose(0, 1).contiguous()
        # video_semantic_result = self.video_self_att(video_semantic_result, attention_mask) 
        # video_semantic_result = video_semantic_result.transpose(0, 1).contiguous() # [batch, 10, 256]
        video_semantic_result = self.video_self_att(video_feat, attention_mask) 
        out_vq, video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result, attention_mask)
        return out_vq, video_vq  # [batch,10, 768], [batch, 10, 256]


    def Text_VQ_Encoder(self, text_feat, attention_mask=None):
        text_feat = text_feat.cuda()
        # text_semantic_result = text_feat.transpose(0, 1).contiguous()
        # text_semantic_result = self.text_self_att(text_semantic_result, attention_mask)
        # text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, 10, 256]
        text_semantic_result = self.text_self_att(text_feat, attention_mask)
        out_vq, text_vq = self.Cross_quantizer.Text_vq_embedding(text_semantic_result, attention_mask)
        return out_vq, text_vq  # [batch,10, 768], [batch, 10, 256]



    def forward(self, audio_feat, video_feat, text_feat, epoch, attention_mask=None):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()

        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        
        # video_semantic_result = video_feat.transpose(0, 1).contiguous()
        # video_semantic_result = self.video_self_att(video_semantic_result, attention_mask)
        # video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        # audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        # audio_semantic_result = self.audio_self_att(audio_semantic_result, attention_mask)# [length, batch, hidden_dim]
        # audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        # text_semantic_result = text_feat.transpose(0, 1).contiguous()
        # text_semantic_result = self.text_self_att(text_semantic_result, attention_mask)
        # text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]


        video_semantic_result = self.video_self_att(video_feat, attention_mask)
        audio_semantic_result = self.audio_self_att(audio_feat, attention_mask)
        text_semantic_result = self.text_self_att(text_feat, attention_mask)

        video_encoder_result = self.Video_encoder(video_feat, attention_mask)
        audio_encoder_result = self.Audio_encoder(audio_feat, attention_mask)
        text_encoder_result = self.Text_encoder(text_feat, attention_mask)

        out_vq_video, video_vq, out_vq_audio, audio_vq, \
        out_vq_text, text_vq, \
        video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, \
        equal_num, cmcm_loss, segment_loss = self.Cross_quantizer(audio_semantic_result, video_semantic_result, text_semantic_result, epoch, attention_mask)

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
    
# """ Sentiment Downstream Decoder for Padding Aware Sentiment Label regression """
# class Sentiment_Decoder_Masked(nn.Module):
#     def __init__(self, input_dim):
#         super(Sentiment_Decoder_Masked, self).__init__()
#         self.linear = nn.Linear(input_dim, input_dim)
#         self.sentiment_regressor = nn.Linear(input_dim, 1)
        
#     def forward(self, input_vq, attention_mask=None):
#         if attention_mask is not None:
#             valid_mask_flat = attention_mask.flatten()  # [B*T]
#             num_valid_positions = valid_mask_flat.sum().item()        
#             if num_valid_positions == 0:
#                 batch_size = input_vq.shape[0]
#                 return torch.zeros(batch_size, 1, device=input_vq.device, dtype=input_vq.dtype)    
#             B, T, D = input_vq.shape
#             input_vq_flat = input_vq.view(-1, D)  # [B*T, D]
#             input_vq_valid = input_vq_flat[valid_mask_flat]  # [num_valid, D]   
#             input_feat_valid = self.linear(input_vq_valid)  # [num_valid, D]
#             lengths = attention_mask.sum(dim=1)  # [B] - number of valid positions per sample
#             pooled_features = []
#             start_idx = 0
#             for i in range(B):
#                 sample_length = lengths[i].item()
#                 end_idx = start_idx + sample_length
#                 if sample_length > 0:
#                     sample_features = input_feat_valid[start_idx:end_idx]  # [sample_length, D]
#                     pooled_sample, _ = sample_features.max(0)  # [D] - max over valid timesteps only
#                 else:
#                     # No valid content in this sample
#                     pooled_sample = torch.zeros(D, device=input_vq.device, dtype=input_vq.dtype)
#                 pooled_features.append(pooled_sample)
#                 start_idx = end_idx
#             pooled_tensor = torch.stack(pooled_features)  # [B, D]
#             sentiment_score = self.sentiment_regressor(pooled_tensor)  # [B, 1]
#         else:
#             input_feat = self.linear(input_vq)
#             input_feat, _ = input_feat.max(1)  # Global max pooling
#             sentiment_score = self.sentiment_regressor(input_feat)
        
#         return sentiment_score


""" Sentiment Downstream Decoder with Attention Pooling """
class Sentiment_Decoder_Masked(nn.Module):
    def __init__(self, input_dim):
        super(Sentiment_Decoder_Masked, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.attention_weight = nn.Linear(input_dim, 1)  # Attention scoring
        self.sentiment_regressor = nn.Linear(input_dim, 1)
        
    def forward(self, input_vq, attention_mask=None):
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()  # [B*T]
            num_valid_positions = valid_mask_flat.sum().item()        
            if num_valid_positions == 0:
                batch_size = input_vq.shape[0]
                return torch.zeros(batch_size, 1, device=input_vq.device, dtype=input_vq.dtype)    
            
            B, T, D = input_vq.shape
            input_vq_flat = input_vq.view(-1, D)  # [B*T, D]
            input_vq_valid = input_vq_flat[valid_mask_flat]  # [num_valid, D]   
            input_feat_valid = self.linear(input_vq_valid)  # [num_valid, D]
            
            # Compute attention scores for valid positions
            attention_scores_valid = self.attention_weight(input_feat_valid)  # [num_valid, 1]
            
            lengths = attention_mask.sum(dim=1)  # [B] - number of valid positions per sample
            pooled_features = []
            start_idx = 0
            
            for i in range(B):
                sample_length = lengths[i].item()
                end_idx = start_idx + sample_length
                
                if sample_length > 0:
                    sample_features = input_feat_valid[start_idx:end_idx]  # [sample_length, D]
                    sample_attention_scores = attention_scores_valid[start_idx:end_idx]  # [sample_length, 1]
                    
                    # Compute attention weights (softmax over valid timesteps)
                    sample_attention_weights = F.softmax(sample_attention_scores, dim=0)  # [sample_length, 1]
                    
                    # Weighted sum (attention pooling)
                    pooled_sample = (sample_features * sample_attention_weights).sum(0)  # [D]
                else:
                    # No valid content in this sample
                    pooled_sample = torch.zeros(D, device=input_vq.device, dtype=input_vq.dtype)
                
                pooled_features.append(pooled_sample)
                start_idx = end_idx
            
            pooled_tensor = torch.stack(pooled_features)  # [B, D]
            sentiment_score = self.sentiment_regressor(pooled_tensor)  # [B, 1]
        else:
            input_feat = self.linear(input_vq)  # [B, T, D]
            
            # Compute attention scores
            attention_scores = self.attention_weight(input_feat)  # [B, T, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]
            
            # Attention pooling
            input_feat = (input_feat * attention_weights).sum(1)  # [B, D]
            sentiment_score = self.sentiment_regressor(input_feat)
        
        return sentiment_score



""" Sentiment Downstream Decoder for Sentiment Label classification """
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
    
""" Sentiment Downstream Decoder for Sentiment Label classification combined """
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


# """ Sentiment Downstream Decoder for Padding Aware Sentiment Label regression combined """
# class Sentiment_Decoder_Combined_Masked(nn.Module):
#     def __init__(self, input_dim):
#         super(Sentiment_Decoder_Combined_Masked, self).__init__()
#         self.linear = nn.Linear(input_dim, input_dim)
#         self.sentiment_regressor = nn.Linear(input_dim, 1)

#     def forward(self, video_vq, audio_vq, text_vq, attention_mask=None):
#         input_vq = (video_vq + audio_vq + text_vq) / 3
#         if attention_mask is not None:
#             valid_mask_flat = attention_mask.flatten()  # [B*T]
#             num_valid_positions = valid_mask_flat.sum().item()
#             if num_valid_positions == 0:
#                 batch_size = input_vq.shape[0]
#                 return torch.zeros(batch_size, 1, device=input_vq.device, dtype=input_vq.dtype)            
#             B, T, D = input_vq.shape
#             input_vq_flat = input_vq.view(-1, D)  # [B*T, D]
#             input_vq_valid = input_vq_flat[valid_mask_flat]  # [num_valid, D]                        
#             input_feat_valid = self.linear(input_vq_valid)  # [num_valid, D]            
#             lengths = attention_mask.sum(dim=1)  # [B] - valid positions per sample
#             pooled_features = []            
#             start_idx = 0
#             for i in range(B):
#                 sample_length = lengths[i].item()
#                 end_idx = start_idx + sample_length                
#                 if sample_length > 0:
#                     sample_features = input_feat_valid[start_idx:end_idx]  # [sample_length, D]
#                     pooled_sample, _ = sample_features.max(0)  # [D] - max over valid timesteps
#                 else:
#                     pooled_sample = torch.zeros(D, device=input_vq.device, dtype=input_vq.dtype)                
#                 pooled_features.append(pooled_sample)
#                 start_idx = end_idx       
#             pooled_tensor = torch.stack(pooled_features)  # [B, D]
#             sentiment_score = self.sentiment_regressor(pooled_tensor)  # [B, 1]            
#         else:
#             input_feat = self.linear(input_vq)
#             input_feat, _ = input_feat.max(1)  # Global max pooling
#             sentiment_score = self.sentiment_regressor(input_feat)
        
#         return sentiment_score


""" Sentiment Downstream Decoder Combined with Attention Pooling """
class Sentiment_Decoder_Combined_Masked(nn.Module):
    def __init__(self, input_dim):
        super(Sentiment_Decoder_Combined_Masked, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.attention_weight = nn.Linear(input_dim, 1)  # Attention scoring
        self.sentiment_regressor = nn.Linear(input_dim, 1)

    def forward(self, video_vq, audio_vq, text_vq, attention_mask=None):
        input_vq = (video_vq + audio_vq + text_vq) / 3
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()  # [B*T]
            num_valid_positions = valid_mask_flat.sum().item()
            if num_valid_positions == 0:
                batch_size = input_vq.shape[0]
                return torch.zeros(batch_size, 1, device=input_vq.device, dtype=input_vq.dtype)            
            
            B, T, D = input_vq.shape
            input_vq_flat = input_vq.view(-1, D)  # [B*T, D]
            input_vq_valid = input_vq_flat[valid_mask_flat]  # [num_valid, D]                        
            input_feat_valid = self.linear(input_vq_valid)  # [num_valid, D]
            
            # Compute attention scores for valid positions
            attention_scores_valid = self.attention_weight(input_feat_valid)  # [num_valid, 1]
            
            lengths = attention_mask.sum(dim=1)  # [B] - valid positions per sample
            pooled_features = []            
            start_idx = 0
            
            for i in range(B):
                sample_length = lengths[i].item()
                end_idx = start_idx + sample_length                
                
                if sample_length > 0:
                    sample_features = input_feat_valid[start_idx:end_idx]  # [sample_length, D]
                    sample_attention_scores = attention_scores_valid[start_idx:end_idx]  # [sample_length, 1]
                    
                    # Compute attention weights (softmax over valid timesteps)
                    sample_attention_weights = F.softmax(sample_attention_scores, dim=0)  # [sample_length, 1]
                    
                    # Weighted sum (attention pooling)
                    pooled_sample = (sample_features * sample_attention_weights).sum(0)  # [D]
                else:
                    pooled_sample = torch.zeros(D, device=input_vq.device, dtype=input_vq.dtype)                
                
                pooled_features.append(pooled_sample)
                start_idx = end_idx       
            
            pooled_tensor = torch.stack(pooled_features)  # [B, D]
            sentiment_score = self.sentiment_regressor(pooled_tensor)  # [B, 1]            
        else:
            input_feat = self.linear(input_vq)  # [B, T, D]
            
            # Compute attention scores
            attention_scores = self.attention_weight(input_feat)  # [B, T, 1]
            attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]
            
            # Attention pooling
            input_feat = (input_feat * attention_weights).sum(1)  # [B, D]
            sentiment_score = self.sentiment_regressor(input_feat)
        
        return sentiment_score



# class Audio_Decoder(nn.Module):
#     def __init__(self, output_dim , vq_dim):
#         super(Audio_Decoder, self).__init__()
#         self.output_dim = output_dim
#         self.vq_dim = vq_dim
#         self.relu = nn.ReLU()
#         self.audio_rec = nn.Linear(vq_dim * 2, output_dim)
#         self.audio_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
#         self.audio_linear = nn.Linear(vq_dim, vq_dim)

#     def forward(self, audio_encoder_result, audio_vq):
#         audio_vq_1 = self.audio_linear_1(audio_vq)
#         audio_vq_result = self.audio_linear(audio_vq_1)
#         audio_encoder_result = torch.cat([audio_vq_result, audio_encoder_result], dim=2)
#         audio_decoder_result = self.audio_rec(audio_encoder_result)
#         return audio_decoder_result


class Audio_Decoder(nn.Module):
    def __init__(self, output_dim, vq_dim):
        super(Audio_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.audio_rec = nn.Linear(vq_dim * 2, output_dim)
        self.audio_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.audio_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, audio_encoder_result, audio_vq, attention_mask=None):
        B, T, encoder_dim = audio_encoder_result.shape  # [batch, seq_len, vq_dim]
        
        audio_decoder_result = torch.zeros(B, T, self.output_dim, 
                                         device=audio_encoder_result.device, 
                                         dtype=audio_encoder_result.dtype)
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()  # [B*T]
            num_valid_positions = valid_mask_flat.sum().item()
            
            if num_valid_positions > 0:
                audio_encoder_flat = audio_encoder_result.view(-1, encoder_dim)  # [B*T, encoder_dim]
                audio_vq_flat = audio_vq.view(-1, audio_vq.shape[-1])  # [B*T, vq_dim*3]
                
                audio_encoder_valid = audio_encoder_flat[valid_mask_flat]  # [num_valid, encoder_dim]
                audio_vq_valid = audio_vq_flat[valid_mask_flat]  # [num_valid, vq_dim*3]
                
                audio_vq_1_valid = self.audio_linear_1(audio_vq_valid)  # [num_valid, vq_dim]
                audio_vq_result_valid = self.audio_linear(audio_vq_1_valid)  # [num_valid, vq_dim]
                
                combined_valid = torch.cat([audio_vq_result_valid, audio_encoder_valid], dim=1)  # [num_valid, vq_dim*2]
                
                reconstructed_valid = self.audio_rec(combined_valid)  # [num_valid, output_dim]
                
                valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                audio_decoder_flat = audio_decoder_result.view(-1, self.output_dim)
                audio_decoder_flat[valid_positions] = reconstructed_valid
                audio_decoder_result = audio_decoder_flat.view(B, T, self.output_dim)
                
        else:
            audio_vq_1 = self.audio_linear_1(audio_vq)
            audio_vq_result = self.audio_linear(audio_vq_1)
            audio_encoder_combined = torch.cat([audio_vq_result, audio_encoder_result], dim=2)
            audio_decoder_result = self.audio_rec(audio_encoder_combined)
        
        return audio_decoder_result



# class Text_Decoder(nn.Module):
#     def __init__(self, output_dim, vq_dim):
#         super(Text_Decoder, self).__init__()
#         self.output_dim = output_dim
#         self.vq_dim = vq_dim
#         self.relu = nn.ReLU()
#         self.text_rec = nn.Linear(vq_dim * 2, output_dim)
#         self.text_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
#         self.text_linear = nn.Linear(vq_dim, vq_dim)

#     def forward(self, text_encoder_result, text_vq):
#         text_vq_1 = self.text_linear_1(text_vq)
#         text_vq_result = self.text_linear(text_vq_1)
#         text_encoder_result = torch.cat([text_vq_result, text_encoder_result], dim=2)
#         text_decoder_result = self.text_rec(text_encoder_result)
#         return text_decoder_result
    

class Text_Decoder(nn.Module):
    def __init__(self, output_dim, vq_dim):
        super(Text_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.text_rec = nn.Linear(vq_dim * 2, output_dim)
        self.text_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.text_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, text_encoder_result, text_vq, attention_mask=None):
        B, T, encoder_dim = text_encoder_result.shape

        text_decoder_result = torch.zeros(B, T, self.output_dim,
                                        device=text_encoder_result.device,
                                        dtype=text_encoder_result.dtype)
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()
            num_valid_positions = valid_mask_flat.sum().item()
            
            if num_valid_positions > 0:
                text_encoder_flat = text_encoder_result.view(-1, encoder_dim)
                text_vq_flat = text_vq.view(-1, text_vq.shape[-1])
                
                text_encoder_valid = text_encoder_flat[valid_mask_flat]
                text_vq_valid = text_vq_flat[valid_mask_flat]
                
                text_vq_1_valid = self.text_linear_1(text_vq_valid)
                text_vq_result_valid = self.text_linear(text_vq_1_valid)
                
                combined_valid = torch.cat([text_vq_result_valid, text_encoder_valid], dim=1)
                reconstructed_valid = self.text_rec(combined_valid)
                
                valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                text_decoder_flat = text_decoder_result.view(-1, self.output_dim)
                text_decoder_flat[valid_positions] = reconstructed_valid
                text_decoder_result = text_decoder_flat.view(B, T, self.output_dim)
                
        else:
            text_vq_1 = self.text_linear_1(text_vq)
            text_vq_result = self.text_linear(text_vq_1)
            text_encoder_combined = torch.cat([text_vq_result, text_encoder_result], dim=2)
            text_decoder_result = self.text_rec(text_encoder_combined)

        return text_decoder_result


# class Video_Decoder(nn.Module):
#     def __init__(self, output_dim, vq_dim):
#         super(Video_Decoder, self).__init__()
#         self.output_dim = output_dim
#         self.vq_dim = vq_dim
#         self.relu = nn.ReLU()
#         self.video_rec = nn.Linear(vq_dim * 2, output_dim)
#         self.video_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
#         self.video_linear = nn.Linear(vq_dim, vq_dim)

#     def forward(self, video_encoder_result, video_vq):
#         video_vq_1 = self.video_linear_1(video_vq)
#         video_vq_result = self.video_linear(video_vq_1)
#         video_encoder_result = torch.cat([video_vq_result, video_encoder_result], dim=2)
#         video_decoder_result = self.video_rec(video_encoder_result)
#         return video_decoder_result
    


class Video_Decoder(nn.Module):
    def __init__(self, output_dim, vq_dim):
        super(Video_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.video_rec = nn.Linear(vq_dim * 2, output_dim)
        self.video_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.video_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, video_encoder_result, video_vq, attention_mask=None):
        B, T, encoder_dim = video_encoder_result.shape
        
        video_decoder_result = torch.zeros(B, T, self.output_dim,
                                         device=video_encoder_result.device,
                                         dtype=video_encoder_result.dtype)
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()
            num_valid_positions = valid_mask_flat.sum().item()
            
            if num_valid_positions > 0:
                video_encoder_flat = video_encoder_result.view(-1, encoder_dim)
                video_vq_flat = video_vq.view(-1, video_vq.shape[-1])
                
                video_encoder_valid = video_encoder_flat[valid_mask_flat]
                video_vq_valid = video_vq_flat[valid_mask_flat]
                
                video_vq_1_valid = self.video_linear_1(video_vq_valid)
                video_vq_result_valid = self.video_linear(video_vq_1_valid)
                
                combined_valid = torch.cat([video_vq_result_valid, video_encoder_valid], dim=1)
                reconstructed_valid = self.video_rec(combined_valid)
                
                valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                video_decoder_flat = video_decoder_result.view(-1, self.output_dim)
                video_decoder_flat[valid_positions] = reconstructed_valid
                video_decoder_result = video_decoder_flat.view(B, T, self.output_dim)
                
        else:
            video_vq_1 = self.video_linear_1(video_vq)
            video_vq_result = self.video_linear(video_vq_1)
            video_encoder_combined = torch.cat([video_vq_result, video_encoder_result], dim=2)
            video_decoder_result = self.video_rec(video_encoder_combined)
        
        return video_decoder_result

# '''Decoder for both uni-modal feature reconstruction. To be used for unsupervised pre-training'''
# class AVT_VQVAE_Decoder(nn.Module):
#     def __init__(self, audio_dim, video_dim, text_dim, embedding_dim):
#         super(AVT_VQVAE_Decoder, self).__init__()
#         self.hidden_dim = embedding_dim #embedding_dim
#         self.video_dim = video_dim
#         self.audio_dim = audio_dim
#         self.text_dim = text_dim
#         self.Video_decoder = Video_Decoder(video_dim, self.hidden_dim)
#         self.Audio_decoder = Audio_Decoder(audio_dim, self.hidden_dim)
#         self.Text_decoder = Text_Decoder(text_dim, self.hidden_dim)

#         self.video_sentiment_decoder = Sentiment_Decoder_Masked(self.hidden_dim * 3)
#         self.audio_sentiment_decoder = Sentiment_Decoder_Masked(self.hidden_dim * 3)
#         self.text_sentiment_decoder = Sentiment_Decoder_Masked(self.hidden_dim * 3)
#         self.combined_sentiment_decoder = Sentiment_Decoder_Combined_Masked(self.hidden_dim * 3)


#     def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_encoder_result, text_encoder_result, out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq, attention_mask=None):
#         video_feat = video_feat.cuda()
#         audio_feat = audio_feat.cuda()
#         text_feat = text_feat.cuda()
#         video_recon_result = self.Video_decoder(video_encoder_result, out_vq_video)
#         audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio)
#         text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text)

#         if attention_mask is not None:
#             attention_mask = attention_mask.cuda()
            
#             audio_mask = attention_mask.unsqueeze(-1).expand_as(audio_feat)
#             video_mask = attention_mask.unsqueeze(-1).expand_as(video_feat)
#             text_mask = attention_mask.unsqueeze(-1).expand_as(text_feat)
            
#             audio_recon_masked = audio_recon_result * audio_mask
#             audio_target_masked = audio_feat * audio_mask
#             video_recon_masked = video_recon_result * video_mask
#             video_target_masked = video_feat * video_mask
#             text_recon_masked = text_recon_result * text_mask
#             text_target_masked = text_feat * text_mask

#             audio_sq_error = F.mse_loss(audio_recon_masked, audio_target_masked, reduction='sum')
#             video_sq_error = F.mse_loss(video_recon_masked, video_target_masked, reduction='sum')
#             text_sq_error = F.mse_loss(text_recon_masked, text_target_masked, reduction='sum')
            
#             num_valid_audio = audio_mask.sum()
#             num_valid_video = video_mask.sum()
#             num_valid_text = text_mask.sum()
            

#             audio_recon_loss = audio_sq_error / (num_valid_audio + 1e-6)
#             video_recon_loss = video_sq_error / (num_valid_video + 1e-6)
#             text_recon_loss = text_sq_error / (num_valid_text + 1e-6)
               
#         else:
#             video_recon_loss = F.mse_loss(video_recon_result, video_feat)
#             audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
#             text_recon_loss = F.mse_loss(text_recon_result, text_feat)
        
#         video_score = self.video_sentiment_decoder(out_vq_video, attention_mask=attention_mask)
#         audio_score = self.audio_sentiment_decoder(out_vq_audio, attention_mask=attention_mask)
#         text_score = self.text_sentiment_decoder(out_vq_text, attention_mask=attention_mask)
#         combined_score = self.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text, attention_mask=attention_mask)

#         return audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score, combined_score



class AVT_VQVAE_Decoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, embedding_dim):
        super(AVT_VQVAE_Decoder, self).__init__()
        self.hidden_dim = embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        
        self.Video_decoder = Video_Decoder(video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder(audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_dim, self.hidden_dim)

        self.video_sentiment_decoder = Sentiment_Decoder_Masked(self.hidden_dim * 3)
        self.audio_sentiment_decoder = Sentiment_Decoder_Masked(self.hidden_dim * 3)
        self.text_sentiment_decoder = Sentiment_Decoder_Masked(self.hidden_dim * 3)
        self.combined_sentiment_decoder = Sentiment_Decoder_Combined_Masked(self.hidden_dim * 3)

    def forward(self, audio_feat, video_feat, text_feat, 
                audio_encoder_result, video_encoder_result, text_encoder_result, 
                out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq, 
                attention_mask=None):
    
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        
        video_recon_result = self.Video_decoder(video_encoder_result, out_vq_video, attention_mask)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio, attention_mask)
        text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text, attention_mask)

        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
            
            audio_mask = attention_mask.unsqueeze(-1).expand_as(audio_feat)
            video_mask = attention_mask.unsqueeze(-1).expand_as(video_feat)
            text_mask = attention_mask.unsqueeze(-1).expand_as(text_feat)
            
            audio_recon_masked = audio_recon_result * audio_mask
            audio_target_masked = audio_feat * audio_mask
            video_recon_masked = video_recon_result * video_mask
            video_target_masked = video_feat * video_mask
            text_recon_masked = text_recon_result * text_mask
            text_target_masked = text_feat * text_mask

            audio_sq_error = F.mse_loss(audio_recon_masked, audio_target_masked, reduction='sum')
            video_sq_error = F.mse_loss(video_recon_masked, video_target_masked, reduction='sum')
            text_sq_error = F.mse_loss(text_recon_masked, text_target_masked, reduction='sum')
            
            num_valid_audio = audio_mask.sum()
            num_valid_video = video_mask.sum()
            num_valid_text = text_mask.sum()
            
            audio_recon_loss = audio_sq_error / (num_valid_audio + 1e-6)
            video_recon_loss = video_sq_error / (num_valid_video + 1e-6)
            text_recon_loss = text_sq_error / (num_valid_text + 1e-6)
               
        else:
            video_recon_loss = F.mse_loss(video_recon_result, video_feat)
            audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
            text_recon_loss = F.mse_loss(text_recon_result, text_feat)
        
        video_score = self.video_sentiment_decoder(out_vq_video, attention_mask=attention_mask)
        audio_score = self.audio_sentiment_decoder(out_vq_audio, attention_mask=attention_mask)
        text_score = self.text_sentiment_decoder(out_vq_text, attention_mask=attention_mask)
        combined_score = self.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text, 
                                                        attention_mask=attention_mask)

        return audio_recon_loss, video_recon_loss, text_recon_loss, audio_score, video_score, text_score, combined_score



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
                new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, 2*D:])  #Audio = (4/9)Video + (4/9)Audio + (1/9)Text
                new_embedding[:, :D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])     #Video = (16/27)Video + (7/27)Audio + (4/27)Text

                '''T -> V -> A''' 
                # new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])  #Audio = (4/9)Video + (4/9)Audio + (1/9)Text
                # new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])     #Video = (16/27)Video + (7/27)Audio + (4/27)Text

                '''V -> A -> T'''
                # new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                # new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                # new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])

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
                if x > 300:  # Dead for too long
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


class Cross_VQEmbeddingEMA_AVT_hierarchical_pad(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AVT_hierarchical_pad, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding_dim = embedding_dim

        init_bound = 1 / n_embeddings
        # embedding = torch.Tensor(n_embeddings, embedding_dim * 3)
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        embedding = torch.cat([embedding, embedding, embedding], dim=1)
        
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_v", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_a", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_t", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))

    def Video_vq_embedding(self, video_semantic, attention_mask=None):
        B, T, D = video_semantic.size()
        
        # Initialize outputs with zeros
        v_quantized = torch.zeros_like(video_semantic)
        out_vq = torch.zeros(B, T, self.embedding_dim * 3, device=video_semantic.device, dtype=video_semantic.dtype)
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()
            num_valid = valid_mask_flat.sum().item()
            
            if num_valid > 0:
                v_flat_all = video_semantic.detach().reshape(-1, D)
                v_flat = v_flat_all[valid_mask_flat]

                video_embedding = self.embedding[:, :D]

                v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                        torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                        v_flat, video_embedding.t(),
                                        alpha=-2.0, beta=1.0)

                v_indices_valid = torch.argmin(v_distances.double(), dim=-1)

                v_quantized_valid = F.embedding(v_indices_valid, video_embedding)
                out_vq_valid = F.embedding(v_indices_valid, self.embedding)

                valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                
                v_quantized_flat = v_quantized.view(-1, D)
                out_vq_flat = out_vq.view(-1, self.embedding_dim * 3)
                
                v_quantized_flat[valid_positions] = v_quantized_valid
                out_vq_flat[valid_positions] = out_vq_valid
                
                v_quantized = v_quantized_flat.view(B, T, D)
                out_vq = out_vq_flat.view(B, T, self.embedding_dim * 3)
        else:
            v_flat = video_semantic.detach().reshape(-1, D)
            video_embedding = self.embedding[:, :D]

            v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                    torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                    v_flat, video_embedding.t(),
                                    alpha=-2.0, beta=1.0)

            v_indices = torch.argmin(v_distances.double(), dim=-1)
            v_quantized = F.embedding(v_indices, video_embedding).view(B, T, D)
            out_vq = F.embedding(v_indices, self.embedding).view(B, T, -1)

        v_quantized = video_semantic + (v_quantized - video_semantic).detach()
        
        return out_vq, v_quantized

    def Audio_vq_embedding(self, audio_semantic, attention_mask=None):
        B, T, D = audio_semantic.size()
        
        a_quantized = torch.zeros_like(audio_semantic)
        out_vq = torch.zeros(B, T, self.embedding_dim * 3, device=audio_semantic.device, dtype=audio_semantic.dtype)
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()
            num_valid = valid_mask_flat.sum().item()
            
            if num_valid > 0:
                a_flat_all = audio_semantic.detach().reshape(-1, D)
                a_flat = a_flat_all[valid_mask_flat]
         
                audio_embedding = self.embedding[:, D:2*D]

                a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                        torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                        a_flat, audio_embedding.t(),
                                        alpha=-2.0, beta=1.0)

                a_indices_valid = torch.argmin(a_distances.double(), dim=-1)

                a_quantized_valid = F.embedding(a_indices_valid, audio_embedding)
                out_vq_valid = F.embedding(a_indices_valid, self.embedding)

                valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                
                a_quantized_flat = a_quantized.view(-1, D)
                out_vq_flat = out_vq.view(-1, self.embedding_dim * 3)
                
                a_quantized_flat[valid_positions] = a_quantized_valid
                out_vq_flat[valid_positions] = out_vq_valid
                
                a_quantized = a_quantized_flat.view(B, T, D)
                out_vq = out_vq_flat.view(B, T, self.embedding_dim * 3)
        else:
            a_flat = audio_semantic.detach().reshape(-1, D)
            audio_embedding = self.embedding[:, D:2*D]

            a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                    torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                    a_flat, audio_embedding.t(),
                                    alpha=-2.0, beta=1.0)

            a_indices = torch.argmin(a_distances.double(), dim=-1)
            a_quantized = F.embedding(a_indices, audio_embedding).view(B, T, D)
            out_vq = F.embedding(a_indices, self.embedding).view(B, T, -1)

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        
        return out_vq, a_quantized

    def Text_vq_embedding(self, text_semantic, attention_mask=None):
        B, T, D = text_semantic.size()
        
        t_quantized = torch.zeros_like(text_semantic)
        out_vq = torch.zeros(B, T, self.embedding_dim * 3, device=text_semantic.device, dtype=text_semantic.dtype)
        
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()
            num_valid = valid_mask_flat.sum().item()
            
            if num_valid > 0:
                t_flat_all = text_semantic.detach().reshape(-1, D)
                t_flat = t_flat_all[valid_mask_flat]

                text_embedding = self.embedding[:, 2*D:]

                t_distances = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                        torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                        t_flat, text_embedding.t(),
                                        alpha=-2.0, beta=1.0)

                t_indices_valid = torch.argmin(t_distances.double(), dim=-1)

                t_quantized_valid = F.embedding(t_indices_valid, text_embedding)
                out_vq_valid = F.embedding(t_indices_valid, self.embedding)

                valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
                
                t_quantized_flat = t_quantized.view(-1, D)
                out_vq_flat = out_vq.view(-1, self.embedding_dim * 3)
                
                t_quantized_flat[valid_positions] = t_quantized_valid
                out_vq_flat[valid_positions] = out_vq_valid
                
                t_quantized = t_quantized_flat.view(B, T, D)
                out_vq = out_vq_flat.view(B, T, self.embedding_dim * 3)
        else:
            t_flat = text_semantic.detach().reshape(-1, D)
            text_embedding = self.embedding[:, 2*D:]

            t_distances = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                    torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                    t_flat, text_embedding.t(),
                                    alpha=-2.0, beta=1.0)

            t_indices = torch.argmin(t_distances.double(), dim=-1)
            t_quantized = F.embedding(t_indices, text_embedding).view(B, T, D)
            out_vq = F.embedding(t_indices, self.embedding).view(B, T, -1)

        t_quantized = text_semantic + (t_quantized - text_semantic).detach()
        
        return out_vq, t_quantized

    def forward(self, audio_semantic, video_semantic, text_semantic, epoch, attention_mask=None):
        M, D_total = self.embedding.size()  # M = 256, D_total = 768
        B, T, D = audio_semantic.size()     # D = 256

        # Extract valid content positions only
        if attention_mask is not None:
            valid_mask_flat = attention_mask.flatten()
            num_valid_positions = valid_mask_flat.sum().item()
            
            if num_valid_positions == 0:
                # Return zero tensors if no valid content
                return (torch.zeros(B, T, D_total, device=audio_semantic.device, dtype=audio_semantic.dtype),
                       torch.zeros_like(video_semantic),
                       torch.zeros(B, T, D_total, device=audio_semantic.device, dtype=audio_semantic.dtype),
                       torch.zeros_like(audio_semantic),
                       torch.zeros(B, T, D_total, device=audio_semantic.device, dtype=audio_semantic.dtype),
                       torch.zeros_like(text_semantic),
                       torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),
                       torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0),
                       0, torch.tensor(0.0), torch.tensor(0.0))
            
            # Extract content only
            a_flat_all = audio_semantic.detach().reshape(-1, D)
            v_flat_all = video_semantic.detach().reshape(-1, D)
            t_flat_all = text_semantic.detach().reshape(-1, D)
            
            a_flat = a_flat_all[valid_mask_flat]
            v_flat = v_flat_all[valid_mask_flat]
            t_flat = t_flat_all[valid_mask_flat]
            
            # Gradient-enabled versions for CMCM
            audio_semantic_flat_all = audio_semantic.reshape(-1, D)
            video_semantic_flat_all = video_semantic.reshape(-1, D)
            text_semantic_flat_all = text_semantic.reshape(-1, D)
            
            audio_semantic_flat = audio_semantic_flat_all[valid_mask_flat]
            video_semantic_flat = video_semantic_flat_all[valid_mask_flat]
            text_semantic_flat = text_semantic_flat_all[valid_mask_flat]
        else:
            valid_mask_flat = None
            num_valid_positions = B * T
            a_flat = audio_semantic.detach().reshape(-1, D)
            v_flat = video_semantic.detach().reshape(-1, D)
            t_flat = text_semantic.detach().reshape(-1, D)
            audio_semantic_flat = audio_semantic.reshape(-1, D)
            video_semantic_flat = video_semantic.reshape(-1, D)
            text_semantic_flat = text_semantic.reshape(-1, D)
        
        # Codebook segments
        video_embedding = self.embedding[:, :D]
        audio_embedding = self.embedding[:, D:2*D]
        text_embedding = self.embedding[:, 2*D:]

        # Distance computation - content only
        v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                v_flat, video_embedding.t(),
                                alpha=-2.0, beta=1.0)
        
        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)

        t_distances = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                t_flat, text_embedding.t(),
                                alpha=-2.0, beta=1.0)

        # Gradient-enabled distances for CMCM
        v_distances_gradient = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                        torch.sum(video_semantic_flat ** 2, dim=1, keepdim=True),
                                        video_semantic_flat, video_embedding.t(),
                                        alpha=-2.0, beta=1.0)
        
        a_distances_gradient = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                        torch.sum(audio_semantic_flat ** 2, dim=1, keepdim=True),
                                        audio_semantic_flat, audio_embedding.t(),
                                        alpha=-2.0, beta=1.0)

        t_distances_gradient = torch.addmm(torch.sum(text_embedding ** 2, dim=1) +
                                        torch.sum(text_semantic_flat ** 2, dim=1, keepdim=True),
                                        text_semantic_flat, text_embedding.t(),
                                        alpha=-2.0, beta=1.0)

        # Soft assignments for CMCM
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)
        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)
        t_ph = F.softmax(-torch.sqrt(t_distances_gradient), dim=1)

        # Batch-level aggregation for CMCM
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            v_pH_list = []
            a_pH_list = []
            t_pH_list = []
            
            start_idx = 0
            for i in range(B):
                sample_length = lengths[i].item()
                end_idx = start_idx + sample_length
                
                if sample_length > 0:
                    v_pH_sample = torch.mean(v_ph[start_idx:end_idx], dim=0)
                    a_pH_sample = torch.mean(a_ph[start_idx:end_idx], dim=0)
                    t_pH_sample = torch.mean(t_ph[start_idx:end_idx], dim=0)
                else:
                    v_pH_sample = torch.ones(M, device=video_semantic.device) / M
                    a_pH_sample = torch.ones(M, device=audio_semantic.device) / M
                    t_pH_sample = torch.ones(M, device=text_semantic.device) / M
                
                v_pH_list.append(v_pH_sample)
                a_pH_list.append(a_pH_sample)
                t_pH_list.append(t_pH_sample)
                
                start_idx = end_idx
            
            v_pH = torch.stack(v_pH_list)
            a_pH = torch.stack(a_pH_list)
            t_pH = torch.stack(t_pH_list)
        else:
            v_ph = torch.reshape(v_ph, (B, T, M))
            a_ph = torch.reshape(a_ph, (B, T, M))
            t_ph = torch.reshape(t_ph, (B, T, M))
            
            v_pH = torch.mean(v_ph, dim=1)
            a_pH = torch.mean(a_ph, dim=1)
            t_pH = torch.mean(t_ph, dim=1)

        # CMCM loss computation
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
        EScode_sumdim1_at = torch.sum(EScode_at, dim=1)
        EScode_sumdim1_tv = torch.sum(EScode_tv, dim=1)
        
        Lcmcm_av = Lcmcm_at = Lcmcm_tv = 0

        for i in range(B):
            Lcmcm_av -= torch.log(EScode_av[i, i] / (EScode_sumdim1_av[i] + self.epsilon))
            Lcmcm_at -= torch.log(EScode_at[i, i] / (EScode_sumdim1_at[i] + self.epsilon))
            Lcmcm_tv -= torch.log(EScode_tv[i, i] / (EScode_sumdim1_tv[i] + self.epsilon))

        Lcmcm_av /= B
        Lcmcm_at /= B
        Lcmcm_tv /= B

        # Content-only codebook assignment
        v_indices_valid = torch.argmin(v_distances.double(), dim=-1)
        a_indices_valid = torch.argmin(a_distances.double(), dim=-1)
        t_indices_valid = torch.argmin(t_distances.double(), dim=-1)

        # Content-only encodings for EMA
        v_encodings_valid = F.one_hot(v_indices_valid, M).double()
        a_encodings_valid = F.one_hot(a_indices_valid, M).double()
        t_encodings_valid = F.one_hot(t_indices_valid, M).double()

        # Pure quantized vector reconstruction - never quantize padding
        v_quantized_segment = torch.zeros(B, T, D, device=video_semantic.device, dtype=video_semantic.dtype)
        a_quantized_segment = torch.zeros(B, T, D, device=audio_semantic.device, dtype=audio_semantic.dtype)
        t_quantized_segment = torch.zeros(B, T, D, device=text_semantic.device, dtype=text_semantic.dtype)
        
        v_full_vectors = torch.zeros(B, T, D_total, device=video_semantic.device, dtype=video_semantic.dtype)
        a_full_vectors = torch.zeros(B, T, D_total, device=audio_semantic.device, dtype=audio_semantic.dtype)
        t_full_vectors = torch.zeros(B, T, D_total, device=text_semantic.device, dtype=text_semantic.dtype)
        
        if attention_mask is not None and num_valid_positions > 0:
            # Generate quantized vectors only for content positions
            v_quantized_valid = F.embedding(v_indices_valid, video_embedding)
            a_quantized_valid = F.embedding(a_indices_valid, audio_embedding)
            t_quantized_valid = F.embedding(t_indices_valid, text_embedding)
            
            v_full_valid = F.embedding(v_indices_valid, self.embedding)
            a_full_valid = F.embedding(a_indices_valid, self.embedding)
            t_full_valid = F.embedding(t_indices_valid, self.embedding)
            
            # Place quantized vectors at valid positions only
            valid_positions = torch.nonzero(valid_mask_flat, as_tuple=True)[0]
            
            v_quantized_flat = v_quantized_segment.view(-1, D)
            a_quantized_flat = a_quantized_segment.view(-1, D)
            t_quantized_flat = t_quantized_segment.view(-1, D)
            
            v_full_flat = v_full_vectors.view(-1, D_total)
            a_full_flat = a_full_vectors.view(-1, D_total)
            t_full_flat = t_full_vectors.view(-1, D_total)
            
            v_quantized_flat[valid_positions] = v_quantized_valid
            a_quantized_flat[valid_positions] = a_quantized_valid
            t_quantized_flat[valid_positions] = t_quantized_valid
            
            v_full_flat[valid_positions] = v_full_valid
            a_full_flat[valid_positions] = a_full_valid
            t_full_flat[valid_positions] = t_full_valid
            
            v_quantized_segment = v_quantized_flat.view(B, T, D)
            a_quantized_segment = a_quantized_flat.view(B, T, D)
            t_quantized_segment = t_quantized_flat.view(B, T, D)
            
            v_full_vectors = v_full_flat.view(B, T, D_total)
            a_full_vectors = a_full_flat.view(B, T, D_total)
            t_full_vectors = t_full_flat.view(B, T, D_total)
            
        elif attention_mask is None:
            v_quantized_segment = F.embedding(v_indices_valid, video_embedding).view(B, T, D)
            a_quantized_segment = F.embedding(a_indices_valid, audio_embedding).view(B, T, D)
            t_quantized_segment = F.embedding(t_indices_valid, text_embedding).view(B, T, D)
            
            v_full_vectors = F.embedding(v_indices_valid, self.embedding).view(B, T, D_total)
            a_full_vectors = F.embedding(a_indices_valid, self.embedding).view(B, T, D_total)
            t_full_vectors = F.embedding(t_indices_valid, self.embedding).view(B, T, D_total)

        # Cross-modal mode consistency check
        if attention_mask is not None and num_valid_positions > 0:
            equal_num = 0
            for i in range(B):
                sample_length = attention_mask[i].sum().item()
                if sample_length > 0:
                    sample_start = i * T
                    sample_end = sample_start + T
                    sample_valid_mask = valid_mask_flat[sample_start:sample_end]
                    sample_valid_positions = torch.nonzero(sample_valid_mask, as_tuple=True)[0]
                    
                    if len(sample_valid_positions) > 0:
                        global_valid_start = valid_mask_flat[:sample_start].sum().item()
                        global_valid_end = global_valid_start + len(sample_valid_positions)
                        
                        v_sample_indices = v_indices_valid[global_valid_start:global_valid_end]
                        a_sample_indices = a_indices_valid[global_valid_start:global_valid_end]
                        t_sample_indices = t_indices_valid[global_valid_start:global_valid_end]
                        
                        if len(v_sample_indices) > 0:
                            v_mode = torch.mode(v_sample_indices).values
                            a_mode = torch.mode(a_sample_indices).values
                            t_mode = torch.mode(t_sample_indices).values
                            
                            if v_mode == a_mode and a_mode == t_mode:
                                equal_num += 1
        elif attention_mask is None:
            v_indices_reshape = v_indices_valid.reshape(B, T)
            a_indices_reshape = a_indices_valid.reshape(B, T)
            t_indices_reshape = t_indices_valid.reshape(B, T)
            
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            t_indices_mode = torch.mode(t_indices_reshape, dim=-1, keepdim=False)
            
            equal_item = (a_indices_mode.values == v_indices_mode.values) & (a_indices_mode.values == t_indices_mode.values)
            equal_num = equal_item.sum()
        
        # Content-aware EMA updates during training
        if self.training and num_valid_positions > 0:
            # Video segment update
            self.ema_count_v = self.decay * self.ema_count_v + (1 - self.decay) * (torch.sum(v_encodings_valid, dim=0))
            n_v = torch.sum(self.ema_count_v)
            self.ema_count_v = (self.ema_count_v + self.epsilon) / (n_v + M * self.epsilon) * n_v

            v_dw = torch.matmul(v_encodings_valid.t(), v_flat)
            a_dw_v = torch.matmul(v_encodings_valid.t(), a_flat)
            t_dw_v = torch.matmul(v_encodings_valid.t(), t_flat)

            v_segment_update = (0.6 * v_dw) + (0.2 * a_dw_v) + (0.2 * t_dw_v)
            with torch.no_grad():
                new_embedding_v = self.embedding.clone()
                self.ema_weight[:, :D] = self.decay * self.ema_weight[:, :D] + (1 - self.decay) * v_segment_update
                new_embedding_v[:, :D] = self.ema_weight[:, :D] / (self.ema_count_v.unsqueeze(-1))
                self.embedding = new_embedding_v

            # Audio segment update
            self.ema_count_a = self.decay * self.ema_count_a + (1 - self.decay) * (torch.sum(a_encodings_valid, dim=0))
            n_a = torch.sum(self.ema_count_a)
            self.ema_count_a = (self.ema_count_a + self.epsilon) / (n_a + M * self.epsilon) * n_a

            a_dw = torch.matmul(a_encodings_valid.t(), a_flat)
            v_dw_a = torch.matmul(a_encodings_valid.t(), v_flat)
            t_dw_a = torch.matmul(a_encodings_valid.t(), t_flat)

            a_segment_update = (0.2 * v_dw_a) + (0.2 * t_dw_a) + (0.6 * a_dw)
            with torch.no_grad():
                new_embedding_a = self.embedding.clone()
                self.ema_weight[:, D:2*D] = self.decay * self.ema_weight[:, D:2*D] + (1 - self.decay) * a_segment_update
                new_embedding_a[:, D:2*D] = self.ema_weight[:, D:2*D] / (self.ema_count_a.unsqueeze(-1))
                self.embedding = new_embedding_a

            # Text segment update
            self.ema_count_t = self.decay * self.ema_count_t + (1 - self.decay) * (torch.sum(t_encodings_valid, dim=0))
            n_t = torch.sum(self.ema_count_t)
            self.ema_count_t = (self.ema_count_t + self.epsilon) / (n_t + M * self.epsilon) * n_t

            t_dw = torch.matmul(t_encodings_valid.t(), t_flat)
            v_dw_t = torch.matmul(t_encodings_valid.t(), v_flat)
            a_dw_t = torch.matmul(t_encodings_valid.t(), a_flat)

            t_segment_update = (0.2 * v_dw_t) + (0.2 * a_dw_t) + (0.6 * t_dw)
            with torch.no_grad():
                new_embedding_t = self.embedding.clone()
                self.ema_weight[:, 2*D:] = self.decay * self.ema_weight[:, 2*D:] + (1 - self.decay) * t_segment_update
                new_embedding_t[:, 2*D:] = self.ema_weight[:, 2*D:] / (self.ema_count_t.unsqueeze(-1))
                self.embedding = new_embedding_t

            # Hierarchical influence between segments
            with torch.no_grad():
                new_embedding = self.embedding.clone()
                '''T -> A -> V''' 
                # new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])
                # new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, 2*D:])
                # new_embedding[:, :D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                # self.embedding = new_embedding

                '''V -> A -> T'''
                new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])
                self.embedding = new_embedding

            # Segment alignment metric
            new_embedding = self.embedding.clone()
            video_embedding_new = new_embedding[:, :D]         
            audio_embedding_new = new_embedding[:, D:2*D] 
            text_embedding_new = new_embedding[:, 2*D:]
            video_segments_norm = F.normalize(video_embedding_new, p=2, dim=1)
            audio_segments_norm = F.normalize(audio_embedding_new, p=2, dim=1)
            text_segments_norm = F.normalize(text_embedding_new, p=2, dim=1)
            temperature = 0.1
            
            va_similarity = torch.matmul(video_segments_norm, audio_segments_norm.t()) / temperature
            va_positive = torch.diag(va_similarity)
            va_logsumexp = torch.logsumexp(va_similarity, dim=1)
            va_loss = torch.mean(-va_positive + va_logsumexp)
            
            vt_similarity = torch.matmul(video_segments_norm, text_segments_norm.t()) / temperature
            vt_positive = torch.diag(vt_similarity)
            vt_logsumexp = torch.logsumexp(vt_similarity, dim=1)
            vt_loss = torch.mean(-vt_positive + vt_logsumexp)
            
            at_similarity = torch.matmul(audio_segments_norm, text_segments_norm.t()) / temperature
            at_positive = torch.diag(at_similarity)
            at_logsumexp = torch.logsumexp(at_similarity, dim=1)
            at_loss = torch.mean(-at_positive + at_logsumexp)
            
            segment_loss_raw = (va_loss + vt_loss + at_loss) / 3
            segment_loss = 0.5 * segment_loss_raw

            self.ema_count = self.ema_count_v + self.ema_count_a + self.ema_count_t

            # Dead vector recovery
            self.unactivated_count = self.unactivated_count + 1
            for indice in a_indices_valid:
                self.unactivated_count[indice.item()] = 0
            for indice in v_indices_valid:
                self.unactivated_count[indice.item()] = 0
            for indice in t_indices_valid:
                self.unactivated_count[indice.item()] = 0
                
            activated_indices = []
            unactivated_indices = []
            for i, x in enumerate(self.unactivated_count):
                if x > 300:
                    unactivated_indices.append(i)
                    self.unactivated_count[i] = 0
                elif x >= 0 and x < 100:
                    activated_indices.append(i)
                    
            if activated_indices and unactivated_indices:
                activated_quantized = F.embedding(
                    torch.tensor(activated_indices, dtype=torch.int32, device=self.embedding.device), 
                    self.embedding
                )
                for i in unactivated_indices:
                    random_idx = random.randint(0, len(activated_indices)-1)
                    self.embedding[i] = activated_quantized[random_idx] + torch.randn_like(self.embedding[i]) * 0.001
        else:
            segment_loss = torch.tensor(0.0, device=audio_semantic.device)

        # CMCM loss
        cmcm_loss = 0.5 * (Lcmcm_av + Lcmcm_at + Lcmcm_tv)

        # Content-aware commitment losses
        if attention_mask is not None:
            mask_expanded_256 = attention_mask.unsqueeze(-1).expand_as(video_semantic)
            
            # Apply masks to semantic and quantized features
            video_semantic_masked = video_semantic * mask_expanded_256
            audio_semantic_masked = audio_semantic * mask_expanded_256
            text_semantic_masked = text_semantic * mask_expanded_256
            
            v_quantized_masked = v_quantized_segment.detach() * mask_expanded_256
            a_quantized_masked = a_quantized_segment.detach() * mask_expanded_256
            t_quantized_masked = t_quantized_segment.detach() * mask_expanded_256
            
            # Cross-modal quantized features for cross-commitment losses
            va_quantized_masked = a_quantized_segment.detach() * mask_expanded_256
            vt_quantized_masked = t_quantized_segment.detach() * mask_expanded_256
            av_quantized_masked = v_quantized_segment.detach() * mask_expanded_256
            at_quantized_masked = t_quantized_segment.detach() * mask_expanded_256
            ta_quantized_masked = a_quantized_segment.detach() * mask_expanded_256
            tv_quantized_masked = v_quantized_segment.detach() * mask_expanded_256
            
            # Compute squared errors
            v_e_squared_error = (video_semantic_masked - v_quantized_masked) ** 2
            va_e_squared_error = (video_semantic_masked - va_quantized_masked) ** 2
            vt_e_squared_error = (video_semantic_masked - vt_quantized_masked) ** 2
            
            a_e_squared_error = (audio_semantic_masked - a_quantized_masked) ** 2
            av_e_squared_error = (audio_semantic_masked - av_quantized_masked) ** 2
            at_e_squared_error = (audio_semantic_masked - at_quantized_masked) ** 2
            
            t_e_squared_error = (text_semantic_masked - t_quantized_masked) ** 2
            ta_e_squared_error = (text_semantic_masked - ta_quantized_masked) ** 2
            tv_e_squared_error = (text_semantic_masked - tv_quantized_masked) ** 2
            
            # Normalize by number of valid elements
            num_valid_elements = mask_expanded_256.sum()
            
            v_e_latent_loss = v_e_squared_error.sum() / (num_valid_elements + 1e-6)
            va_e_latent_loss = va_e_squared_error.sum() / (num_valid_elements + 1e-6)
            vt_e_latent_loss = vt_e_squared_error.sum() / (num_valid_elements + 1e-6)
            
            a_e_latent_loss = a_e_squared_error.sum() / (num_valid_elements + 1e-6)
            av_e_latent_loss = av_e_squared_error.sum() / (num_valid_elements + 1e-6)
            at_e_latent_loss = at_e_squared_error.sum() / (num_valid_elements + 1e-6)
            
            t_e_latent_loss = t_e_squared_error.sum() / (num_valid_elements + 1e-6)
            ta_e_latent_loss = ta_e_squared_error.sum() / (num_valid_elements + 1e-6)
            tv_e_latent_loss = tv_e_squared_error.sum() / (num_valid_elements + 1e-6)
            
        else:
            v_e_latent_loss = F.mse_loss(video_semantic, v_quantized_segment.detach())
            va_e_latent_loss = F.mse_loss(video_semantic, a_quantized_segment.detach())
            vt_e_latent_loss = F.mse_loss(video_semantic, t_quantized_segment.detach())
            
            a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized_segment.detach())
            av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized_segment.detach())
            at_e_latent_loss = F.mse_loss(audio_semantic, t_quantized_segment.detach())
            
            t_e_latent_loss = F.mse_loss(text_semantic, t_quantized_segment.detach())
            ta_e_latent_loss = F.mse_loss(text_semantic, a_quantized_segment.detach())
            tv_e_latent_loss = F.mse_loss(text_semantic, v_quantized_segment.detach())

        # Combine commitment losses
        v_loss = (self.commitment_cost * 2.0 * v_e_latent_loss) + (0.5*self.commitment_cost * va_e_latent_loss) + (0.5*self.commitment_cost * vt_e_latent_loss)
        a_loss = (self.commitment_cost * 2.0 * a_e_latent_loss) + (0.5*self.commitment_cost * av_e_latent_loss) + (0.5*self.commitment_cost * at_e_latent_loss)
        t_loss = (self.commitment_cost * 2.0 * t_e_latent_loss) + (0.5*self.commitment_cost * ta_e_latent_loss) + (0.5*self.commitment_cost * tv_e_latent_loss)
        
        # Straight-through estimator with padding safety
        if attention_mask is not None:
            mask_expanded_256 = attention_mask.unsqueeze(-1)
            video_semantic_masked = video_semantic * mask_expanded_256
            audio_semantic_masked = audio_semantic * mask_expanded_256  
            text_semantic_masked = text_semantic * mask_expanded_256
        else:
            video_semantic_masked = video_semantic
            audio_semantic_masked = audio_semantic
            text_semantic_masked = text_semantic

        v_full_continuous = torch.cat([video_semantic_masked, audio_semantic_masked, text_semantic_masked], dim=-1)
        a_full_continuous = torch.cat([video_semantic_masked, audio_semantic_masked, text_semantic_masked], dim=-1)
        t_full_continuous = torch.cat([video_semantic_masked, audio_semantic_masked, text_semantic_masked], dim=-1)

        # Apply straight-through estimator
        v_quantized_segment = video_semantic + (v_quantized_segment - video_semantic).detach()
        a_quantized_segment = audio_semantic + (a_quantized_segment - audio_semantic).detach()
        t_quantized_segment = text_semantic + (t_quantized_segment - text_semantic).detach()
        
        v_full_vectors = v_full_continuous + (v_full_vectors - v_full_continuous).detach()
        a_full_vectors = a_full_continuous + (a_full_vectors - a_full_continuous).detach()
        t_full_vectors = t_full_continuous + (t_full_vectors - t_full_continuous).detach()

        # Content-aware perplexity computation
        if attention_mask is not None and num_valid_positions > 0:
            v_avg_probs = torch.mean(v_encodings_valid, dim=0)
            a_avg_probs = torch.mean(a_encodings_valid, dim=0)
            t_avg_probs = torch.mean(t_encodings_valid, dim=0)
        else:
            v_avg_probs = torch.ones(M, device=video_semantic.device) / M
            a_avg_probs = torch.ones(M, device=audio_semantic.device) / M
            t_avg_probs = torch.ones(M, device=text_semantic.device) / M

        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        t_perplexity = torch.exp(-torch.sum(t_avg_probs * torch.log(t_avg_probs + 1e-10)))

        return  v_full_vectors, v_quantized_segment,\
                a_full_vectors, a_quantized_segment,\
                t_full_vectors, t_quantized_segment,\
                v_loss, a_loss, t_loss, v_perplexity, a_perplexity, t_perplexity,\
                equal_num, cmcm_loss, segment_loss