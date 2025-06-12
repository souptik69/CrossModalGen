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

class Video_Semantic_Encoder(nn.Module):
    def __init__(self, video_dim, video_output_dim):
        super(Video_Semantic_Encoder, self).__init__()
        self.reduction = 8
        self.aver_pool = nn.AdaptiveAvgPool2d(1)
        self.se_layer = nn.Sequential(
            nn.Linear(video_dim, video_dim // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(video_dim // self.reduction, video_dim, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.affine_video_ave = nn.Linear(video_dim, video_dim // 2)
        self.affine_video_self = nn.Linear(video_dim, video_dim // 2)
        self.ave_v_att = nn.Linear(video_dim // 2, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        self.spatial_conv_stack = nn.Sequential(
            # First conv layer: 512 → 1024 channels, 7×7 → 5×5
            nn.Conv2d(video_dim, video_output_dim // 2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(video_output_dim // 2),
            nn.ReLU(inplace=True),
            
            # Second conv layer: 1024 → 2048 channels, 5×5 → 3×3
            nn.Conv2d(video_output_dim // 2, video_output_dim , kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(video_output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, video_feat):
        batch, length, h, w, v_dim = video_feat.size()
        video_feat = video_feat.reshape(batch * length, h, w, v_dim)
        average_video_feat = video_feat.permute(0, 3, 1, 2)
        average_video_feat = self.aver_pool(average_video_feat).view(batch * length, v_dim)
        average_attention = self.se_layer(average_video_feat).view(batch * length, v_dim, 1, 1).permute(0, 2, 3, 1)
        video_channel_att = video_feat * average_attention.expand_as(video_feat) + video_feat

        video_average = self.relu(self.affine_video_ave(average_video_feat)).unsqueeze(-2)
        self_video_att_feat = video_channel_att.reshape((batch * length, -1, v_dim))
        self_video_att_query = self.relu(self.affine_video_self(self_video_att_feat))
        self_query = self_video_att_query * video_average
        self_spatial_att_maps = self.softmax(self.tanh(self.ave_v_att(self_query))).transpose(2, 1)

        self_att_feat = torch.bmm(self_spatial_att_maps,
                                  video_channel_att.view(batch * length, -1, v_dim)).squeeze().reshape(batch, length,
                                                                                                       v_dim)
        
        spatial_att_reshaped = self_spatial_att_maps.permute(0, 2, 1)  # [batch*time,7x7, 1]
        spatial_att_expanded = spatial_att_reshaped.expand(-1, -1, v_dim) # [batch*length, 49, 512]
        
        spatially_attended_features = self_video_att_feat * spatial_att_expanded #[batch*length, 49, 512]
        spatial_preserved_reshaped = spatially_attended_features.view(batch * length, h, w, v_dim) # [batch*time,7,7,512]
        spatially_attended_reshaped = spatial_preserved_reshaped.permute(0, 3, 1, 2) # [batch*time,512, 7,7]
        
        transformed_features = self.spatial_conv_stack(spatially_attended_reshaped) #[batch*time, 2048, 3,3 ]
        _, channel_dim, out_height, out_width = transformed_features.size() 
        spatial_preserved_features = transformed_features.permute(0, 2, 3, 1).contiguous() #[batch*time, 3, 3, 2048]
        spatial_preserved_features = spatial_preserved_features.view(batch, length, out_height, out_width, channel_dim) #[batch, time, 3, 3, 2048]

        return self_att_feat, spatial_preserved_features
    

class Audio_Encoder(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Audio_Encoder, self).__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feat):
        return self.relu(self.audio_linear(audio_feat))


class Audio_Encoder_1(nn.Module):
    def __init__(self, audio_dim, hidden_dim, noise_std=0.1):
        super(Audio_Encoder_1, self).__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feat):
        if self.training:
            noise = torch.randn_like(audio_feat) * self.noise_std
            audio_feat = audio_feat + noise   
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


class Text_Encoder_1(nn.Module):
    def __init__(self, text_dim, hidden_dim, noise_std=0.1):
        super(Text_Encoder_1, self).__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text_feat):
        if self.training:
            noise = torch.randn_like(text_feat) * self.noise_std
            text_feat = text_feat + noise 
        return self.relu(self.text_linear(text_feat))



class AV_VQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim,  video_output_dim, n_embeddings, embedding_dim):
        super(AV_VQVAE_Encoder, self).__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_dim = 256


        self.Cross_quantizer = Cross_VQEmbeddingEMA_AV_hierarchical_softmax(n_embeddings, self.hidden_dim)
        # self.Cross_quantizer = Cross_VQEmbeddingEMA_AV_hierarchical_1(n_embeddings, self.hidden_dim)
        

        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim, video_output_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, self.hidden_dim)
        # self.Audio_encoder = Audio_Encoder_1(audio_dim, self.hidden_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()  
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result) 
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous() # [batch, 10, 256]
        out_vq, audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        return out_vq, audio_vq  # [batch, 10, 512], [batch, 10, 256]

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result, video_spatial = self.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, 10, 256]
        out_vq, video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        return out_vq, video_vq  # [batch,10, 512], [batch, 10, 256]

    def forward(self, audio_feat, video_feat, epoch):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        
        video_semantic_result, video_spatial = self.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]
        
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        audio_encoder_result = self.Audio_encoder(audio_feat)

        out_vq_video, video_vq, out_vq_audio, audio_vq, \
        video_embedding_loss, audio_embedding_loss,\
        video_perplexity, audio_perplexity,\
        equal_num, cmcm_loss, segment_loss = self.Cross_quantizer(audio_semantic_result, video_semantic_result, epoch)

        return audio_semantic_result, audio_encoder_result, video_semantic_result, video_spatial,\
               out_vq_video, video_vq, out_vq_audio, audio_vq, video_embedding_loss, audio_embedding_loss,\
                video_perplexity, audio_perplexity, equal_num, cmcm_loss, segment_loss




class AVT_VQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim,  video_output_dim, n_embeddings, embedding_dim):
        super(AVT_VQVAE_Encoder, self).__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = 256
        self.Cross_quantizer = Cross_VQEmbeddingEMA_AVT_hierarchical(n_embeddings, self.hidden_dim)
        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim, video_output_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, self.hidden_dim)
        # self.Audio_encoder = Audio_Encoder_1(audio_dim, self.hidden_dim)
        self.Text_encoder = Text_Encoder(text_dim, self.hidden_dim)
        # self.Text_encoder = Text_Encoder_1(text_dim, self.hidden_dim)
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
        video_semantic_result, video_spatial = self.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, 10, 256]
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
        
        video_semantic_result, video_spatial = self.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]
        
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim = 256]

        audio_encoder_result = self.Audio_encoder(audio_feat)
        text_encoder_result = self.Text_encoder(text_feat)

        out_vq_video, video_vq, out_vq_audio, audio_vq, \
        out_vq_text, text_vq, \
        video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, \
        equal_num, cmcm_loss, segment_loss = self.Cross_quantizer(audio_semantic_result, video_semantic_result, text_semantic_result, epoch)

        return audio_semantic_result, audio_encoder_result, video_semantic_result, video_spatial, \
               text_semantic_result, text_encoder_result, \
               out_vq_video, video_vq, out_vq_audio, audio_vq,\
               out_vq_text, text_vq, video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
               video_perplexity, audio_perplexity, text_perplexity, equal_num, cmcm_loss, segment_loss



""" class_num AVE:28  VGGSOUND:141+1 """
class Semantic_Decoder(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder, self).__init__()
        # self.fusion_layer = nn.Linear(input_dim, input_dim // 2)
        self.linear = nn.Linear(input_dim , input_dim)
        self.event_classifier = nn.Linear(input_dim , class_num)  

    def forward(self, input_vq):
        # fused_feat = self.fusion_layer(input_vq)  
        input_feat = self.linear(input_vq)
        input_feat, _ = input_feat.max(1)  
        class_logits = self.event_classifier(input_feat)
        return class_logits



""" class_num AVVP:25+1(negative label) AVE_AVVP:12+1 """
class Semantic_Decoder_AVVP(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder_AVVP, self).__init__()
        # self.fusion_layer = nn.Linear(input_dim, input_dim // 2)
        self.linear = nn.Linear(input_dim , input_dim )
        self.event_classifier = nn.Linear(input_dim , class_num)   

    def forward(self, input_vq):
        # fused_feat = self.fusion_layer(input_vq)  
        input_feat = self.linear(input_vq)
        class_logits = self.event_classifier(input_feat)
        return class_logits



""" class_num AVVP:25+1(negative label) AVE_AVVP:12+1 """
class Semantic_Decoder_AVVP_1(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder_AVVP_1, self).__init__()
        # self.fusion_layer = nn.Linear(input_dim, input_dim // 2)
        self.linear = nn.Linear(input_dim , input_dim )
        self.fusion_layer = nn.Linear(input_dim, input_dim // 2)
        self.linear1 = nn.Linear(input_dim // 2 , input_dim // 2)
        self.event_classifier = nn.Linear(input_dim // 2 , class_num)    

    def forward(self, input_vq):
        # fused_feat = self.fusion_layer(input_vq)  
        input_feat = self.linear(input_vq)
        fused_feat = self.fusion_layer(input_feat)
        input_feat_1 = self.linear1(fused_feat)
        class_logits = self.event_classifier(input_feat_1)
        return class_logits
    
""" class_num AVVP:25+1(negative label) AVE_AVVP:12+1 """
class Semantic_Decoder_AVVP_2(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder_AVVP_2, self).__init__()
        # self.fusion_layer = nn.Linear(input_dim, input_dim // 2)
        self.linear = nn.Linear(input_dim , input_dim )
        self.fusion_layer_1 = nn.Linear(input_dim, input_dim // 2)
        self.linear1 = nn.Linear(input_dim // 2 , input_dim // 2)
        self.fusion_layer_2 = nn.Linear(input_dim // 2, input_dim // 3)
        self.linear2 = nn.Linear(input_dim // 3 , input_dim // 3)
        self.event_classifier = nn.Linear(input_dim // 3 , class_num)    

    def forward(self, input_vq):
        # fused_feat = self.fusion_layer(input_vq)  
        input_feat = self.linear(input_vq)
        fused_feat_1 = self.fusion_layer_1(input_feat)
        input_feat_1 = self.linear1(fused_feat_1)
        fused_feat_2 = self.fusion_layer_2(input_feat_1)
        input_feat_2 = self.linear2(fused_feat_2)
        class_logits = self.event_classifier(input_feat_2)
        return class_logits




class Audio_Decoder(nn.Module):
    def __init__(self, output_dim = 128, vq_dim = 256):
        super(Audio_Decoder, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.audio_rec = nn.Linear(vq_dim * 2, output_dim)
        self.audio_linear_1 = nn.Linear(vq_dim * 2, vq_dim)
        self.audio_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, audio_encoder_result, audio_vq):
        audio_vq_1 = self.audio_linear_1(audio_vq)
        audio_vq_result = self.audio_linear(audio_vq_1)
        audio_encoder_result = torch.cat([audio_vq_result, audio_encoder_result], dim=2)
        audio_decoder_result = self.audio_rec(audio_encoder_result)
        return audio_decoder_result
    

class Audio_Decoder_1(nn.Module):
    def __init__(self, output_dim , vq_dim):
        super(Audio_Decoder_1, self).__init__()
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
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Video_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        kernel = 3
        stride = 1

        self.inverse_conv_block = nn.Sequential(
            nn.ConvTranspose2d(input_dim + vq_dim, input_dim // 2, kernel_size=kernel, stride=stride, padding=0),
            ResidualStack(input_dim // 2, input_dim // 2, input_dim // 2, 1),
            nn.ConvTranspose2d(input_dim // 2, output_dim, kernel_size=kernel, stride=stride, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim, output_dim, kernel_size=1, stride=1, padding=0)
        )
        self.video_linear_1 = nn.Linear(vq_dim * 2, vq_dim)
        self.video_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, video_spatial, video_vq):
        batch, length, h1, w1, dim = video_spatial.size()
        video_vq_1 = self.video_linear_1(video_vq)
        video_vq_result = self.video_linear(video_vq_1).unsqueeze(2).unsqueeze(3)
        video_vq_result = video_vq_result.repeat(1, 1, h1, w1, 1).reshape(batch * length, h1, w1, -1)
        video_spatial = video_spatial.reshape(batch * length, h1, w1, dim)
        video_spatial = torch.cat([video_vq_result, video_spatial], dim=3)
        video_spatial = video_spatial.permute(0, 3, 1, 2)

        video_recon_result = self.inverse_conv_block(video_spatial)
        _, dim, H, W = video_recon_result.size()
        video_recon_result = video_recon_result.reshape(batch, length, dim, H, W)
        video_recon_result = video_recon_result.permute(0, 1, 3, 4, 2)

        return video_recon_result


class Video_Decoder_1(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Video_Decoder_1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        kernel = 3
        stride = 1

        self.inverse_conv_block = nn.Sequential(
            nn.ConvTranspose2d(input_dim + vq_dim, input_dim // 2, kernel_size=kernel, stride=stride, padding=0),
            ResidualStack(input_dim // 2, input_dim // 2, input_dim // 2, 1),
            nn.ConvTranspose2d(input_dim // 2, output_dim, kernel_size=kernel, stride=stride, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim, output_dim, kernel_size=1, stride=1, padding=0)
        )
        self.video_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.video_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, video_spatial, video_vq):
        batch, length, h1, w1, dim = video_spatial.size()
        video_vq_1 = self.video_linear_1(video_vq)
        video_vq_result = self.video_linear(video_vq_1).unsqueeze(2).unsqueeze(3)
        video_vq_result = video_vq_result.repeat(1, 1, h1, w1, 1).reshape(batch * length, h1, w1, -1)
        video_spatial = video_spatial.reshape(batch * length, h1, w1, dim)
        video_spatial = torch.cat([video_vq_result, video_spatial], dim=3)
        video_spatial = video_spatial.permute(0, 3, 1, 2)

        video_recon_result = self.inverse_conv_block(video_spatial)
        _, dim, H, W = video_recon_result.size()
        video_recon_result = video_recon_result.reshape(batch, length, dim, H, W)
        video_recon_result = video_recon_result.permute(0, 1, 3, 4, 2)

        return video_recon_result



class AV_VQVAE_Decoder(nn.Module):
    def __init__(self, audio_dim, video_dim, video_output_dim):
        super(AV_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 256 #embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.video_output_dim = video_output_dim
        self.Video_decoder = Video_Decoder(video_output_dim, video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder(audio_dim, self.hidden_dim)
        self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim * 2, class_num=142)
        self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim * 2, class_num=142)

        ### Equal distribution ###
        # self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim , class_num=142)
        # self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim , class_num=142)
        ### Equal distribution ###

    def forward(self, audio_feat, video_feat, audio_encoder_result, video_spatial, out_vq_audio, audio_vq, out_vq_video, video_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_recon_result = self.Video_decoder(video_spatial, out_vq_video)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        video_class = self.video_semantic_decoder(out_vq_video)
        audio_class = self.audio_semantic_decoder(out_vq_audio)

        ### Equal distribution ###
        # video_class = self.video_semantic_decoder(video_vq)
        # audio_class = self.audio_semantic_decoder(audio_vq)
        ### Equal distribution ###

        return audio_recon_loss, video_recon_loss, audio_class, video_class



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





# Code book split + Feature-Level Meta Learning + Hierarchical segment level Meta Learning softmax
class Cross_VQEmbeddingEMA_AV_hierarchical_softmax(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AV_hierarchical_softmax, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim * 2)
        
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_v", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_a", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))# unactivated:-1

        # Meta weights or parameters evolutionary weights
        self.register_buffer("modal_weights", torch.zeros(n_embeddings, 4))
        self.register_buffer("hier_weights", torch.zeros(n_embeddings, 4))

        # self.modal_weights = nn.Parameter(torch.zeros(n_embeddings, 4))
        # self.hier_weights = nn.Parameter(torch.zeros(n_embeddings, 4))


        log_ratio = math.log(0.75/0.25)  # ≈ 1.3863
        init_modality_weights = torch.zeros(n_embeddings, 4)
        init_modality_weights[:, 0] = log_ratio  # v->v weights
        init_modality_weights[:, 3] = log_ratio  # a->a weights
        self.modality_weights = nn.Parameter(init_modality_weights)

        # log_ratio_hier = math.log(0.7/0.3)  # ≈ 1.3863
        log_ratio_hier = math.log(0.5/0.5)
        init_hier_weights = torch.zeros(n_embeddings, 4)
        init_hier_weights[:, 0] = log_ratio_hier  # v->v weights
        init_hier_weights[:, 3] = log_ratio_hier  # a->a weights
        self.hierarchical_weights = nn.Parameter(init_hier_weights)



        # logit_01 = math.log(0.1/(1-0.1))  # ≈ -2.2
        # init_hier_weights = torch.full((n_embeddings, 2), logit_01)
        # self.hierarchical_weights = nn.Parameter(init_hier_weights)

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

        out_vq = F.embedding(v_indices, self.embedding)  # [BxT, 512]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 512]
        
        return out_vq, v_quantized   # [batch,10, 512], [batch, 10, 256]

    def Audio_vq_embedding(self, audio_semantic):
        B, T, D = audio_semantic.size()  # D = 256
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
     
        audio_embedding = self.embedding[:, D:]  # [n_embeddings, 256]

        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT]

        a_quantized = F.embedding(a_indices, audio_embedding)  # [BxT, 256]
        a_quantized = a_quantized.view_as(audio_semantic)  # [B, T, 256]

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()

        out_vq = F.embedding(a_indices, self.embedding)  # [BxT, 512]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 512]
        
        return out_vq, a_quantized   # [batch,10, 512], [batch, 10, 256]

    def forward(self, audio_semantic, video_semantic, epoch):
        M, D_total = self.embedding.size()  # M = num codebook vectors, D_total = 512
        B, T, D = audio_semantic.size()     # B = batch size, T = timesteps, D = 256


        modal_weights_video = F.softmax(self.modality_weights[:, 0:2], dim=1)
        modal_weights_audio = F.softmax(self.modality_weights[:, 2:4], dim=1)

        hier_weights_video = F.softmax(self.hierarchical_weights[:, 0:2], dim=1)
        hier_weights_audio = F.softmax(self.hierarchical_weights[:, 2:4], dim=1)


        with torch.no_grad():
            self.modal_weights[:, 0:2] = modal_weights_video
            self.modal_weights[:, 2:4] = modal_weights_audio

            self.hier_weights[:, 0:2] = hier_weights_video
            self.hier_weights[:, 2:4] = hier_weights_audio



        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        
        # CODEBOOK DIMENSION SPLIT FOR MODALITY
        video_embedding = self.embedding[:, :D]         # First 256 dims for video
        audio_embedding = self.embedding[:, D:]         # Second 256 dims for audio
    
        v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                v_flat, video_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]
        
        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                        torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        video_semantic.reshape(-1, D), video_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]
        
        a_distances_gradient = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                        torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        audio_semantic.reshape(-1, D), audio_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]

        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M]
        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M]

        v_ph = torch.reshape(v_ph, (B, T, M))  # [BxT, M] -> [B, T, M]
        a_ph = torch.reshape(a_ph, (B, T, M))  # [BxT, M] -> [B, T, M]

        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)

        EScode_sumdim1 = torch.sum(EScode, dim=1)
        Lcmcm_av = 0
        for i in range(B):
            Lcmcm_av -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm_av /= B

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT]

        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]

        v_quantized_segment = F.embedding(v_indices, video_embedding)  # [BxT, 256]
        a_quantized_segment = F.embedding(a_indices, audio_embedding)  # [BxT, 256]

        v_quantized_segment = v_quantized_segment.view_as(video_semantic)  # [B, T, 256]
        a_quantized_segment = a_quantized_segment.view_as(audio_semantic)  # [B, T, 256]

        v_full_vectors = F.embedding(v_indices, self.embedding)  # [BxT, 512]
        a_full_vectors = F.embedding(a_indices, self.embedding)  # [BxT, 512]

        v_full_vectors = v_full_vectors.view(B, T, D_total)  # [B, T, 512]
        a_full_vectors = a_full_vectors.view(B, T, D_total)  # [B, T, 512]

        if True:
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_reshape = a_indices.reshape(B, T)
            
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            
            equal_item = (v_indices_mode.values == a_indices_mode.values)
            equal_num = equal_item.sum()
        
        if self.training:

            # STEP 1: FEATURE-LEVEL CROSS-MODAL INFLUENCE
            
            # Video segment update (first half of embedding)
            self.ema_count_v = self.decay * self.ema_count_v + (1 - self.decay) * (torch.sum(v_encodings, dim=0))
            n_v = torch.sum(self.ema_count_v)
            self.ema_count_v = (self.ema_count_v + self.epsilon) / (n_v + M * self.epsilon) * n_v
            v_dw = torch.matmul(v_encodings.t(), v_flat)    # Video encodings with Video features → video segment
            a_dw_v = torch.matmul(v_encodings.t(), a_flat)  # Video encodings with Audio features → video segment
            v_segment_update = self.modal_weights[:, 0].unsqueeze(1) * v_dw + self.modal_weights[:, 1].unsqueeze(1) * a_dw_v
            with torch.no_grad():
                new_embedding_v = self.embedding.clone()
                self.ema_weight[:, :D] = self.decay * self.ema_weight[:, :D] + (1 - self.decay) * v_segment_update
                new_embedding_v[:, :D] = self.ema_weight[:, :D] / (self.ema_count_v.unsqueeze(-1))
                self.embedding = new_embedding_v
            # v_segment_update = 0.75 * v_dw + 0.25 * a_dw_v
            
            # Audio segment update (second half of embedding)
            self.ema_count_a = self.decay * self.ema_count_a + (1 - self.decay) * (torch.sum(a_encodings, dim=0))
            n_a = torch.sum(self.ema_count_a)
            self.ema_count_a = (self.ema_count_a + self.epsilon) / (n_a + M * self.epsilon) * n_a
            a_dw = torch.matmul(a_encodings.t(), a_flat)   # Audio encodings with Audio features → audio segment
            v_dw_a = torch.matmul(a_encodings.t(), v_flat) # Audio encodings with video features → audio segment
            a_segment_update = self.modal_weights[:, 2].unsqueeze(1) * v_dw_a + self.modal_weights[:, 3].unsqueeze(1) * a_dw
            with torch.no_grad():
                new_embedding_a = self.embedding.clone()
                self.ema_weight[:, D:] = self.decay * self.ema_weight[:, D:] + (1 - self.decay) * a_segment_update
                new_embedding_a[:, D:] = self.ema_weight[:, D:] / (self.ema_count_a.unsqueeze(-1))
                self.embedding = new_embedding_a
            # a_segment_update = 0.25 * v_dw_a + 0.75 * a_dw

            # STEP 2: HIERARCHICAL INFLUENCE BETWEEN SEGMENTS
            with torch.no_grad():
                new_embedding = self.embedding.clone()
                video_embedding_new = new_embedding[:, :D]         
                audio_embedding_new = new_embedding[:, D:]        

                new_embedding[:, D:] = self.hier_weights[:, 3].unsqueeze(1) * audio_embedding_new + \
                                        self.hier_weights[:, 2].unsqueeze(1) * video_embedding_new    #Audio = (1/2)Video + (1/2)Audio 
                new_embedding[:, :D] = self.hier_weights[:, 0].unsqueeze(1) * video_embedding_new + \
                                        self.hier_weights[:, 1].unsqueeze(1) *  audio_embedding_new   #Video = (3/4)Video + (1/4)Audio

                ### Equal distribution ###
                # audio_emb = self.hier_weights[:, 3].unsqueeze(1) * audio_embedding_new + \
                #                         self.hier_weights[:, 2].unsqueeze(1) * video_embedding_new     
                # video_emb = self.hier_weights[:, 0].unsqueeze(1) * video_embedding_new + \
                #                         self.hier_weights[:, 1].unsqueeze(1) *  audio_embedding_new   

                # new_embedding[:, D:] = audio_emb   #Audio = (1/2)Video + (1/2)Audio
                # new_embedding[:, :D] = video_emb   #Video = (1/2)Video + (1/2)Audio
                ### Equal distribution ###
                
             
                self.embedding = new_embedding

            # SEGMENT ALIGNMENT Metric
            new_embedding = self.embedding.clone()
            video_embedding_new = new_embedding[:, :D]         
            audio_embedding_new = new_embedding[:, D:]  
            video_segments_norm = F.normalize(video_embedding_new, p=2, dim=1)  # [M, D]
            audio_segments_norm = F.normalize(audio_embedding_new, p=2, dim=1)  # [M, D]
            similarity_matrix = torch.matmul(video_segments_norm, audio_segments_norm.t())  # [M, M]
            temperature = 0.1
            similarity_matrix = similarity_matrix / temperature
            positive_logits = torch.diag(similarity_matrix)  # Diagonal elements = positive pairs [M]
            logsumexp_logits = torch.logsumexp(similarity_matrix, dim=1)  # [M]
            segment_loss_raw = torch.mean(-positive_logits + logsumexp_logits)
            segment_loss = 0.5 * segment_loss_raw

            self.ema_count = self.ema_count_v +self.ema_count_a

            # Dead Codebook vectors Alleviation with modality and hierarchical weights
            self.unactivated_count = self.unactivated_count + 1
            for indice in a_indices:
                self.unactivated_count[indice.item()] = 0
            for indice in v_indices:
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

        cmcm_loss = 0.5 * Lcmcm_av

        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized_segment.detach())
        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized_segment.detach())

        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized_segment.detach())  
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized_segment.detach())  

        v_loss = self.commitment_cost * (2.0 * v_e_latent_loss + 0.5 * va_e_latent_loss)
        a_loss = self.commitment_cost * (2.0 * a_e_latent_loss + 0.5 * av_e_latent_loss)

        v_quantized_segment = video_semantic + (v_quantized_segment - video_semantic).detach()    #[B,T,D = 256]
        a_quantized_segment = audio_semantic + (a_quantized_segment - audio_semantic).detach()    #[B,T,D = 256]
        
        ################### Use v_full_vectors for cross modal reconstruction gradients #################

        v_full_continuous = torch.cat([video_semantic, audio_semantic], dim=-1)
        a_full_continuous = torch.cat([video_semantic, audio_semantic], dim=-1)
        v_full_vectors = v_full_continuous + (v_full_vectors - v_full_continuous).detach()  #[B,T,D = 512]
        a_full_vectors = a_full_continuous + (a_full_vectors - a_full_continuous).detach()  #[B,T,D = 512]

        ################## Use v_full_vectors for cross modal reconstruction gradients ###################


        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))

        return  v_full_vectors, v_quantized_segment,\
                a_full_vectors, a_quantized_segment,\
                v_loss, a_loss, v_perplexity, a_perplexity,\
                equal_num, cmcm_loss, segment_loss  



# Code book split + Feature-Level Meta Learning + Hierarchical segment level non - Meta Learning
class Cross_VQEmbeddingEMA_AV_hierarchical_1(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AV_hierarchical_1, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim * 2)
        
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_v", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_a", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))# unactivated:-1

        # Meta weights or parameters
        self.register_buffer("modal_weights", torch.zeros(n_embeddings, 4))
        log_ratio = math.log(0.8/0.2)  # ≈ 1.3863
        init_modality_weights = torch.zeros(n_embeddings, 4)
        init_modality_weights[:, 0] = log_ratio  # v->v weights
        init_modality_weights[:, 3] = log_ratio  # a->a weights
        self.modality_weights = nn.Parameter(init_modality_weights)

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

        out_vq = F.embedding(v_indices, self.embedding)  # [BxT, 512]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 512]
        
        return out_vq, v_quantized   # [batch,10, 512], [batch, 10, 256]

    def Audio_vq_embedding(self, audio_semantic):
        B, T, D = audio_semantic.size()  # D = 256
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
     
        audio_embedding = self.embedding[:, D:]  # [n_embeddings, 256]

        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT]

        a_quantized = F.embedding(a_indices, audio_embedding)  # [BxT, 256]
        a_quantized = a_quantized.view_as(audio_semantic)  # [B, T, 256]

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()

        out_vq = F.embedding(a_indices, self.embedding)  # [BxT, 512]
        out_vq = out_vq.view(B, T, -1)  # [B, T, 512]
        
        return out_vq, a_quantized   # [batch,10, 512], [batch, 10, 256]

    def forward(self, audio_semantic, video_semantic, epoch):
        M, D_total = self.embedding.size()  # M = num codebook vectors, D_total = 512
        B, T, D = audio_semantic.size()     # B = batch size, T = timesteps, D = 256


        modal_weights_video = F.softmax(self.modality_weights[:, 0:2], dim=1)
        modal_weights_audio = F.softmax(self.modality_weights[:, 2:4], dim=1)

        with torch.no_grad():
            self.modal_weights[:, 0:2] = modal_weights_video
            self.modal_weights[:, 2:4] = modal_weights_audio


        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        
        # CODEBOOK DIMENSION SPLIT FOR MODALITY
        video_embedding = self.embedding[:, :D]         # First 256 dims for video
        audio_embedding = self.embedding[:, D:]         # Second 256 dims for audio
    
        v_distances = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                v_flat, video_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]
        
        a_distances = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                a_flat, audio_embedding.t(),
                                alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(video_embedding ** 2, dim=1) +
                                        torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        video_semantic.reshape(-1, D), video_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]
        
        a_distances_gradient = torch.addmm(torch.sum(audio_embedding ** 2, dim=1) +
                                        torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                        audio_semantic.reshape(-1, D), audio_embedding.t(),
                                        alpha=-2.0, beta=1.0)  # [BxT, M]

        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M]
        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M]

        v_ph = torch.reshape(v_ph, (B, T, M))  # [BxT, M] -> [B, T, M]
        a_ph = torch.reshape(a_ph, (B, T, M))  # [BxT, M] -> [B, T, M]

        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)

        EScode_sumdim1 = torch.sum(EScode, dim=1)
        Lcmcm_av = 0
        for i in range(B):
            Lcmcm_av -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm_av /= B

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT]

        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]

        v_quantized_segment = F.embedding(v_indices, video_embedding)  # [BxT, 256]
        a_quantized_segment = F.embedding(a_indices, audio_embedding)  # [BxT, 256]

        v_quantized_segment = v_quantized_segment.view_as(video_semantic)  # [B, T, 256]
        a_quantized_segment = a_quantized_segment.view_as(audio_semantic)  # [B, T, 256]

        v_full_vectors = F.embedding(v_indices, self.embedding)  # [BxT, 512]
        a_full_vectors = F.embedding(a_indices, self.embedding)  # [BxT, 512]

        v_full_vectors = v_full_vectors.view(B, T, D_total)  # [B, T, 512]
        a_full_vectors = a_full_vectors.view(B, T, D_total)  # [B, T, 512]

        if True:
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_reshape = a_indices.reshape(B, T)
            
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            
            equal_item = (v_indices_mode.values == a_indices_mode.values)
            equal_num = equal_item.sum()
        
        if self.training:

            # STEP 1: FEATURE-LEVEL CROSS-MODAL INFLUENCE
            
            # Video segment update (first half of embedding)
            self.ema_count_v = self.decay * self.ema_count_v + (1 - self.decay) * (torch.sum(v_encodings, dim=0))
            n_v = torch.sum(self.ema_count_v)
            self.ema_count_v = (self.ema_count_v + self.epsilon) / (n_v + M * self.epsilon) * n_v
            v_dw = torch.matmul(v_encodings.t(), v_flat)    # Video encodings with Video features → video segment
            a_dw_v = torch.matmul(v_encodings.t(), a_flat)  # Video encodings with Audio features → video segment
            v_segment_update = self.modal_weights[:, 0].unsqueeze(1) * v_dw + self.modal_weights[:, 1].unsqueeze(1) * a_dw_v
            with torch.no_grad():
                new_embedding_v = self.embedding.clone()
                self.ema_weight[:, :D] = self.decay * self.ema_weight[:, :D] + (1 - self.decay) * v_segment_update
                new_embedding_v[:, :D] = self.ema_weight[:, :D] / (self.ema_count_v.unsqueeze(-1))
                self.embedding = new_embedding_v
            # v_segment_update = 0.75 * v_dw + 0.25 * a_dw_v
            
            # Audio segment update (second half of embedding)
            self.ema_count_a = self.decay * self.ema_count_a + (1 - self.decay) * (torch.sum(a_encodings, dim=0))
            n_a = torch.sum(self.ema_count_a)
            self.ema_count_a = (self.ema_count_a + self.epsilon) / (n_a + M * self.epsilon) * n_a
            a_dw = torch.matmul(a_encodings.t(), a_flat)   # Audio encodings with Audio features → audio segment
            v_dw_a = torch.matmul(a_encodings.t(), v_flat) # Audio encodings with video features → audio segment
            a_segment_update = self.modal_weights[:, 2].unsqueeze(1) * v_dw_a + self.modal_weights[:, 3].unsqueeze(1) * a_dw
            with torch.no_grad():
                new_embedding_a = self.embedding.clone()
                self.ema_weight[:, D:] = self.decay * self.ema_weight[:, D:] + (1 - self.decay) * a_segment_update
                new_embedding_a[:, D:] = self.ema_weight[:, D:] / (self.ema_count_a.unsqueeze(-1))
                self.embedding = new_embedding_a
            # a_segment_update = 0.25 * v_dw_a + 0.75 * a_dw


            # STEP 2: HIERARCHICAL INFLUENCE BETWEEN SEGMENTS without meta parameters
            new_embedding = self.embedding.clone()
            video_embedding_new = new_embedding[:, :D]         
            audio_embedding_new = new_embedding[:, D:]         
            new_embedding[:, D:] = (0.5 * audio_embedding_new) + (0.5 * video_embedding_new)   #Audio = (1/2)Video + (1/2)Audio 
            new_embedding[:, :D] = (0.5 * video_embedding_new) + (0.5 *  audio_embedding_new)  #Video = (3/4)Video + (1/4)Audio
            self.embedding = new_embedding

            # SEGMENT ALIGNMENT Metric
            new_embedding = self.embedding.clone()
            video_embedding_new = new_embedding[:, :D]         
            audio_embedding_new = new_embedding[:, D:]  
            video_segments_norm = F.normalize(video_embedding_new, p=2, dim=1)  # [M, D]
            audio_segments_norm = F.normalize(audio_embedding_new, p=2, dim=1)  # [M, D]
            similarity_matrix = torch.matmul(video_segments_norm, audio_segments_norm.t())  # [M, M]
            temperature = 0.1
            similarity_matrix = similarity_matrix / temperature
            positive_logits = torch.diag(similarity_matrix)  # Diagonal elements = positive pairs [M]
            logsumexp_logits = torch.logsumexp(similarity_matrix, dim=1)  # [M]
            segment_loss_raw = torch.mean(-positive_logits + logsumexp_logits)
            segment_loss = 0.5 * segment_loss_raw

            self.ema_count = self.ema_count_v +self.ema_count_a

            # Dead Codebook vectors Alleviation with modality and hierarchical weights
            self.unactivated_count = self.unactivated_count + 1
            for indice in a_indices:
                self.unactivated_count[indice.item()] = 0
            for indice in v_indices:
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
                    with torch.no_grad():
                        modality_logit_noise = 0.1
                        self.modality_weights[i] = self.modality_weights[random_idx] + torch.randn_like(self.modality_weights[i]) * modality_logit_noise

        cmcm_loss = 0.5 * Lcmcm_av

        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized_segment.detach())
        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized_segment.detach())

        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized_segment.detach())  
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized_segment.detach())  

        v_loss = self.commitment_cost * (2.0 * v_e_latent_loss + 0.5 * va_e_latent_loss)
        a_loss = self.commitment_cost * (2.0 * a_e_latent_loss + 0.5 * av_e_latent_loss)

        v_quantized_segment = video_semantic + (v_quantized_segment - video_semantic).detach()    #[B,T,D = 256]
        a_quantized_segment = audio_semantic + (a_quantized_segment - audio_semantic).detach()    #[B,T,D = 256]

        ################## Use v_full_vectors for cross modal reconstruction gradients ###################

        v_full_continuous = torch.cat([video_semantic, audio_semantic], dim=-1)
        a_full_continuous = torch.cat([video_semantic, audio_semantic], dim=-1)
        v_full_vectors = v_full_continuous + (v_full_vectors - v_full_continuous).detach()  #[B,T,D = 512]
        a_full_vectors = a_full_continuous + (a_full_vectors - a_full_continuous).detach()  #[B,T,D = 512]

        ################## Use v_full_vectors for cross modal reconstruction gradients ###################


        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))

        return  v_full_vectors, v_quantized_segment,\
                a_full_vectors, a_quantized_segment,\
                v_loss, a_loss, v_perplexity, a_perplexity,\
                equal_num, cmcm_loss, segment_loss                     
    

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
                video_embedding_new = new_embedding[:, :D]         
                audio_embedding_new = new_embedding[:, D:2*D] 
                text_embedding_new = new_embedding[:, 2*D:]
                
                
                new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, 2*D:])  #Audio = (4/9)Video + (4/9)Audio + (1/9)Text
                new_embedding[:, :D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])     #Video = (16/27)Video + (7/27)Audio + (4/27)Text

                ### Equal distribution ###
                # text_emb = ((1/3) * video_embedding_new) + ((1/3) * audio_embedding_new) + ((1/3) * text_embedding_new)   
                # audio_emb = ((1/3) * video_embedding_new) + ((1/3) * audio_embedding_new) + ((1/3) * text_embedding_new)   
                # video_emb = ((1/3) * video_embedding_new) + ((1/3) * audio_embedding_new) + ((1/3) * text_embedding_new)    

                # new_embedding[:, 2*D:] = text_emb    #Text = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, D:2*D] = audio_emb  #Audio = (1/3)Video + (1/3)Audio + (1/3)Text
                # new_embedding[:, :D] = video_emb     #Video = (1/3)Video + (1/3)Audio + (1/3)Text
                ### Equal distribution ###
                
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


