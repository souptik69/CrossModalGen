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




class AVT_VQVAE_Encoder_SingleTimestep(nn.Module):
    """
    Modified encoder with individual modality VQ encoding functions.
    """
    def __init__(self, audio_dim, video_dim, text_dim, n_embeddings, embedding_dim):
        super(AVT_VQVAE_Encoder_SingleTimestep, self).__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = embedding_dim
        
        # Use single timestep quantizer
        self.Cross_quantizer = Cross_VQEmbeddingEMA_AVT_SingleTimestep(n_embeddings, self.hidden_dim)
        
        self.Video_encoder = Video_Encoder(video_dim, self.hidden_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, self.hidden_dim)
        self.Text_encoder = Text_Encoder(text_dim, self.hidden_dim)
        
        # Self-attention modules
        self.video_self_att = InternalTemporalRelationModule_New(
            input_dim=video_dim, d_model=self.hidden_dim, 
            num_heads=6, num_layers=6, dropout=0.1
        )
        self.audio_self_att = InternalTemporalRelationModule_New(
            input_dim=audio_dim, d_model=self.hidden_dim,
            num_heads=6, num_layers=6, dropout=0.1
        )
        self.text_self_att = InternalTemporalRelationModule_New(
            input_dim=text_dim, d_model=self.hidden_dim,
            num_heads=6, num_layers=6, dropout=0.1
        )

    def extract_last_valid_timestep(self, sequence, lengths, attention_mask):
        """Extract last valid timestep for each sample."""
        B, T, D = sequence.shape
        last_timesteps = []
        
        for i in range(B):
            valid_length = lengths[i].item()
            last_valid_idx = valid_length - 1
            last_timestep = sequence[i, last_valid_idx, :]
            last_timesteps.append(last_timestep)
        
        return torch.stack(last_timesteps, dim=0)  # [B, D]

    def Audio_VQ_Encoder(self, audio_feat, attention_mask=None, lengths=None):
        """
        Encode audio and quantize last timestep only.
        
        Args:
            audio_feat: [B, T, audio_dim] - Audio features
            attention_mask: [B, T] - Padding mask
            lengths: [B] - Actual sequence lengths
        
        Returns:
            out_vq: [B, 3D] - Global quantized vector
            audio_vq: [B, D] - Quantized audio segment
        """
        audio_feat = audio_feat.cuda()
        
        # Self-attention
        audio_semantic_result = self.audio_self_att(audio_feat, attention_mask)
        
        # Extract last valid timestep
        audio_last = self.extract_last_valid_timestep(audio_semantic_result, lengths, attention_mask)
        
        # Quantize
        out_vq, audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_last)
        
        return out_vq, audio_vq

    def Video_VQ_Encoder(self, video_feat, attention_mask=None, lengths=None):
        """
        Encode video and quantize last timestep only.
        
        Args:
            video_feat: [B, T, video_dim] - Video features
            attention_mask: [B, T] - Padding mask
            lengths: [B] - Actual sequence lengths
        
        Returns:
            out_vq: [B, 3D] - Global quantized vector
            video_vq: [B, D] - Quantized video segment
        """
        video_feat = video_feat.cuda()
        
        # Self-attention
        video_semantic_result = self.video_self_att(video_feat, attention_mask)
        
        # Extract last valid timestep
        video_last = self.extract_last_valid_timestep(video_semantic_result, lengths, attention_mask)
        
        # Quantize
        out_vq, video_vq = self.Cross_quantizer.Video_vq_embedding(video_last)
        
        return out_vq, video_vq

    def Text_VQ_Encoder(self, text_feat, attention_mask=None, lengths=None):
        """
        Encode text and quantize last timestep only.
        
        Args:
            text_feat: [B, T, text_dim] - Text features
            attention_mask: [B, T] - Padding mask
            lengths: [B] - Actual sequence lengths
        
        Returns:
            out_vq: [B, 3D] - Global quantized vector
            text_vq: [B, D] - Quantized text segment
        """
        text_feat = text_feat.cuda()
        
        # Self-attention
        text_semantic_result = self.text_self_att(text_feat, attention_mask)
        
        # Extract last valid timestep
        text_last = self.extract_last_valid_timestep(text_semantic_result, lengths, attention_mask)
        
        # Quantize
        out_vq, text_vq = self.Cross_quantizer.Text_vq_embedding(text_last)
        
        return out_vq, text_vq

    def forward(self, audio_feat, video_feat, text_feat, epoch, attention_mask=None, lengths=None):
        """
        Full forward pass with all three modalities.
        """
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()

        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        if lengths is not None:
            lengths = lengths.cuda()
        
        # Self-attention to get semantic representations [B, T, D]
        video_semantic_result = self.video_self_att(video_feat, attention_mask)
        audio_semantic_result = self.audio_self_att(audio_feat, attention_mask)
        text_semantic_result = self.text_self_att(text_feat, attention_mask)

        # Simple encoder (linear + ReLU) [B, T, D]
        video_encoder_result = self.Video_encoder(video_feat, attention_mask)
        audio_encoder_result = self.Audio_encoder(audio_feat, attention_mask)
        text_encoder_result = self.Text_encoder(text_feat, attention_mask)

        # Extract LAST valid timestep from semantic results [B, D]
        video_last = self.extract_last_valid_timestep(video_semantic_result, lengths, attention_mask)
        audio_last = self.extract_last_valid_timestep(audio_semantic_result, lengths, attention_mask)
        text_last = self.extract_last_valid_timestep(text_semantic_result, lengths, attention_mask)

        # Quantize ONLY the last timesteps
        out_vq_video, video_vq, \
        out_vq_audio, audio_vq, \
        out_vq_text, text_vq, \
        video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
        video_perplexity, audio_perplexity, text_perplexity, \
        equal_num, cmcm_loss, segment_loss = self.Cross_quantizer(
            audio_last, video_last, text_last, epoch
        )

        return audio_semantic_result, audio_encoder_result, \
               video_semantic_result, video_encoder_result, \
               text_semantic_result, text_encoder_result, \
               out_vq_video, video_vq, \
               out_vq_audio, audio_vq, \
               out_vq_text, text_vq, \
               video_embedding_loss, audio_embedding_loss, text_embedding_loss, \
               video_perplexity, audio_perplexity, text_perplexity, \
               equal_num, cmcm_loss, segment_loss


class Sentiment_Decoder_Global(nn.Module):

    def __init__(self, input_dim, dropout=0.1):
        super(Sentiment_Decoder_Global, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        
        self.proj1 = nn.Linear(input_dim, input_dim)
        self.proj2 = nn.Linear(input_dim, input_dim)
        
        self.out_layer = nn.Linear(input_dim, 1)  # Regression to single sentiment value
        
    def forward(self, vq_global):

        proj = self.proj1(vq_global)                                           # [B, input_dim]
        proj = F.relu(proj)                                                    # [B, input_dim]
        proj = F.dropout(proj, p=self.dropout, training=self.training)        # [B, input_dim]
        proj = self.proj2(proj)                                                # [B, input_dim]
        
        last_hs_proj = proj + vq_global                                        # [B, input_dim]
        
        sentiment_score = self.out_layer(last_hs_proj)                         # [B, 1]
        
        return sentiment_score


class Sentiment_Decoder_Combined_Global(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(Sentiment_Decoder_Combined_Global, self).__init__()
        self.input_dim = input_dim
        self.combined_dim = input_dim * 3
        self.dropout = dropout
        
        # Separate residual blocks for each modality
        self.video_proj1 = nn.Linear(input_dim, input_dim)
        self.video_proj2 = nn.Linear(input_dim, input_dim)
        
        self.audio_proj1 = nn.Linear(input_dim, input_dim)
        self.audio_proj2 = nn.Linear(input_dim, input_dim)
        
        self.text_proj1 = nn.Linear(input_dim, input_dim)
        self.text_proj2 = nn.Linear(input_dim, input_dim)
        
        # Final output layer on concatenated features
        self.out_layer = nn.Linear(self.combined_dim, 1)
        
    def forward(self, video_vq_global, audio_vq_global, text_vq_global):
        # Video residual block
        video_proj = self.video_proj1(video_vq_global)
        video_proj = F.relu(video_proj)
        video_proj = F.dropout(video_proj, p=self.dropout, training=self.training)
        video_proj = self.video_proj2(video_proj)
        video_out = video_proj + video_vq_global
        
        # Audio residual block
        audio_proj = self.audio_proj1(audio_vq_global)
        audio_proj = F.relu(audio_proj)
        audio_proj = F.dropout(audio_proj, p=self.dropout, training=self.training)
        audio_proj = self.audio_proj2(audio_proj)
        audio_out = audio_proj + audio_vq_global
        
        # Text residual block
        text_proj = self.text_proj1(text_vq_global)
        text_proj = F.relu(text_proj)
        text_proj = F.dropout(text_proj, p=self.dropout, training=self.training)
        text_proj = self.text_proj2(text_proj)
        text_out = text_proj + text_vq_global
        
        # Concatenate and output
        combined = torch.cat([video_out, audio_out, text_out], dim=1)
        sentiment_score = self.out_layer(combined)
        
        return sentiment_score
    

class Audio_Decoder_Broadcast(nn.Module):
    """
    Modified audio decoder that broadcasts global VQ vector across all timesteps.
    """
    def __init__(self, output_dim, vq_dim):
        super(Audio_Decoder_Broadcast, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.audio_rec = nn.Linear(vq_dim * 2, output_dim)
        self.audio_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.audio_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, audio_encoder_result, audio_vq_global, attention_mask=None):
        """
        Args:
            audio_encoder_result: [B, T, vq_dim]
            audio_vq_global: [B, vq_dim*3]
            attention_mask: [B, T]
        Returns:
            audio_decoder_result: [B, T, output_dim]
        """
        B, T, encoder_dim = audio_encoder_result.shape
        
        # Process global VQ: [B, 3D] → [B, D]
        audio_vq_processed = self.audio_linear_1(audio_vq_global)
        audio_vq_processed = self.audio_linear(audio_vq_processed)
        
        # Broadcast: [B, D] → [B, T, D]
        audio_vq_broadcasted = audio_vq_processed.unsqueeze(1).expand(B, T, self.vq_dim)
        
        # Concatenate: [B, T, D] + [B, T, D] → [B, T, 2D]
        combined = torch.cat([audio_vq_broadcasted, audio_encoder_result], dim=2)
        
        # Reconstruct: [B, T, 2D] → [B, T, output_dim]
        audio_decoder_result = self.audio_rec(combined)
        
        # Apply padding mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(audio_decoder_result)
            audio_decoder_result = audio_decoder_result * mask_expanded
        
        return audio_decoder_result




class Video_Decoder_Broadcast(nn.Module):
    """
    Modified video decoder that broadcasts global VQ vector across all timesteps.
    """
    def __init__(self, output_dim, vq_dim):
        super(Video_Decoder_Broadcast, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.video_rec = nn.Linear(vq_dim * 2, output_dim)
        self.video_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.video_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, video_encoder_result, video_vq_global, attention_mask=None):
        """
        Args:
            video_encoder_result: [B, T, vq_dim]
            video_vq_global: [B, vq_dim*3]
            attention_mask: [B, T]
        Returns:
            video_decoder_result: [B, T, output_dim]
        """
        B, T, encoder_dim = video_encoder_result.shape
        
        # Process global VQ
        video_vq_processed = self.video_linear_1(video_vq_global)
        video_vq_processed = self.video_linear(video_vq_processed)
        
        # Broadcast
        video_vq_broadcasted = video_vq_processed.unsqueeze(1).expand(B, T, self.vq_dim)
        
        # Concatenate and reconstruct
        combined = torch.cat([video_vq_broadcasted, video_encoder_result], dim=2)
        video_decoder_result = self.video_rec(combined)
        
        # Apply padding mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(video_decoder_result)
            video_decoder_result = video_decoder_result * mask_expanded
        
        return video_decoder_result




class Text_Decoder_Broadcast(nn.Module):
    """
    Modified text decoder that broadcasts global VQ vector across all timesteps.
    """
    def __init__(self, output_dim, vq_dim):
        super(Text_Decoder_Broadcast, self).__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        self.relu = nn.ReLU()
        self.text_rec = nn.Linear(vq_dim * 2, output_dim)
        self.text_linear_1 = nn.Linear(vq_dim * 3, vq_dim)
        self.text_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, text_encoder_result, text_vq_global, attention_mask=None):
        """
        Args:
            text_encoder_result: [B, T, vq_dim]
            text_vq_global: [B, vq_dim*3]
            attention_mask: [B, T]
        Returns:
            text_decoder_result: [B, T, output_dim]
        """
        B, T, encoder_dim = text_encoder_result.shape
        
        # Process global VQ
        text_vq_processed = self.text_linear_1(text_vq_global)
        text_vq_processed = self.text_linear(text_vq_processed)
        
        # Broadcast
        text_vq_broadcasted = text_vq_processed.unsqueeze(1).expand(B, T, self.vq_dim)
        
        # Concatenate and reconstruct
        combined = torch.cat([text_vq_broadcasted, text_encoder_result], dim=2)
        text_decoder_result = self.text_rec(combined)
        
        # Apply padding mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(text_decoder_result)
            text_decoder_result = text_decoder_result * mask_expanded
        
        return text_decoder_result
    



class AVT_VQVAE_Decoder_Broadcast(nn.Module):
    """
    Complete decoder with individual modality reconstruction methods.
    """
    def __init__(self, audio_dim, video_dim, text_dim, embedding_dim):
        super(AVT_VQVAE_Decoder_Broadcast, self).__init__()
        self.hidden_dim = embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        
        # Individual decoders
        self.Video_decoder = Video_Decoder_Broadcast(video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder_Broadcast(audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder_Broadcast(text_dim, self.hidden_dim)

        # Sentiment decoders
        self.video_sentiment_decoder = Sentiment_Decoder_Global(self.hidden_dim * 3)
        self.audio_sentiment_decoder = Sentiment_Decoder_Global(self.hidden_dim * 3)
        self.text_sentiment_decoder = Sentiment_Decoder_Global(self.hidden_dim * 3)
        self.combined_sentiment_decoder = Sentiment_Decoder_Combined_Global(self.hidden_dim * 3)

    
    def forward(self, audio_feat, video_feat, text_feat, 
                audio_encoder_result, video_encoder_result, text_encoder_result, 
                out_vq_audio, audio_vq, out_vq_video, video_vq, out_vq_text, text_vq, 
                attention_mask=None):
  
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_feat = text_feat.cuda()
        
        # Reconstruct all modalities
        video_recon_result = self.Video_decoder(video_encoder_result, out_vq_video, attention_mask)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, out_vq_audio, attention_mask)
        text_recon_result = self.Text_decoder(text_encoder_result, out_vq_text, attention_mask)

        # Compute reconstruction losses
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
        

        video_score = self.video_sentiment_decoder(out_vq_video)
        audio_score = self.audio_sentiment_decoder(out_vq_audio)
        text_score = self.text_sentiment_decoder(out_vq_text)
        combined_score = self.combined_sentiment_decoder(out_vq_video, out_vq_audio, out_vq_text)

        return audio_recon_loss, video_recon_loss, text_recon_loss, \
               audio_score, video_score, text_score, combined_score





class Cross_VQEmbeddingEMA_AVT_SingleTimestep(nn.Module):
    """
    Modified VQ-VAE with individual modality embedding functions.
    Quantizes only single timesteps [B, D] instead of sequences [B, T, D].
    """
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AVT_SingleTimestep, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding_dim = embedding_dim

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        # Replicate for 3 modalities
        embedding = torch.cat([embedding, embedding, embedding], dim=1)  # [M, 3D]
        
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_v", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_a", torch.zeros(n_embeddings))
        self.register_buffer("ema_count_t", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))

    def Video_vq_embedding(self, video_last):
        """
        Quantize the last video timestep.
        
        Args:
            video_last: [B, D] - Last valid timestep of video
        
        Returns:
            out_vq: [B, 3D] - Full quantized vector
            v_quantized: [B, D] - Quantized video segment
        """
        B, D = video_last.size()
        v_flat = video_last.detach()  # [B, D]

        video_embedding = self.embedding[:, :D]  # [M, D]

        # Compute distances: [B, M]
        v_distances = torch.addmm(
            torch.sum(video_embedding ** 2, dim=1) + torch.sum(v_flat ** 2, dim=1, keepdim=True),
            v_flat, video_embedding.t(),
            alpha=-2.0, beta=1.0
        )

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [B]

        v_quantized = F.embedding(v_indices, video_embedding)  # [B, D]
        v_quantized = video_last + (v_quantized - video_last).detach()

        out_vq = F.embedding(v_indices, self.embedding)  # [B, 3D]
        
        return out_vq, v_quantized

    def Audio_vq_embedding(self, audio_last):
        """
        Quantize the last audio timestep.
        
        Args:
            audio_last: [B, D] - Last valid timestep of audio
        
        Returns:
            out_vq: [B, 3D] - Full quantized vector
            a_quantized: [B, D] - Quantized audio segment
        """
        B, D = audio_last.size()
        a_flat = audio_last.detach()  # [B, D]
     
        audio_embedding = self.embedding[:, D:2*D]  # [M, D]

        # Compute distances: [B, M]
        a_distances = torch.addmm(
            torch.sum(audio_embedding ** 2, dim=1) + torch.sum(a_flat ** 2, dim=1, keepdim=True),
            a_flat, audio_embedding.t(),
            alpha=-2.0, beta=1.0
        )

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [B]

        a_quantized = F.embedding(a_indices, audio_embedding)  # [B, D]
        a_quantized = audio_last + (a_quantized - audio_last).detach()

        out_vq = F.embedding(a_indices, self.embedding)  # [B, 3D]
        
        return out_vq, a_quantized

    def Text_vq_embedding(self, text_last):
        """
        Quantize the last text timestep.
        
        Args:
            text_last: [B, D] - Last valid timestep of text
        
        Returns:
            out_vq: [B, 3D] - Full quantized vector
            t_quantized: [B, D] - Quantized text segment
        """
        B, D = text_last.size()
        t_flat = text_last.detach()  # [B, D]

        text_embedding = self.embedding[:, 2*D:]  # [M, D]

        # Compute distances: [B, M]
        t_distances = torch.addmm(
            torch.sum(text_embedding ** 2, dim=1) + torch.sum(t_flat ** 2, dim=1, keepdim=True),
            t_flat, text_embedding.t(),
            alpha=-2.0, beta=1.0
        )

        t_indices = torch.argmin(t_distances.double(), dim=-1)  # [B]

        t_quantized = F.embedding(t_indices, text_embedding)  # [B, D]
        t_quantized = text_last + (t_quantized - text_last).detach()

        out_vq = F.embedding(t_indices, self.embedding)  # [B, 3D]
        
        return out_vq, t_quantized

    def forward(self, audio_last, video_last, text_last, epoch):
        """
        Quantize the last timestep of each modality.
        
        Args:
            audio_last: [B, D] - Last valid timestep of audio
            video_last: [B, D] - Last valid timestep of video  
            text_last: [B, D] - Last valid timestep of text
            epoch: Current training epoch
            
        Returns:
            v_full_vectors: [B, 3D] - Full quantized vector for video
            v_quantized: [B, D] - Quantized video segment
            a_full_vectors: [B, 3D] - Full quantized vector for audio
            a_quantized: [B, D] - Quantized audio segment
            t_full_vectors: [B, 3D] - Full quantized vector for text
            t_quantized: [B, D] - Quantized text segment
            v_loss, a_loss, t_loss: Commitment losses
            v_perplexity, a_perplexity, t_perplexity: Codebook usage metrics
            equal_num: Number of samples with same codebook across modalities
            cmcm_loss: Cross-modal code matching loss
            segment_loss: Segment alignment loss
        """
        M, D_total = self.embedding.size()  # M = num codebooks, D_total = 3D
        B, D = audio_last.size()            # B = batch size, D = embedding_dim
        
        device = audio_last.device
        
        # Detach for distance computation
        a_flat = audio_last.detach()  # [B, D]
        v_flat = video_last.detach()  # [B, D]
        t_flat = text_last.detach()   # [B, D]
        
        # Extract modality-specific segments from codebook
        video_embedding = self.embedding[:, :D]         # [M, D]
        audio_embedding = self.embedding[:, D:2*D]      # [M, D]
        text_embedding = self.embedding[:, 2*D:]        # [M, D]

        # Compute distances: [B, M]
        v_distances = torch.addmm(
            torch.sum(video_embedding ** 2, dim=1) + torch.sum(v_flat ** 2, dim=1, keepdim=True),
            v_flat, video_embedding.t(),
            alpha=-2.0, beta=1.0
        )
        
        a_distances = torch.addmm(
            torch.sum(audio_embedding ** 2, dim=1) + torch.sum(a_flat ** 2, dim=1, keepdim=True),
            a_flat, audio_embedding.t(),
            alpha=-2.0, beta=1.0
        )
        
        t_distances = torch.addmm(
            torch.sum(text_embedding ** 2, dim=1) + torch.sum(t_flat ** 2, dim=1, keepdim=True),
            t_flat, text_embedding.t(),
            alpha=-2.0, beta=1.0
        )

        # Gradient-enabled distances for CMCM (using original tensors, not detached)
        v_distances_gradient = torch.addmm(
            torch.sum(video_embedding ** 2, dim=1) + torch.sum(video_last ** 2, dim=1, keepdim=True),
            video_last, video_embedding.t(),
            alpha=-2.0, beta=1.0
        )
        
        a_distances_gradient = torch.addmm(
            torch.sum(audio_embedding ** 2, dim=1) + torch.sum(audio_last ** 2, dim=1, keepdim=True),
            audio_last, audio_embedding.t(),
            alpha=-2.0, beta=1.0
        )
        
        t_distances_gradient = torch.addmm(
            torch.sum(text_embedding ** 2, dim=1) + torch.sum(text_last ** 2, dim=1, keepdim=True),
            text_last, text_embedding.t(),
            alpha=-2.0, beta=1.0
        )

        # Soft assignments for CMCM: [B, M]
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)
        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)
        t_ph = F.softmax(-torch.sqrt(t_distances_gradient), dim=1)

        # CMCM Loss (already per-sample, no temporal averaging needed)
        Scode_av = a_ph @ torch.log(v_ph.t() + 1e-10) + v_ph @ torch.log(a_ph.t() + 1e-10)
        Scode_at = a_ph @ torch.log(t_ph.t() + 1e-10) + t_ph @ torch.log(a_ph.t() + 1e-10)
        Scode_tv = t_ph @ torch.log(v_ph.t() + 1e-10) + v_ph @ torch.log(t_ph.t() + 1e-10)

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

        # Find nearest codebook indices: [B]
        v_indices = torch.argmin(v_distances.double(), dim=-1)
        a_indices = torch.argmin(a_distances.double(), dim=-1)
        t_indices = torch.argmin(t_distances.double(), dim=-1)

        # One-hot encodings: [B, M]
        v_encodings = F.one_hot(v_indices, M).double()
        a_encodings = F.one_hot(a_indices, M).double()
        t_encodings = F.one_hot(t_indices, M).double()

        # Get quantized segments: [B, D]
        v_quantized_segment = F.embedding(v_indices, video_embedding)
        a_quantized_segment = F.embedding(a_indices, audio_embedding)
        t_quantized_segment = F.embedding(t_indices, text_embedding)

        # Get full vectors: [B, 3D]
        v_full_vectors = F.embedding(v_indices, self.embedding)
        a_full_vectors = F.embedding(a_indices, self.embedding)
        t_full_vectors = F.embedding(t_indices, self.embedding)

        # Check cross-modal agreement
        equal_item = (a_indices == v_indices) & (a_indices == t_indices)
        equal_num = equal_item.sum().item()
        
        # EMA Updates during training
        if self.training:
            # Video segment update
            self.ema_count_v = self.decay * self.ema_count_v + (1 - self.decay) * torch.sum(v_encodings, dim=0)
            n_v = torch.sum(self.ema_count_v)
            self.ema_count_v = (self.ema_count_v + self.epsilon) / (n_v + M * self.epsilon) * n_v

            v_dw = torch.matmul(v_encodings.t(), v_flat)
            a_dw_v = torch.matmul(v_encodings.t(), a_flat)
            t_dw_v = torch.matmul(v_encodings.t(), t_flat)

            v_segment_update = (0.6 * v_dw) + (0.2 * a_dw_v) + (0.2 * t_dw_v)
            with torch.no_grad():
                new_embedding_v = self.embedding.clone()
                self.ema_weight[:, :D] = self.decay * self.ema_weight[:, :D] + (1 - self.decay) * v_segment_update
                new_embedding_v[:, :D] = self.ema_weight[:, :D] / self.ema_count_v.unsqueeze(-1)
                self.embedding = new_embedding_v

            # Audio segment update
            self.ema_count_a = self.decay * self.ema_count_a + (1 - self.decay) * torch.sum(a_encodings, dim=0)
            n_a = torch.sum(self.ema_count_a)
            self.ema_count_a = (self.ema_count_a + self.epsilon) / (n_a + M * self.epsilon) * n_a

            a_dw = torch.matmul(a_encodings.t(), a_flat)
            v_dw_a = torch.matmul(a_encodings.t(), v_flat)
            t_dw_a = torch.matmul(a_encodings.t(), t_flat)

            a_segment_update = (0.2 * v_dw_a) + (0.2 * t_dw_a) + (0.6 * a_dw)
            with torch.no_grad():
                new_embedding_a = self.embedding.clone()
                self.ema_weight[:, D:2*D] = self.decay * self.ema_weight[:, D:2*D] + (1 - self.decay) * a_segment_update
                new_embedding_a[:, D:2*D] = self.ema_weight[:, D:2*D] / self.ema_count_a.unsqueeze(-1)
                self.embedding = new_embedding_a

            # Text segment update
            self.ema_count_t = self.decay * self.ema_count_t + (1 - self.decay) * torch.sum(t_encodings, dim=0)
            n_t = torch.sum(self.ema_count_t)
            self.ema_count_t = (self.ema_count_t + self.epsilon) / (n_t + M * self.epsilon) * n_t

            t_dw = torch.matmul(t_encodings.t(), t_flat)
            v_dw_t = torch.matmul(t_encodings.t(), v_flat)
            a_dw_t = torch.matmul(t_encodings.t(), a_flat)

            t_segment_update = (0.2 * v_dw_t) + (0.2 * a_dw_t) + (0.6 * t_dw)
            with torch.no_grad():
                new_embedding_t = self.embedding.clone()
                self.ema_weight[:, 2*D:] = self.decay * self.ema_weight[:, 2*D:] + (1 - self.decay) * t_segment_update
                new_embedding_t[:, 2*D:] = self.ema_weight[:, 2*D:] / self.ema_count_t.unsqueeze(-1)
                self.embedding = new_embedding_t

            # Hierarchical influence
            with torch.no_grad():
                new_embedding = self.embedding.clone()
                new_embedding[:, :D] =  ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                new_embedding[:, D:2*D] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D] ) + ((1/3) * new_embedding[:, 2*D:])
                new_embedding[:, 2*D:] = ((1/3) * new_embedding[:, :D]) + ((1/3) * new_embedding[:, D:2*D]) + ((1/3) * new_embedding[:, 2*D:])
                self.embedding = new_embedding

            # Segment alignment loss
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
            for indice in a_indices:
                self.unactivated_count[indice.item()] = 0
            for indice in v_indices:
                self.unactivated_count[indice.item()] = 0
            for indice in t_indices:
                self.unactivated_count[indice.item()] = 0
                
            activated_indices = []
            unactivated_indices = []
            for i, x in enumerate(self.unactivated_count):
                if x > 150:
                    unactivated_indices.append(i)
                    self.unactivated_count[i] = 0
                elif x >= 0 and x < 50:
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
            segment_loss = torch.tensor(0.0, device=device)

        # CMCM loss
        cmcm_loss = 0.5 * (Lcmcm_av + Lcmcm_at + Lcmcm_tv)

        # Commitment losses (single timestep MSE)
        v_e_latent_loss = F.mse_loss(video_last, v_quantized_segment.detach())
        va_e_latent_loss = F.mse_loss(video_last, a_quantized_segment.detach())
        vt_e_latent_loss = F.mse_loss(video_last, t_quantized_segment.detach())
        
        a_e_latent_loss = F.mse_loss(audio_last, a_quantized_segment.detach())
        av_e_latent_loss = F.mse_loss(audio_last, v_quantized_segment.detach())
        at_e_latent_loss = F.mse_loss(audio_last, t_quantized_segment.detach())
        
        t_e_latent_loss = F.mse_loss(text_last, t_quantized_segment.detach())
        ta_e_latent_loss = F.mse_loss(text_last, a_quantized_segment.detach())
        tv_e_latent_loss = F.mse_loss(text_last, v_quantized_segment.detach())

        v_loss = (self.commitment_cost * 2.0 * v_e_latent_loss) + (0.5*self.commitment_cost * va_e_latent_loss) + (0.5*self.commitment_cost * vt_e_latent_loss)
        a_loss = (self.commitment_cost * 2.0 * a_e_latent_loss) + (0.5*self.commitment_cost * av_e_latent_loss) + (0.5*self.commitment_cost * at_e_latent_loss)
        t_loss = (self.commitment_cost * 2.0 * t_e_latent_loss) + (0.5*self.commitment_cost * ta_e_latent_loss) + (0.5*self.commitment_cost * tv_e_latent_loss)
        
        # Straight-through estimator
        v_quantized_segment = video_last + (v_quantized_segment - video_last).detach()
        a_quantized_segment = audio_last + (a_quantized_segment - audio_last).detach()
        t_quantized_segment = text_last + (t_quantized_segment - text_last).detach()
        
        v_full_continuous = torch.cat([video_last, audio_last, text_last], dim=-1)
        a_full_continuous = torch.cat([video_last, audio_last, text_last], dim=-1)
        t_full_continuous = torch.cat([video_last, audio_last, text_last], dim=-1)

        v_full_vectors = v_full_continuous + (v_full_vectors - v_full_continuous).detach()
        a_full_vectors = a_full_continuous + (a_full_vectors - a_full_continuous).detach()
        t_full_vectors = t_full_continuous + (t_full_vectors - t_full_continuous).detach()

        # Perplexity
        v_avg_probs = torch.mean(v_encodings, dim=0)
        a_avg_probs = torch.mean(a_encodings, dim=0)
        t_avg_probs = torch.mean(t_encodings, dim=0)

        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        t_perplexity = torch.exp(-torch.sum(t_avg_probs * torch.log(t_avg_probs + 1e-10)))

        return  v_full_vectors, v_quantized_segment,\
                a_full_vectors, a_quantized_segment,\
                t_full_vectors, t_quantized_segment,\
                v_loss, a_loss, t_loss, \
                v_perplexity, a_perplexity, t_perplexity,\
                equal_num, cmcm_loss, segment_loss