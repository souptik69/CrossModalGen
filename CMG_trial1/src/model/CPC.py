import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Cross_CPC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
        # Autoregressive LSTM network for video
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for video
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for audio
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
    
    """
    video_forward_seq took the first t_samples+1 samples [0:t_samples].
    video_encode_samples took the samples [t_samples+1:t_samples+self.n_prediction_steps] as true future results.
    The LSTM utilized video_forward_seq to obtain video_context,
    then used video_predictors to predict video_pred.
    Calculate NCE between pred and encode_samples.
    """
    def forward(self, video_vq, audio_vq):
        batch_dim, time_length, _ = video_vq.shape# [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]
        # closedOpen
        # Choose any number from [3, 8) as a starting point, then predict the next two digits. Therefore, forward_seq has a minimum length of 4 (starting from 0).
        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        # losses = list()
        nce = 0 # average over timestep and batch
        video_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        for i in range(1, self.n_prediction_steps+1):# closedOpen
            video_encode_samples[i-1] = video_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            audio_encode_samples[i-1] = audio_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
        video_forward_seq = video_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        audio_forward_seq = audio_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # Autoregressive LSTM for video
        video_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double())
        video_context, video_hidden = self.video_ar_lstm(video_forward_seq, video_hidden)
        
        # Autoregressive LSTM for audio
        audio_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double())
        audio_context, audio_hidden = self.audio_ar_lstm(audio_forward_seq, audio_hidden)
        
        video_context = video_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        audio_context = audio_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        
        video_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        for i in range(0, self.n_prediction_steps):
            video_linear = self.video_predictors[i]
            video_pred[i] = video_linear(video_context) #e.g. size 80*512 -> 80*256
            audio_linear = self.audio_predictors[i]
            audio_pred[i] = audio_linear(audio_context) #e.g. size 80*512 -> 80*256
        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total3 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total4 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            w1 = 1.0
            w2 = 1.0
            # Slightly computing self nce for each modality can provide a direction to align different modalities.
            w3 = 0.1
            w4 = 0.1
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1))) # nce is a tensor
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2))) # nce is a tensor
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3))) # nce is a tensor
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4))) # nce is a tensor
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        accuracy1 = 1.*correct1/batch_dim
        accuracy2 = 1.*correct2/batch_dim
        accuracy3 = 1.*correct3/batch_dim
        accuracy4 = 1.*correct4/batch_dim
        return accuracy1, accuracy2, accuracy3, accuracy4, nce
    

class Cross_CPC_AVT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC_AVT, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
        # Autoregressive LSTM network for video
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for video
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for audio
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for text
        self.text_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
    
    """
    video_forward_seq took the first t_samples+1 samples [0:t_samples].
    video_encode_samples took the samples [t_samples+1:t_samples+self.n_prediction_steps] as true future results.
    The LSTM utilized video_forward_seq to obtain video_context,
    then used video_predictors to predict video_pred.
    Calculate NCE between pred and encode_samples.
    """
    def forward(self, audio_vq, video_vq, text_vq):
        batch_dim, time_length, _ = video_vq.shape# [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]

        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        # losses = list()
        nce = 0 # average over timestep and batch
        video_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        text_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = text_vq.device).double() # e.g. size 5*80*256
        for i in range(1, self.n_prediction_steps+1):
            video_encode_samples[i-1] = video_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            audio_encode_samples[i-1] = audio_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            text_encode_samples[i-1] = text_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
        video_forward_seq = video_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        audio_forward_seq = audio_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        text_forward_seq = text_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # Autoregressive LSTM for video
        video_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double())
        video_context, video_hidden = self.video_ar_lstm(video_forward_seq, video_hidden)
        
        # Autoregressive LSTM for audio
        audio_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double())
        audio_context, audio_hidden = self.audio_ar_lstm(audio_forward_seq, audio_hidden)
        
        # Autoregressive LSTM for text
        text_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).double())
        text_context, text_hidden = self.text_ar_lstm(text_forward_seq, text_hidden)
        
        video_context = video_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        audio_context = audio_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        text_context = text_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        
        video_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        text_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        
        for i in range(0, self.n_prediction_steps):
            video_linear = self.video_predictors[i]
            video_pred[i] = video_linear(video_context) #e.g. size 80*512 -> 80*256
            audio_linear = self.audio_predictors[i]
            audio_pred[i] = audio_linear(audio_context) #e.g. size 80*512 -> 80*256
            text_linear = self.text_predictors[i]
            text_pred[i] = text_linear(text_context) #e.g. size 80*512 -> 80*256
        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total3 = torch.mm(video_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total4 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total5 = torch.mm(text_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total6 = torch.mm(text_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total7 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total8 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total9 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=0), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            w1 = 1.0
            w2 = 1.0
            w3 = 1.0
            w4 = 1.0
            w5 = 1.0
            w6 = 1.0
            # Slightly computing self nce for each modality can provide a direction to align different modalities.
            w7 = 0.1
            w8 = 0.1
            w9 = 0.1
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1))) # nce is a tensor
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2))) # nce is a tensor
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3))) # nce is a tensor
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4))) # nce is a tensor
            nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5))) # nce is a tensor
            nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6))) # nce is a tensor
            nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7))) # nce is a tensor
            nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8))) # nce is a tensor
            nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9))) # nce is a tensor
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        accuracy1 = 1.*correct1/batch_dim
        accuracy2 = 1.*correct2/batch_dim
        accuracy3 = 1.*correct3/batch_dim
        accuracy4 = 1.*correct4/batch_dim
        accuracy5 = 1.*correct5/batch_dim
        accuracy6 = 1.*correct6/batch_dim
        accuracy7 = 1.*correct7/batch_dim
        accuracy8 = 1.*correct8/batch_dim
        accuracy9 = 1.*correct9/batch_dim
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce




class Cross_CPC_AVT_pad(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC_AVT_pad, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
        # Autoregressive LSTM networks for each modality
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor networks for each modality
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        self.text_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
    
    def forward(self, audio_vq, video_vq, text_vq, lengths=None, attention_mask=None):
        """
        Adaptive CPC that selects global context length but adapts per-sample based on actual content.
        
        Key Innovation: Uses full sequence capacity for context selection, but intelligently
        adapts for individual samples based on their actual content length.
        """
        batch_dim, time_length, _ = video_vq.shape
        
        if lengths is None or attention_mask is None:
            raise ValueError("This adaptive CPC requires both lengths and attention_mask")
        
        lengths = lengths.to(video_vq.device)
        attention_mask = attention_mask.to(video_vq.device)
        
        # Validate consistency between lengths and attention_mask
        for i in range(batch_dim):
            actual_length = lengths[i].item()
            mask_length = attention_mask[i].sum().item()
            assert actual_length == mask_length, f"Length mismatch for sample {i}: lengths={actual_length}, attention_mask={mask_length}"
        
        # STEP 1: Global Context Length Selection Based on Full Sequence Capacity
        global_max_context = time_length - self.n_prediction_steps - self.min_start_steps
        global_t_samples = torch.randint(global_max_context, size=(1,)).item() + self.min_start_steps
        
        # STEP 2: Per-Sample Context Length Adaptation
        per_sample_context_lengths = []
        per_sample_t_samples = []
        
        for i in range(batch_dim):
            sample_length = lengths[i].item()
            sample_max_context = max(0, sample_length - self.n_prediction_steps - self.min_start_steps)
            adapted_t_samples = min(global_t_samples, sample_max_context + self.min_start_steps - 1)
            adapted_context_length = adapted_t_samples + 1
            
            per_sample_context_lengths.append(adapted_context_length)
            per_sample_t_samples.append(adapted_t_samples)
        
        # STEP 3: Extract Context Sequences with Individual Adaptation
        max_context_length = max(per_sample_context_lengths)
        
        video_forward_seq = video_vq[:, :max_context_length, :]
        audio_forward_seq = audio_vq[:, :max_context_length, :]
        text_forward_seq = text_vq[:, :max_context_length, :]
        
        # STEP 4: Efficient LSTM Processing with Pack/Unpack
        context_lengths_tensor = torch.tensor(per_sample_context_lengths, device=video_vq.device)
        sorted_lengths, sorted_idx = context_lengths_tensor.sort(0, descending=True)
        unsorted_idx = sorted_idx.argsort(0)
        
        # Sort sequences for efficient packing
        video_forward_sorted = video_forward_seq[sorted_idx]
        audio_forward_sorted = audio_forward_seq[sorted_idx]
        text_forward_sorted = text_forward_seq[sorted_idx]
        
        # Pack sequences for LSTM processing
        video_packed = pack_padded_sequence(
            video_forward_sorted, sorted_lengths.cpu(), 
            batch_first=True, enforce_sorted=True
        )
        audio_packed = pack_padded_sequence(
            audio_forward_sorted, sorted_lengths.cpu(),
            batch_first=True, enforce_sorted=True
        )
        text_packed = pack_padded_sequence(
            text_forward_sorted, sorted_lengths.cpu(),
            batch_first=True, enforce_sorted=True
        )
        
        # Initialize hidden states
        video_hidden = (
            torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).double(),
            torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).double()
        )
        audio_hidden = (
            torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).double(),
            torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).double()
        )
        text_hidden = (
            torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=text_vq.device).double(),
            torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=text_vq.device).double()
        )
        
        # Process through LSTMs
        video_context_packed, _ = self.video_ar_lstm(video_packed, video_hidden)
        audio_context_packed, _ = self.audio_ar_lstm(audio_packed, audio_hidden)
        text_context_packed, _ = self.text_ar_lstm(text_packed, text_hidden)
        
        # Unpack sequences and restore original order
        video_context, _ = pad_packed_sequence(video_context_packed, batch_first=True)
        audio_context, _ = pad_packed_sequence(audio_context_packed, batch_first=True)
        text_context, _ = pad_packed_sequence(text_context_packed, batch_first=True)
        
        video_context = video_context[unsorted_idx]
        audio_context = audio_context[unsorted_idx]
        text_context = text_context[unsorted_idx]
        
        # STEP 5: Extract Context Representations at Appropriate Positions
        video_context_list = []
        audio_context_list = []
        text_context_list = []
        
        for i in range(batch_dim):
            context_pos = per_sample_t_samples[i]
            context_pos = min(context_pos, per_sample_context_lengths[i] - 1)
            
            video_context_list.append(video_context[i, context_pos, :])
            audio_context_list.append(audio_context[i, context_pos, :])
            text_context_list.append(text_context[i, context_pos, :])
        
        video_context = torch.stack(video_context_list)
        audio_context = torch.stack(audio_context_list)
        text_context = torch.stack(text_context_list)
        
        # STEP 6: Intelligent Target Extraction Respecting Individual Sequence Lengths
        video_encode_samples = []
        audio_encode_samples = []
        text_encode_samples = []
        
        for step in range(self.n_prediction_steps):
            step_video = []
            step_audio = []
            step_text = []
            
            for i in range(batch_dim):
                target_pos = per_sample_t_samples[i] + step + 1
                
                if target_pos < lengths[i]:
                    step_video.append(video_vq[i, target_pos, :])
                    step_audio.append(audio_vq[i, target_pos, :])
                    step_text.append(text_vq[i, target_pos, :])
                else:
                    last_valid_pos = lengths[i] - 1
                    step_video.append(video_vq[i, last_valid_pos, :])
                    step_audio.append(audio_vq[i, last_valid_pos, :])
                    step_text.append(text_vq[i, last_valid_pos, :])
            
            video_encode_samples.append(torch.stack(step_video))
            audio_encode_samples.append(torch.stack(step_audio))
            text_encode_samples.append(torch.stack(step_text))
        
        video_encode_samples = torch.stack(video_encode_samples)
        audio_encode_samples = torch.stack(audio_encode_samples)
        text_encode_samples = torch.stack(text_encode_samples)
        
        # STEP 7: Generate Predictions Using Adapted Context Representations
        video_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=video_vq.device).double()
        audio_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=audio_vq.device).double()
        text_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=text_vq.device).double()
        
        for i in range(self.n_prediction_steps):
            video_pred[i] = self.video_predictors[i](video_context)
            audio_pred[i] = self.audio_predictors[i](audio_context)
            text_pred[i] = self.text_predictors[i](text_context)
        
        # STEP 8: Calculate NCE Loss and Accuracies
        nce = 0
        accuracy1 = accuracy2 = accuracy3 = accuracy4 = accuracy5 = 0
        accuracy6 = accuracy7 = accuracy8 = accuracy9 = 0
        
        for i in range(self.n_prediction_steps):
            # Compute similarity matrices
            total1 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(text_pred[i], 0, 1))
            total3 = torch.mm(video_encode_samples[i], torch.transpose(text_pred[i], 0, 1))
            total4 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
            total5 = torch.mm(text_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
            total6 = torch.mm(text_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
            total7 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
            total8 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
            total9 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i], 0, 1))
            
            # Calculate correct predictions
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            
            # Weights for loss components
            w1 = w2 = w3 = w4 = w5 = w6 = 1.0
            w7 = w8 = w9 = 0.1
            
            # Accumulate NCE loss
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1)))
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2)))
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3)))
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4)))
            nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5)))
            nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6)))
            nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7)))
            nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8)))
            nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9)))
        
        # Normalize NCE loss
        nce /= -1. * batch_dim * self.n_prediction_steps
        
        # Calculate accuracies
        accuracy1 = 1. * correct1 / batch_dim
        accuracy2 = 1. * correct2 / batch_dim
        accuracy3 = 1. * correct3 / batch_dim
        accuracy4 = 1. * correct4 / batch_dim
        accuracy5 = 1. * correct5 / batch_dim
        accuracy6 = 1. * correct6 / batch_dim
        accuracy7 = 1. * correct7 / batch_dim
        accuracy8 = 1. * correct8 / batch_dim
        accuracy9 = 1. * correct9 / batch_dim
        
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce

