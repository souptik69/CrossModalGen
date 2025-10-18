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
        Maximum Context CPC: Uses all available timesteps as context to predict the last timestep.
        
        Key Innovation: For each sequence, uses maximum possible context (length - 1) to predict
        the final timestep, maximizing cross-modal alignment at the sequence endpoint.
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
        
        # STEP 1: Per-Sample Maximum Context Length (all timesteps except last)
        per_sample_context_lengths = []
        per_sample_t_samples = []
        
        for i in range(batch_dim):
            sample_length = lengths[i].item()
            # Use maximum context: all timesteps except the last one
            context_length = max(1, sample_length - 1)  # At least 1 for LSTM
            t_sample = context_length - 1  # Last context position (0-indexed)
            
            per_sample_context_lengths.append(context_length)
            per_sample_t_samples.append(t_sample)
        
        # STEP 2: Extract Context Sequences with Maximum Length
        max_context_length = max(per_sample_context_lengths)
        
        video_forward_seq = video_vq[:, :max_context_length, :]
        audio_forward_seq = audio_vq[:, :max_context_length, :]
        text_forward_seq = text_vq[:, :max_context_length, :]
        
        # STEP 3: Efficient LSTM Processing with Pack/Unpack
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
        
        # STEP 4: Extract Context Representations at Maximum Position
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
        
        # STEP 5: Extract Last Timestep as Target (Final Representation)
        video_encode_samples = []
        audio_encode_samples = []
        text_encode_samples = []
        
        for step in range(self.n_prediction_steps):
            step_video = []
            step_audio = []
            step_text = []
            
            for i in range(batch_dim):
                # Target is the last valid timestep
                last_pos = lengths[i] - 1
                
                step_video.append(video_vq[i, last_pos, :])
                step_audio.append(audio_vq[i, last_pos, :])
                step_text.append(text_vq[i, last_pos, :])
            
            video_encode_samples.append(torch.stack(step_video))
            audio_encode_samples.append(torch.stack(step_audio))
            text_encode_samples.append(torch.stack(step_text))
        
        video_encode_samples = torch.stack(video_encode_samples)
        audio_encode_samples = torch.stack(audio_encode_samples)
        text_encode_samples = torch.stack(text_encode_samples)
        
        # STEP 6: Generate Predictions Using Maximum Context Representations
        video_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=video_vq.device).double()
        audio_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=audio_vq.device).double()
        text_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=text_vq.device).double()
        
        for i in range(self.n_prediction_steps):
            video_pred[i] = self.video_predictors[i](video_context)
            audio_pred[i] = self.audio_predictors[i](audio_context)
            text_pred[i] = self.text_predictors[i](text_context)
        
        # STEP 7: Calculate NCE Loss and Accuracies
        nce = 0
        accuracy1 = accuracy2 = accuracy3 = accuracy4 = accuracy5 = accuracy6 = 0
        accuracy7 = accuracy8 = accuracy9 = 0
        
        for i in range(self.n_prediction_steps):
            # Compute similarity matrices for cross-modal alignment
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




class Cross_CPC_AVT_pad_window(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC_AVT_pad_window, self).__init__()
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
        
        batch_dim, time_length, _ = video_vq.shape
        
        if lengths is None or attention_mask is None:
            raise ValueError("This adaptive CPC requires both lengths and attention_mask")
        
        lengths = lengths.to(video_vq.device)
        attention_mask = attention_mask.to(video_vq.device)
        
        # Initialize accumulators as tensors
        total_nce = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc1 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc2 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc3 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc4 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc5 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc6 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc7 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc8 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        total_acc9 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
        valid_windows = 0
        
        for t_samples in range(self.min_start_steps, time_length - self.n_prediction_steps):
            # Only process samples that have enough content
            valid_mask = lengths > (t_samples + self.n_prediction_steps)
            if valid_mask.sum() == 0:
                continue
            
            valid_batch_size = valid_mask.sum().item()
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            # Extract context sequences up to t_samples
            video_forward_seq = video_vq[valid_mask, :t_samples+1, :]
            audio_forward_seq = audio_vq[valid_mask, :t_samples+1, :]
            text_forward_seq = text_vq[valid_mask, :t_samples+1, :]
            
            # Initialize hidden states
            video_hidden = (
                torch.zeros(self.num_layers, valid_batch_size, self.hidden_dim, device=video_vq.device).double(),
                torch.zeros(self.num_layers, valid_batch_size, self.hidden_dim, device=video_vq.device).double()
            )
            audio_hidden = (
                torch.zeros(self.num_layers, valid_batch_size, self.hidden_dim, device=audio_vq.device).double(),
                torch.zeros(self.num_layers, valid_batch_size, self.hidden_dim, device=audio_vq.device).double()
            )
            text_hidden = (
                torch.zeros(self.num_layers, valid_batch_size, self.hidden_dim, device=text_vq.device).double(),
                torch.zeros(self.num_layers, valid_batch_size, self.hidden_dim, device=text_vq.device).double()
            )
            
            # Process through LSTMs
            video_context, _ = self.video_ar_lstm(video_forward_seq, video_hidden)
            audio_context, _ = self.audio_ar_lstm(audio_forward_seq, audio_hidden)
            text_context, _ = self.text_ar_lstm(text_forward_seq, text_hidden)
            
            # Extract context at timestep t_samples
            video_context = video_context[:, t_samples, :].reshape(valid_batch_size, self.context_dim)
            audio_context = audio_context[:, t_samples, :].reshape(valid_batch_size, self.context_dim)
            text_context = text_context[:, t_samples, :].reshape(valid_batch_size, self.context_dim)
            
            # Extract target sequences for prediction
            video_encode_samples = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=video_vq.device).double()
            audio_encode_samples = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=audio_vq.device).double()
            text_encode_samples = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=text_vq.device).double()
            
            for i in range(1, self.n_prediction_steps+1):
                video_encode_samples[i-1] = video_vq[valid_mask, t_samples+i, :].reshape(valid_batch_size, self.embedding_dim)
                audio_encode_samples[i-1] = audio_vq[valid_mask, t_samples+i, :].reshape(valid_batch_size, self.embedding_dim)
                text_encode_samples[i-1] = text_vq[valid_mask, t_samples+i, :].reshape(valid_batch_size, self.embedding_dim)
            
            # Generate predictions
            video_pred = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=video_vq.device).double()
            audio_pred = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=audio_vq.device).double()
            text_pred = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=text_vq.device).double()
            
            for i in range(self.n_prediction_steps):
                video_pred[i] = self.video_predictors[i](video_context)
                audio_pred[i] = self.audio_predictors[i](audio_context)
                text_pred[i] = self.text_predictors[i](text_context)
            
            # Initialize window accumulators as tensors
            window_nce = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc1 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc2 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc3 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc4 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc5 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc6 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc7 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc8 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            window_acc9 = torch.tensor(0.0, device=video_vq.device, dtype=torch.float64)
            
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
                correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=0), torch.arange(0, valid_batch_size, device=video_vq.device)))
                
                # Weights for loss components
                w1 = w2 = w3 = w4 = w5 = w6 = 1.0
                w7 = w8 = w9 = 0.1
                
                # Accumulate NCE loss
                window_nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1)))
                window_nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2)))
                window_nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3)))
                window_nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4)))
                window_nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5)))
                window_nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6)))
                window_nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7)))
                window_nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8)))
                window_nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9)))
                
                # Accumulate accuracies (keep as tensors)
                window_acc1 += correct1.float() / valid_batch_size
                window_acc2 += correct2.float() / valid_batch_size
                window_acc3 += correct3.float() / valid_batch_size
                window_acc4 += correct4.float() / valid_batch_size
                window_acc5 += correct5.float() / valid_batch_size
                window_acc6 += correct6.float() / valid_batch_size
                window_acc7 += correct7.float() / valid_batch_size
                window_acc8 += correct8.float() / valid_batch_size
                window_acc9 += correct9.float() / valid_batch_size
            
            # Normalize window metrics
            window_nce /= (-1. * valid_batch_size * self.n_prediction_steps)
            
            # Accumulate to totals
            total_nce += window_nce
            total_acc1 += window_acc1 / self.n_prediction_steps
            total_acc2 += window_acc2 / self.n_prediction_steps
            total_acc3 += window_acc3 / self.n_prediction_steps
            total_acc4 += window_acc4 / self.n_prediction_steps
            total_acc5 += window_acc5 / self.n_prediction_steps
            total_acc6 += window_acc6 / self.n_prediction_steps
            total_acc7 += window_acc7 / self.n_prediction_steps
            total_acc8 += window_acc8 / self.n_prediction_steps
            total_acc9 += window_acc9 / self.n_prediction_steps
            valid_windows += 1
        
        # Calculate final averages (all as tensors)
        if valid_windows > 0:
            nce = total_nce / valid_windows
            accuracy1 = total_acc1 / valid_windows
            accuracy2 = total_acc2 / valid_windows
            accuracy3 = total_acc3 / valid_windows
            accuracy4 = total_acc4 / valid_windows
            accuracy5 = total_acc5 / valid_windows
            accuracy6 = total_acc6 / valid_windows
            accuracy7 = total_acc7 / valid_windows
            accuracy8 = total_acc8 / valid_windows
            accuracy9 = total_acc9 / valid_windows
        else:
            # Return zero tensors if no valid windows
            nce = torch.tensor(0.0, device=video_vq.device)
            accuracy1 = torch.tensor(0.0, device=video_vq.device)
            accuracy2 = torch.tensor(0.0, device=video_vq.device)
            accuracy3 = torch.tensor(0.0, device=video_vq.device)
            accuracy4 = torch.tensor(0.0, device=video_vq.device)
            accuracy5 = torch.tensor(0.0, device=video_vq.device)
            accuracy6 = torch.tensor(0.0, device=video_vq.device)
            accuracy7 = torch.tensor(0.0, device=video_vq.device)
            accuracy8 = torch.tensor(0.0, device=video_vq.device)
            accuracy9 = torch.tensor(0.0, device=video_vq.device)
        
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce