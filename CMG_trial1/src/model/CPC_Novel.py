import torch
import torch.nn as nn

class Cross_CPC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, max_prediction_steps=5, min_start_steps=1):
        super(Cross_CPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.max_prediction_steps = max_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
        # Autoregressive LSTM network for video
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for video 
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(max_prediction_steps)
        ])
        
        # Predictor network for audio
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(max_prediction_steps)
        ])
    
    def forward(self, video_vq, audio_vq):
        batch_dim, time_length, _ = video_vq.shape  # [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]
        
        # Choose any number from [3, 8) as a starting point, then predict the next two digits. Therefore, forward_seq has a minimum length of 4 (starting from 0).
        t_samples = (torch.randint(time_length - self.max_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps

        available_steps = time_length - t_samples - 1
        max_steps = min(available_steps, self.max_prediction_steps)
        n_prediction_steps_actual = torch.randint(1, max_steps + 1, size=(1,)).item()
        
        nce = 0  # average over timestep and batch
        video_encode_samples = torch.empty((n_prediction_steps_actual, batch_dim, self.embedding_dim), 
                                           device=video_vq.device).double()
        audio_encode_samples = torch.empty((n_prediction_steps_actual, batch_dim, self.embedding_dim), 
                                           device=audio_vq.device).double()
        for i in range(1, n_prediction_steps_actual + 1):  # closedOpen
            video_encode_samples[i-1] = video_vq[:, t_samples+i, :].reshape(batch_dim, self.embedding_dim)
            audio_encode_samples[i-1] = audio_vq[:, t_samples+i, :].reshape(batch_dim, self.embedding_dim)
        video_forward_seq = video_vq[:, :t_samples+1, :]
        audio_forward_seq = audio_vq[:, :t_samples+1, :]
        # Autoregressive LSTM for video
        video_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).float(),
                       torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).float())
        video_context, video_hidden = self.video_ar_lstm(video_forward_seq, video_hidden)
        
        # Autoregressive LSTM for audio
        audio_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).float(),
                       torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).float())
        audio_context, audio_hidden = self.audio_ar_lstm(audio_forward_seq, audio_hidden)

        video_context = video_context[:, t_samples, :].reshape(batch_dim, self.context_dim)
        audio_context = audio_context[:, t_samples, :].reshape(batch_dim, self.context_dim)
        
        video_pred = torch.empty((n_prediction_steps_actual, batch_dim, self.embedding_dim), 
                                device=video_vq.device).double()
        audio_pred = torch.empty((n_prediction_steps_actual, batch_dim, self.embedding_dim), 
                                device=audio_vq.device).double()
        
        for i in range(0, n_prediction_steps_actual):
            video_linear = self.video_predictors[i]  
            video_pred[i] = video_linear(video_context)
            audio_linear = self.audio_predictors[i]
            audio_pred[i] = audio_linear(audio_context)

        for i in range(0, n_prediction_steps_actual):
            total1 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
            total3 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
            total4 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), 
                                         torch.arange(0, batch_dim, device=video_vq.device)))
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), 
                                         torch.arange(0, batch_dim, device=video_vq.device)))
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), 
                                         torch.arange(0, batch_dim, device=video_vq.device)))
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), 
                                         torch.arange(0, batch_dim, device=video_vq.device)))
            w1 = 1.0  # Cross-modal: video->audio
            w2 = 1.0  # Cross-modal: audio->video
            w3 = 0.1  # Self-modal: video->video
            w4 = 0.1  # Self-modal: audio->audio
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1)))
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2)))
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3)))
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4)))
        nce /= -1. * batch_dim * n_prediction_steps_actual
        accuracy1 = 1. * correct1 / batch_dim
        accuracy2 = 1. * correct2 / batch_dim
        accuracy3 = 1. * correct3 / batch_dim
        accuracy4 = 1. * correct4 / batch_dim
        
        return accuracy1, accuracy2, accuracy3, accuracy4, nce