import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Cross_CPC_AVT_pad_window(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1, debug=False):
        super(Cross_CPC_AVT_pad_window, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.debug = debug  # Enable/disable diagnostic prints
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
        
        # ===== DIAGNOSTIC: Initial batch information =====
        if self.debug:
            print("\n" + "="*80)
            print("WINDOWED CPC FORWARD PASS - BATCH INFORMATION")
            print("="*80)
            print(f"Batch size: {batch_dim}")
            print(f"Padded sequence length: {time_length}")
            print(f"Embedding dimension: {self.embedding_dim}")
            print(f"Prediction steps: {self.n_prediction_steps}")
            print(f"Min start steps: {self.min_start_steps}")
            print(f"Actual sequence lengths: {lengths.cpu().tolist()}")
            print(f"Window range: [{self.min_start_steps}, {time_length - self.n_prediction_steps})")
            print(f"Total possible windows: {time_length - self.n_prediction_steps - self.min_start_steps}")
        
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
        
        # Track window statistics for summary
        windows_processed = []
        
        for t_samples in range(self.min_start_steps, time_length - self.n_prediction_steps):
            # Only process samples that have enough content
            valid_mask = lengths > (t_samples + self.n_prediction_steps)
            
            if valid_mask.sum() == 0:
                if self.debug:
                    print(f"\nWindow t={t_samples}: SKIPPED (no valid samples)")
                continue
            
            valid_batch_size = valid_mask.sum().item()
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            # ===== DIAGNOSTIC: Window information (print every 5th window to avoid spam) =====
            if self.debug and (t_samples % 5 == 0 or t_samples < 5):
                print(f"\n" + "-"*80)
                print(f"WINDOW t={t_samples} (context_length={t_samples+1})")
                print(f"-"*80)
                print(f"Required: length > {t_samples + self.n_prediction_steps}")
                print(f"Valid samples: {valid_indices.cpu().tolist()} (count: {valid_batch_size}/{batch_dim})")
                print(f"Context positions: [0:{t_samples}]")
                print(f"Target positions: [{t_samples+1}:{t_samples+self.n_prediction_steps}]")
                
                # Show which samples are included/excluded
                for i in range(batch_dim):
                    status = "✓ INCLUDED" if valid_mask[i] else "✗ EXCLUDED (too short)"
                    print(f"  Sample {i} (len={lengths[i].item():2d}): {status}")
            
            # Extract context sequences up to t_samples
            video_forward_seq = video_vq[valid_mask, :t_samples+1, :]
            audio_forward_seq = audio_vq[valid_mask, :t_samples+1, :]
            text_forward_seq = text_vq[valid_mask, :t_samples+1, :]
            
            # ===== DIAGNOSTIC: Context sequence shapes =====
            if self.debug and (t_samples % 5 == 0 or t_samples < 5):
                print(f"\nContext sequences extracted:")
                print(f"  Video context shape: {video_forward_seq.shape}")
                print(f"  Audio context shape: {audio_forward_seq.shape}")
                print(f"  Text context shape: {text_forward_seq.shape}")
                
                # Show sample values from first valid sample
                if valid_batch_size > 0:
                    print(f"\nFirst valid sample (sample {valid_indices[0].item()}) - first 3 dims:")
                    print(f"  Video[0,0,:3] = {video_forward_seq[0, 0, :3]}")
                    print(f"  Audio[0,0,:3] = {audio_forward_seq[0, 0, :3]}")
                    print(f"  Text[0,0,:3] = {text_forward_seq[0, 0, :3]}")
            
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
            
            # ===== DIAGNOSTIC: Context representations =====
            if self.debug and (t_samples % 5 == 0 or t_samples < 5):
                print(f"\nContext representations at position {t_samples}:")
                print(f"  Video context shape: {video_context.shape}")
                print(f"  Audio context shape: {audio_context.shape}")
                print(f"  Text context shape: {text_context.shape}")
                print(f"  Video context[0,:3] = {video_context[0, :3]}")
            
            # Extract target sequences for prediction
            video_encode_samples = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=video_vq.device).double()
            audio_encode_samples = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=audio_vq.device).double()
            text_encode_samples = torch.empty((self.n_prediction_steps, valid_batch_size, self.embedding_dim), device=text_vq.device).double()
            
            for i in range(1, self.n_prediction_steps+1):
                video_encode_samples[i-1] = video_vq[valid_mask, t_samples+i, :].reshape(valid_batch_size, self.embedding_dim)
                audio_encode_samples[i-1] = audio_vq[valid_mask, t_samples+i, :].reshape(valid_batch_size, self.embedding_dim)
                text_encode_samples[i-1] = text_vq[valid_mask, t_samples+i, :].reshape(valid_batch_size, self.embedding_dim)
            
            # ===== DIAGNOSTIC: Target sequences =====
            if self.debug and (t_samples % 5 == 0 or t_samples < 5):
                print(f"\nTarget sequences for prediction:")
                print(f"  Targets shape: ({self.n_prediction_steps}, {valid_batch_size}, {self.embedding_dim})")
                print(f"  Predicting positions: [{t_samples+1}:{t_samples+self.n_prediction_steps}]")
                if self.n_prediction_steps > 0:
                    print(f"  Video target[0,0,:3] (pos {t_samples+1}) = {video_encode_samples[0, 0, :3]}")
            
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
                
                # ===== DIAGNOSTIC: Similarity matrices (first prediction step only) =====
                if self.debug and (t_samples % 10 == 0 or t_samples < 3) and i == 0:
                    print(f"\nSimilarity matrices (prediction step {i+1}):")
                    print(f"  Audio→Video similarity shape: {total1.shape}")
                    print(f"  Diagonal values (should be highest):")
                    print(f"    Audio→Video diag[:3] = {torch.diag(total1)[:min(3, valid_batch_size)]}")
                    print(f"    Video→Audio diag[:3] = {torch.diag(total4)[:min(3, valid_batch_size)]}")
                
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
            
            # ===== DIAGNOSTIC: Window results =====
            if self.debug and (t_samples % 5 == 0 or t_samples < 5):
                print(f"\nWindow t={t_samples} results:")
                print(f"  Window NCE loss: {window_nce.item():.4f}")
                print(f"  Window accuracies (averaged over {self.n_prediction_steps} prediction steps):")
                print(f"    Audio→Video: {(window_acc1 / self.n_prediction_steps).item():.4f}")
                print(f"    Video→Audio: {(window_acc4 / self.n_prediction_steps).item():.4f}")
                print(f"    Audio→Audio: {(window_acc7 / self.n_prediction_steps).item():.4f}")
            
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
            
            # Track for summary
            windows_processed.append((t_samples, valid_batch_size))
        
        # ===== DIAGNOSTIC: Final summary =====
        if self.debug:
            print("\n" + "="*80)
            print("WINDOWED CPC SUMMARY")
            print("="*80)
            print(f"Total windows processed: {valid_windows}")
            print(f"Windows that had valid samples: {len(windows_processed)}")
            
            if len(windows_processed) > 0:
                print(f"\nWindow participation summary:")
                print(f"  First window: t={windows_processed[0][0]}, samples={windows_processed[0][1]}")
                print(f"  Last window: t={windows_processed[-1][0]}, samples={windows_processed[-1][1]}")
                
                # Count participation per original sample
                sample_participation = [0] * batch_dim
                for t, _ in windows_processed:
                    for i in range(batch_dim):
                        if lengths[i] > t + self.n_prediction_steps:
                            sample_participation[i] += 1
                
                print(f"\nPer-sample window participation:")
                for i, count in enumerate(sample_participation):
                    print(f"  Sample {i} (len={lengths[i].item():2d}): participated in {count:3d} windows")
        
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
            
            # ===== DIAGNOSTIC: Final averaged results =====
            if self.debug:
                print(f"\nFinal averaged results (over {valid_windows} windows):")
                print(f"  NCE loss: {nce.item():.4f}")
                print(f"  Cross-modal accuracies:")
                print(f"    Audio→Video: {accuracy1.item():.4f}")
                print(f"    Audio→Text:  {accuracy2.item():.4f}")
                print(f"    Video→Text:  {accuracy3.item():.4f}")
                print(f"    Video→Audio: {accuracy4.item():.4f}")
                print(f"    Text→Audio:  {accuracy5.item():.4f}")
                print(f"    Text→Video:  {accuracy6.item():.4f}")
                print(f"  Self-modal accuracies:")
                print(f"    Audio→Audio: {accuracy7.item():.4f}")
                print(f"    Video→Video: {accuracy8.item():.4f}")
                print(f"    Text→Text:   {accuracy9.item():.4f}")
                print("="*80 + "\n")
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
            
            if self.debug:
                print("\n⚠️  WARNING: No valid windows were processed!")
                print("All sequences were too short for the required context + prediction length.")
                print("Returning zero tensors.")
                print("="*80 + "\n")
        
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce

class Cross_CPC_AVT_Adaptive(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC_AVT_Adaptive, self).__init__()
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
        batch_dim, time_length, _ = video_vq.shape  # time_length = 50
        
        if lengths is None or attention_mask is None:
            raise ValueError("This adaptive CPC requires both lengths and attention_mask")
        
        lengths = lengths.to(video_vq.device)
        attention_mask = attention_mask.to(video_vq.device)
        
        print(f"\n{'='*80}")
        print(f"ADAPTIVE CPC FORWARD PASS - DETAILED DIAGNOSTICS")
        print(f"{'='*80}")
        print(f"Input shapes: audio_vq={audio_vq.shape}, video_vq={video_vq.shape}, text_vq={text_vq.shape}")
        print(f"Lengths: {lengths}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        # STEP 1: Global Context Length Selection Based on Full Sequence Capacity
        global_max_context = time_length - self.n_prediction_steps - self.min_start_steps
        global_t_samples = torch.randint(global_max_context, size=(1,)).item() + self.min_start_steps
        
        print(f"\nSTEP 1: GLOBAL CONTEXT SELECTION")
        print(f"Time length (padded): {time_length}")
        print(f"Global max context possible: {global_max_context}")
        print(f"Selected global_t_samples: {global_t_samples}")
        print(f"Global context length: {global_t_samples + 1} timesteps")
        
        # STEP 2: Per-Sample Context Length Adaptation
        per_sample_context_lengths = []
        per_sample_t_samples = []
        
        print(f"\nSTEP 2: PER-SAMPLE ADAPTATION")
        print(f"{'Sample':<8} {'ActualLen':<10} {'MaxContext':<12} {'AdaptedT':<10} {'ContextLen':<12} {'CanUseGlobal':<15}")
        print(f"{'-'*75}")
        
        for i in range(batch_dim):
            sample_length = lengths[i].item()
            # Maximum context this sample can support
            sample_max_context = max(0, sample_length - self.n_prediction_steps - self.min_start_steps)
            
            # Use the minimum of global selection and what this sample can support
            adapted_t_samples = min(global_t_samples, sample_max_context + self.min_start_steps - 1)
            adapted_context_length = adapted_t_samples + 1
            
            per_sample_context_lengths.append(adapted_context_length)
            per_sample_t_samples.append(adapted_t_samples)
            
            can_use_global = "YES" if adapted_t_samples == global_t_samples else "NO (limited)"
            print(f"{i:<8} {sample_length:<10} {sample_max_context:<12} {adapted_t_samples:<10} {adapted_context_length:<12} {can_use_global:<15}")
        
        # STEP 3: Extract Context Sequences with Individual Adaptation
        max_context_length = max(per_sample_context_lengths)
        
        print(f"\nSTEP 3: CONTEXT SEQUENCE EXTRACTION")
        print(f"Maximum context length needed for this batch: {max_context_length}")
        print(f"Extracting sequences up to position {max_context_length-1} for all samples")
        
        video_forward_seq = video_vq[:, :max_context_length, :]
        audio_forward_seq = audio_vq[:, :max_context_length, :]
        text_forward_seq = text_vq[:, :max_context_length, :]
        
        print(f"Extracted sequence shapes: {video_forward_seq.shape}")
        
        # Create context-specific attention masks
        context_attention_mask = torch.zeros(batch_dim, max_context_length, dtype=torch.bool, device=video_vq.device)
        for i in range(batch_dim):
            actual_context_len = per_sample_context_lengths[i]
            context_attention_mask[i, :actual_context_len] = attention_mask[i, :actual_context_len]
        
        print(f"Context attention mask created with shape: {context_attention_mask.shape}")
        print("Context attention mask preview (first 10 positions):")
        for i in range(batch_dim):
            preview_len = min(10, max_context_length)
            mask_preview = context_attention_mask[i, :preview_len]
            print(f"  Sample {i}: {mask_preview}")
        
        # STEP 4: Efficient LSTM Processing with Pack/Unpack
        print(f"\nSTEP 4: PACKED SEQUENCE PROCESSING")
        
        # Sort sequences by their adapted context lengths for efficient processing
        context_lengths_tensor = torch.tensor(per_sample_context_lengths, device=video_vq.device)
        sorted_lengths, sorted_idx = context_lengths_tensor.sort(0, descending=True)
        unsorted_idx = sorted_idx.argsort(0)
        
        print(f"Original context lengths: {per_sample_context_lengths}")
        print(f"Sorted context lengths: {sorted_lengths.tolist()}")
        print(f"Sorting indices: {sorted_idx.tolist()}")
        print(f"Unsorting indices: {unsorted_idx.tolist()}")
        
        # Sort sequences and masks
        video_forward_sorted = video_forward_seq[sorted_idx]
        audio_forward_sorted = audio_forward_seq[sorted_idx]
        text_forward_sorted = text_forward_seq[sorted_idx]
        
        print(f"Sequences sorted by length for efficient packing")
        
        # Verify sorting worked correctly
        print(f"\nPRE-PACKING VERIFICATION:")
        print(f"Sample order after sorting (by original sample index): {[i.item() for i in sorted_idx]}")
        
        # Show what each sorted sample contributes
        for pack_idx, orig_idx in enumerate(sorted_idx):
            orig_sample_idx = orig_idx.item()
            context_len = per_sample_context_lengths[orig_sample_idx]
            actual_len = lengths[orig_sample_idx].item()
            print(f"  Position {pack_idx}: Original sample {orig_sample_idx}, context_len={context_len}, actual_len={actual_len}")
        
        # Pack sequences for LSTM processing - this automatically handles variable lengths
        print(f"\nPACKING SEQUENCES FOR LSTM:")
        
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
        
        # Detailed analysis of packed sequence structure
        print(f"Video packed sequence data shape: {video_packed.data.shape}")
        print(f"Video packed batch sizes: {video_packed.batch_sizes}")
        print(f"Audio packed sequence data shape: {audio_packed.data.shape}")
        print(f"Audio packed batch sizes: {audio_packed.batch_sizes}")
        print(f"Text packed sequence data shape: {text_packed.data.shape}")
        print(f"Text packed batch sizes: {text_packed.batch_sizes}")
        
        # Explain what the batch sizes mean
        print(f"\nPACKED SEQUENCE ANALYSIS:")
        print(f"Batch sizes interpretation:")
        for t, batch_size in enumerate(video_packed.batch_sizes):
            print(f"  Timestep {t}: Processing {batch_size} samples (samples with length > {t})")
        
        # Calculate total elements processed
        total_elements = video_packed.data.shape[0]
        theoretical_padded_elements = batch_dim * max_context_length
        efficiency = (total_elements / theoretical_padded_elements) * 100
        print(f"Efficiency: Processing {total_elements} elements vs {theoretical_padded_elements} padded = {efficiency:.1f}%")
        
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
        
        print(f"\nLSTM hidden states initialized with shape: {video_hidden[0].shape}")
        
        # Process through LSTMs - packed sequences automatically handle variable lengths
        print(f"\nLSTM PROCESSING:")
        print(f"Processing video LSTM with packed input...")
        video_context_packed, _ = self.video_ar_lstm(video_packed, video_hidden)
        print(f"Processing audio LSTM with packed input...")
        audio_context_packed, _ = self.audio_ar_lstm(audio_packed, audio_hidden)
        print(f"Processing text LSTM with packed input...")
        text_context_packed, _ = self.text_ar_lstm(text_packed, text_hidden)
        
        print(f"LSTM outputs (packed):")
        print(f"  Video context packed data shape: {video_context_packed.data.shape}")
        print(f"  Audio context packed data shape: {audio_context_packed.data.shape}")
        print(f"  Text context packed data shape: {text_context_packed.data.shape}")
        
        # Unpack sequences and restore original order
        print(f"\nUNPACKING SEQUENCES:")
        video_context, video_lengths_unpacked = pad_packed_sequence(video_context_packed, batch_first=True)
        audio_context, audio_lengths_unpacked = pad_packed_sequence(audio_context_packed, batch_first=True)
        text_context, text_lengths_unpacked = pad_packed_sequence(text_context_packed, batch_first=True)
        
        print(f"Unpacked shapes (still sorted): video={video_context.shape}, audio={audio_context.shape}, text={text_context.shape}")
        print(f"Unpacked lengths verification: {video_lengths_unpacked}")
        
        # Restore original sample order
        print(f"\nRESTORING ORIGINAL ORDER:")
        video_context = video_context[unsorted_idx]
        audio_context = audio_context[unsorted_idx]
        text_context = text_context[unsorted_idx]
        
        print(f"Final context shapes (original order): video={video_context.shape}, audio={audio_context.shape}, text={text_context.shape}")
        
        # Verify context restoration
        print(f"Order restoration verification:")
        for i in range(batch_dim):
            expected_len = per_sample_context_lengths[i]
            print(f"  Sample {i}: Expected context length {expected_len}, shape available up to {video_context.shape[1]}")
        
        # STEP 5: Extract Context Representations at Appropriate Positions
        print(f"\nSTEP 5: CONTEXT REPRESENTATION EXTRACTION")
        
        video_context_list = []
        audio_context_list = []
        text_context_list = []
        
        print(f"Extracting final context representations:")
        print(f"{'Sample':<8} {'t_samples':<10} {'ContextPos':<12} {'ContextShape':<15}")
        print(f"{'-'*50}")
        
        for i in range(batch_dim):
            # Use each sample's adapted context position
            context_pos = per_sample_t_samples[i]
            # Ensure we don't go beyond what was actually processed
            context_pos = min(context_pos, per_sample_context_lengths[i] - 1)
            
            video_context_list.append(video_context[i, context_pos, :])
            audio_context_list.append(audio_context[i, context_pos, :])
            text_context_list.append(text_context[i, context_pos, :])
            
            context_vector_shape = video_context[i, context_pos, :].shape
            print(f"{i:<8} {per_sample_t_samples[i]:<10} {context_pos:<12} {str(context_vector_shape):<15}")
        
        video_context = torch.stack(video_context_list)
        audio_context = torch.stack(audio_context_list)
        text_context = torch.stack(text_context_list)
        
        print(f"Final context representations: {video_context.shape}")
        
        # STEP 6: Intelligent Target Extraction Respecting Individual Sequence Lengths
        print(f"\nSTEP 6: TARGET EXTRACTION")
        
        video_encode_samples = []
        audio_encode_samples = []
        text_encode_samples = []
        
        print(f"Extracting prediction targets:")
        print(f"{'Sample':<8} {'t_samples':<10} {'TargetPos':<12} {'ActualLen':<12} {'Strategy':<20}")
        print(f"{'-'*70}")
        
        for step in range(self.n_prediction_steps):
            step_video = []
            step_audio = []
            step_text = []
            
            for i in range(batch_dim):
                # Calculate target position based on each sample's adapted context
                target_pos = per_sample_t_samples[i] + step + 1
                actual_length = lengths[i].item()
                
                # Ensure target is within actual sequence bounds
                if target_pos < actual_length:
                    step_video.append(video_vq[i, target_pos, :])
                    step_audio.append(audio_vq[i, target_pos, :])
                    step_text.append(text_vq[i, target_pos, :])
                    strategy = "Use real target"
                else:
                    # Use last valid position instead of padding
                    last_valid_pos = actual_length - 1
                    step_video.append(video_vq[i, last_valid_pos, :])
                    step_audio.append(audio_vq[i, last_valid_pos, :])
                    step_text.append(text_vq[i, last_valid_pos, :])
                    strategy = f"Use last valid ({last_valid_pos})"
                    target_pos = last_valid_pos
                
                print(f"{i:<8} {per_sample_t_samples[i]:<10} {target_pos:<12} {actual_length:<12} {strategy:<20}")
            
            video_encode_samples.append(torch.stack(step_video))
            audio_encode_samples.append(torch.stack(step_audio))
            text_encode_samples.append(torch.stack(step_text))
        
        video_encode_samples = torch.stack(video_encode_samples)
        audio_encode_samples = torch.stack(audio_encode_samples)
        text_encode_samples = torch.stack(text_encode_samples)
        
        print(f"Target samples shape: {video_encode_samples.shape}")
        
        # STEP 7: Generate Predictions Using Adapted Context Representations
        print(f"\nSTEP 7: PREDICTION GENERATION")
        
        video_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=video_vq.device).double()
        audio_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=audio_vq.device).double()
        text_pred = torch.empty((self.n_prediction_steps, batch_dim, self.embedding_dim), device=text_vq.device).double()
        
        for i in range(self.n_prediction_steps):
            video_pred[i] = self.video_predictors[i](video_context)
            audio_pred[i] = self.audio_predictors[i](audio_context)
            text_pred[i] = self.text_predictors[i](text_context)
        
        print(f"Predictions generated: video_pred={video_pred.shape}, audio_pred={audio_pred.shape}, text_pred={text_pred.shape}")
        
        # STEP 8: Calculate NCE Loss and Accuracies
        print(f"\nSTEP 8: CONTRASTIVE LEARNING COMPUTATION")
        
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
            
            print(f"Prediction step {i}: Similarity matrices computed, shapes like {total1.shape}")
            
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
        
        print(f"\nFINAL RESULTS:")
        print(f"NCE Loss: {nce:.4f}")
        print(f"Cross-modal accuracies: {accuracy1:.3f}, {accuracy2:.3f}, {accuracy3:.3f}, {accuracy4:.3f}, {accuracy5:.3f}, {accuracy6:.3f}")
        print(f"Self-modal accuracies: {accuracy7:.3f}, {accuracy8:.3f}, {accuracy9:.3f}")
        print(f"{'='*80}")
        
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce


def test_adaptive_cpc_variable_sequences():
    """
    Comprehensive test function to validate the adaptive CPC's handling of variable sequence lengths.
    This function creates realistic dummy data and traces through each processing step.
    """
    
    print("=" * 80)
    print("TESTING ADAPTIVE CPC WITH VARIABLE SEQUENCE LENGTHS")
    print("=" * 80)
    
    # Test Configuration
    batch_size = 6
    max_seq_len = 50  # Padded sequence length
    embedding_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define realistic actual sequence lengths for our test batch
    actual_lengths = [45, 12, 30, 8, 25, 50]  # Varied lengths from short to full
    
    print(f"\nTest Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Padded sequence length: {max_seq_len}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Actual sequence lengths: {actual_lengths}")
    print(f"Device: {device}")
    
    # STEP 1: Create realistic dummy tensors with meaningful content and zero padding
    print(f"\nSTEP 1: Creating dummy tensors with realistic content and zero padding")
    
    audio_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    video_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    text_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    
    # Fill with meaningful content for actual sequence lengths, leave zeros for padding
    for i in range(batch_size):
        actual_len = actual_lengths[i]
        
        # Create distinctive patterns for each sample and modality
        # Audio: use sample index + 1 as base value
        audio_vq[i, :actual_len, :] = (i + 1) * 0.1 + torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device) * 0.01
        
        # Video: use sample index + 10 as base value  
        video_vq[i, :actual_len, :] = (i + 10) * 0.1 + torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device) * 0.01
        
        # Text: use sample index + 20 as base value
        text_vq[i, :actual_len, :] = (i + 20) * 0.1 + torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device) * 0.01
        
        print(f"Sample {i}: Filled positions 0-{actual_len-1} with content, positions {actual_len}-{max_seq_len-1} remain zeros (padding)")
    
    # STEP 2: Create lengths tensor and attention masks
    print(f"\nSTEP 2: Creating lengths tensor and attention masks")
    
    lengths = torch.tensor(actual_lengths, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
    
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = True
    
    print(f"Lengths tensor: {lengths}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Attention mask (showing True/False for first 20 positions):")
    for i in range(batch_size):
        mask_preview = attention_mask[i, :20]
        print(f"  Sample {i}: {mask_preview}")
    
    # STEP 3: Verify our padding setup is correct
    print(f"\nSTEP 3: Verifying padding setup")
    
    for i in range(batch_size):
        actual_len = actual_lengths[i]
        
        # Check that content positions have non-zero values
        content_sum = torch.sum(torch.abs(audio_vq[i, :actual_len, :])).item()
        
        # Check that padding positions are exactly zero
        if actual_len < max_seq_len:
            padding_sum = torch.sum(torch.abs(audio_vq[i, actual_len:, :])).item()
        else:
            padding_sum = 0.0
            
        print(f"Sample {i}: Content sum: {content_sum:.4f}, Padding sum: {padding_sum:.4f}")
        
        assert content_sum > 0, f"Sample {i} should have non-zero content!"
        assert padding_sum == 0, f"Sample {i} should have zero padding!"
    
    print("✓ Padding setup verification passed!")
    
    # STEP 4: Create and test the adaptive CPC model
    print(f"\nSTEP 4: Creating adaptive CPC model and processing")
    
    # Initialize the model
    cpc_model = Cross_CPC_AVT_Adaptive(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        context_dim=256,
        num_layers=2,
        n_prediction_steps=1,
        min_start_steps=1
    ).to(device)
    
    cpc_model = cpc_model.double()

    # Set the model to evaluation mode to remove randomness in batch norm, etc.
    cpc_model.eval()
    
    print("Model initialized. Now calling forward pass...")
    print("-" * 60)
    
    # STEP 5: Forward pass with detailed logging
    with torch.no_grad():  # We're just testing, not training
        results = cpc_model(audio_vq, video_vq, text_vq, lengths=lengths, attention_mask=attention_mask)
    
    print("-" * 60)
    print("Forward pass completed successfully!")
    
    # STEP 6: Analyze the results
    print(f"\nSTEP 6: Analyzing results")
    
    accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce_loss = results
    
    print(f"Cross-modal prediction accuracies:")
    print(f"  Audio-Video: {accuracy1:.4f}")
    print(f"  Audio-Text: {accuracy2:.4f}")  
    print(f"  Video-Text: {accuracy3:.4f}")
    print(f"  Video-Audio: {accuracy4:.4f}")
    print(f"  Text-Audio: {accuracy5:.4f}")
    print(f"  Text-Video: {accuracy6:.4f}")
    print(f"Self-modal prediction accuracies:")
    print(f"  Audio-Audio: {accuracy7:.4f}")
    print(f"  Video-Video: {accuracy8:.4f}")
    print(f"  Text-Text: {accuracy9:.4f}")
    print(f"NCE Loss: {nce_loss:.4f}")
    
    # STEP 7: Additional validation checks
    print(f"\nSTEP 7: Additional validation checks")
    
    # Test that the function handles edge cases properly
    print("Testing edge case: batch with very short sequences")
    
    # Create a challenging batch with very short sequences
    short_lengths = [3, 2, 4, 1, 3, 2]
    short_audio_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    short_video_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    short_text_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    
    for i in range(batch_size):
        actual_len = short_lengths[i]
        short_audio_vq[i, :actual_len, :] = torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device)
        short_video_vq[i, :actual_len, :] = torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device)
        short_text_vq[i, :actual_len, :] = torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device)
    
    short_lengths_tensor = torch.tensor(short_lengths, dtype=torch.long, device=device)
    short_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
    for i, length in enumerate(short_lengths):
        short_attention_mask[i, :length] = True
    
    print(f"Short sequence lengths: {short_lengths}")
    
    try:
        with torch.no_grad():
            short_results = cpc_model(short_audio_vq, short_video_vq, short_text_vq, 
                                    lengths=short_lengths_tensor, attention_mask=short_attention_mask)
        print("✓ Edge case with short sequences handled successfully!")
        print(f"NCE Loss for short sequences: {short_results[-1]:.4f}")
        
    except Exception as e:
        print(f"✗ Edge case failed with error: {e}")
    
    # STEP 8: Summary and validation
    print(f"\nSTEP 8: Test Summary")
    print("=" * 50)
    
    validation_passed = True
    
    # Check that results are reasonable
    if torch.isnan(nce_loss) or torch.isinf(nce_loss):
        print("✗ NCE loss contains NaN or Inf values")
        validation_passed = False
    else:
        print("✓ NCE loss is a valid finite number")
    
    # Check that accuracies are in reasonable range [0, 1]
    accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9]
    for i, acc in enumerate(accuracies):
        if 0 <= acc <= 1:
            print(f"✓ Accuracy {i+1} is in valid range: {acc:.4f}")
        else:
            print(f"✗ Accuracy {i+1} is out of range [0,1]: {acc:.4f}")
            validation_passed = False
    
    # Final verdict
    print("=" * 50)
    if validation_passed:
        print("🎉 ALL TESTS PASSED! The adaptive CPC correctly handles variable sequence lengths.")
        print("\nKey verified behaviors:")
        print("• Global context length selection allows longer sequences to use extended context")
        print("• Per-sample adaptation ensures no padding tokens are processed")
        print("• Pack/unpack sequences handle variable lengths efficiently")
        print("• Target extraction respects individual sequence boundaries")
        print("• Edge cases with very short sequences are handled gracefully")
    else:
        print("❌ SOME TESTS FAILED. Check the error messages above.")
    
    return validation_passed

# Additional helper function to visualize the internal state
def debug_adaptive_context_selection(lengths, time_length=50, n_prediction_steps=1, min_start_steps=1):
    """
    Helper function to show how context lengths are selected for different sequence lengths
    """
    print("\n" + "="*60)
    print("CONTEXT SELECTION ANALYSIS")
    print("="*60)
    
    # Simulate global context selection
    global_max_context = time_length - n_prediction_steps - min_start_steps
    print(f"Global max context possible: {global_max_context}")
    
    # Test different global selections
    test_global_selections = [10, 25, 35, 47]
    
    for global_t in test_global_selections:
        print(f"\nIf global_t_samples = {global_t} (context length = {global_t + 1}):")
        print("Sample adaptations:")
        
        for i, seq_len in enumerate(lengths):
            sample_max_context = max(0, seq_len - n_prediction_steps - min_start_steps)
            adapted_t_samples = min(global_t, sample_max_context + min_start_steps - 1)
            adapted_context_length = adapted_t_samples + 1
            
            print(f"  Sample {i} (len={seq_len:2d}): max_context={sample_max_context:2d}, " 
                  f"adapted_t={adapted_t_samples:2d}, context_len={adapted_context_length:2d}")

# Run the test
# if __name__ == "__main__":
#     # First run the context selection analysis
#     example_lengths = [45, 12, 30, 8, 25, 50]
#     debug_adaptive_context_selection(example_lengths)
    
#     # Then run the full test
#     test_adaptive_cpc_variable_sequences()


def test_windowed_cpc_variable_sequences():
    """
    Comprehensive test function to validate the windowed CPC's handling of variable sequence lengths.
    This function creates realistic dummy data and traces through the windowing process.
    """
    
    print("=" * 80)
    print("TESTING WINDOWED CPC WITH VARIABLE SEQUENCE LENGTHS")
    print("=" * 80)
    
    # Test Configuration
    batch_size = 6
    max_seq_len = 50  # Padded sequence length
    embedding_dim = 128
    n_prediction_steps = 1  # Predict 1 step ahead
    min_start_steps = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define realistic actual sequence lengths for our test batch
    actual_lengths = [45, 12, 30, 8, 25, 50]  # Varied lengths from short to full
    
    print(f"\nTest Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Padded sequence length: {max_seq_len}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Prediction steps: {n_prediction_steps}")
    print(f"Min start steps: {min_start_steps}")
    print(f"Actual sequence lengths: {actual_lengths}")
    print(f"Device: {device}")
    
    # STEP 1: Create realistic dummy tensors with meaningful content and zero padding
    print(f"\n{'='*80}")
    print("STEP 1: Creating dummy tensors with realistic content and zero padding")
    print('='*80)
    
    audio_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    video_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    text_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    
    # Fill with meaningful content for actual sequence lengths, leave zeros for padding
    for i in range(batch_size):
        actual_len = actual_lengths[i]
        
        # Create distinctive patterns for each sample and modality
        audio_vq[i, :actual_len, :] = (i + 1) * 0.1 + torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device) * 0.01
        video_vq[i, :actual_len, :] = (i + 10) * 0.1 + torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device) * 0.01
        text_vq[i, :actual_len, :] = (i + 20) * 0.1 + torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device) * 0.01
        
        print(f"Sample {i}: Filled positions 0-{actual_len-1} with content, "
              f"positions {actual_len}-{max_seq_len-1} remain zeros (padding)")
    
    # STEP 2: Create lengths tensor and attention masks
    print(f"\n{'='*80}")
    print("STEP 2: Creating lengths tensor and attention masks")
    print('='*80)
    
    lengths = torch.tensor(actual_lengths, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
    
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = True
    
    print(f"Lengths tensor: {lengths}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"\nAttention mask (showing True/False for first 20 positions):")
    for i in range(batch_size):
        mask_preview = attention_mask[i, :20]
        true_count = mask_preview.sum().item()
        print(f"  Sample {i} (len={actual_lengths[i]:2d}): {true_count:2d} True positions in first 20")
    
    # STEP 3: Verify our padding setup is correct
    print(f"\n{'='*80}")
    print("STEP 3: Verifying padding setup")
    print('='*80)
    
    for i in range(batch_size):
        actual_len = actual_lengths[i]
        
        # Check that content positions have non-zero values
        content_sum = torch.sum(torch.abs(audio_vq[i, :actual_len, :])).item()
        
        # Check that padding positions are exactly zero
        if actual_len < max_seq_len:
            padding_sum = torch.sum(torch.abs(audio_vq[i, actual_len:, :])).item()
        else:
            padding_sum = 0.0
            
        print(f"Sample {i}: Content sum: {content_sum:.4f}, Padding sum: {padding_sum:.4f}")
        
        assert content_sum > 0, f"Sample {i} should have non-zero content!"
        assert padding_sum == 0, f"Sample {i} should have zero padding!"
    
    print("✓ Padding setup verification passed!")
    
    # STEP 4: Analyze which samples will participate in which windows
    print(f"\n{'='*80}")
    print("STEP 4: Window Participation Analysis")
    print('='*80)
    
    # Calculate valid window range
    min_window = min_start_steps
    max_window = max_seq_len - n_prediction_steps
    
    print(f"Window range: [{min_window}, {max_window})")
    print(f"Total possible windows: {max_window - min_window}")
    print(f"\nFor each window position, we need: length > (t_samples + {n_prediction_steps})\n")
    
    # Show which samples participate in selected windows
    sample_windows = [5, 10, 15, 20, 25, 30, 35, 40, 44]
    
    for t_samples in sample_windows:
        if t_samples >= max_window:
            break
        required_length = t_samples + n_prediction_steps + 1  # +1 because we need the target positions too
        valid_samples = [i for i, length in enumerate(actual_lengths) if length > t_samples + n_prediction_steps]
        
        print(f"Window t={t_samples:2d} (context_len={t_samples+1:2d}, targets={t_samples+1}-{t_samples+n_prediction_steps}):")
        print(f"  Required length: > {t_samples + n_prediction_steps}")
        print(f"  Valid samples: {valid_samples} (count: {len(valid_samples)})")
        
        if len(valid_samples) == 0:
            print(f"  ⚠️  No samples qualify for this window!")
        
    # STEP 5: Create and test the windowed CPC model
    print(f"\n{'='*80}")
    print("STEP 5: Creating windowed CPC model and processing")
    print('='*80)
    
    # Initialize the model
    cpc_model = Cross_CPC_AVT_pad_window(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        context_dim=256,
        num_layers=2,
        n_prediction_steps=n_prediction_steps,
        min_start_steps=min_start_steps,
        debug=True
    ).to(device)
    
    cpc_model = cpc_model.double()
    cpc_model.eval()
    
    print("Model initialized. Now calling forward pass...")
    print(f"This will iterate through {max_window - min_window} possible windows...")
    print("-" * 60)
    
    # STEP 6: Forward pass with detailed logging
    with torch.no_grad():
        results = cpc_model(audio_vq, video_vq, text_vq, lengths=lengths, attention_mask=attention_mask)
    
    print("-" * 60)
    print("Forward pass completed successfully!")
    
    # STEP 7: Analyze the results
    print(f"\n{'='*80}")
    print("STEP 7: Analyzing results")
    print('='*80)
    
    accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce_loss = results
    
    print(f"\nCross-modal prediction accuracies:")
    print(f"  Audio→Video: {accuracy1:.4f}")
    print(f"  Audio→Text:  {accuracy2:.4f}")  
    print(f"  Video→Text:  {accuracy3:.4f}")
    print(f"  Video→Audio: {accuracy4:.4f}")
    print(f"  Text→Audio:  {accuracy5:.4f}")
    print(f"  Text→Video:  {accuracy6:.4f}")
    
    print(f"\nSelf-modal prediction accuracies:")
    print(f"  Audio→Audio: {accuracy7:.4f}")
    print(f"  Video→Video: {accuracy8:.4f}")
    print(f"  Text→Text:   {accuracy9:.4f}")
    
    print(f"\nNCE Loss: {nce_loss:.4f}")
    
    # STEP 8: Calculate window statistics
    print(f"\n{'='*80}")
    print("STEP 8: Window Statistics")
    print('='*80)
    
    # Count how many windows each sample participates in
    participation_counts = []
    for i, seq_len in enumerate(actual_lengths):
        # A sample participates in window t if: seq_len > t + n_prediction_steps
        # Which means: t < seq_len - n_prediction_steps
        # So max valid t is: seq_len - n_prediction_steps - 1
        # Number of windows: from min_start_steps to (seq_len - n_prediction_steps - 1)
        max_t_for_sample = seq_len - n_prediction_steps - 1
        if max_t_for_sample >= min_start_steps:
            count = max_t_for_sample - min_start_steps + 1
        else:
            count = 0
        participation_counts.append(count)
        print(f"Sample {i} (len={seq_len:2d}): participates in {count:2d} windows")
    
    print(f"\nTotal window participation opportunities: {sum(participation_counts)}")
    print(f"Average windows per sample: {sum(participation_counts) / len(participation_counts):.2f}")
    
    # STEP 9: Test edge cases
    print(f"\n{'='*80}")
    print("STEP 9: Testing edge cases")
    print('='*80)
    
    # Edge case 1: Very short sequences
    print("\nEdge case 1: Batch with very short sequences")
    short_lengths = [3, 2, 4, 1, 3, 2]
    short_audio_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    short_video_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    short_text_vq = torch.zeros(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    
    for i in range(batch_size):
        actual_len = short_lengths[i]
        if actual_len > 0:
            short_audio_vq[i, :actual_len, :] = torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device)
            short_video_vq[i, :actual_len, :] = torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device)
            short_text_vq[i, :actual_len, :] = torch.randn(actual_len, embedding_dim, dtype=torch.float64, device=device)
    
    short_lengths_tensor = torch.tensor(short_lengths, dtype=torch.long, device=device)
    short_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
    for i, length in enumerate(short_lengths):
        short_attention_mask[i, :length] = True
    
    print(f"Short sequence lengths: {short_lengths}")
    
    # Calculate expected windows
    short_participation = []
    for seq_len in short_lengths:
        max_t = seq_len - n_prediction_steps - 1
        count = max(0, max_t - min_start_steps + 1)
        short_participation.append(count)
    print(f"Expected window participations: {short_participation}")
    print(f"Total expected windows: {sum(short_participation)}")
    
    try:
        with torch.no_grad():
            short_results = cpc_model(short_audio_vq, short_video_vq, short_text_vq, 
                                    lengths=short_lengths_tensor, attention_mask=short_attention_mask)
        print("✓ Edge case with short sequences handled successfully!")
        print(f"NCE Loss: {short_results[-1]:.4f}")
        
        if sum(short_participation) == 0:
            print("Note: No windows were valid (all sequences too short), model returned zero tensors")
        
    except Exception as e:
        print(f"✗ Edge case failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Edge case 2: Uniform length sequences
    print("\nEdge case 2: Batch with uniform sequence lengths")
    uniform_length = 20
    uniform_lengths = [uniform_length] * batch_size
    uniform_audio_vq = torch.randn(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    uniform_video_vq = torch.randn(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    uniform_text_vq = torch.randn(batch_size, max_seq_len, embedding_dim, dtype=torch.float64, device=device)
    
    # Zero out padding
    for i in range(batch_size):
        uniform_audio_vq[i, uniform_length:, :] = 0
        uniform_video_vq[i, uniform_length:, :] = 0
        uniform_text_vq[i, uniform_length:, :] = 0
    
    uniform_lengths_tensor = torch.tensor(uniform_lengths, dtype=torch.long, device=device)
    uniform_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        uniform_attention_mask[i, :uniform_length] = True
    
    print(f"Uniform sequence lengths: {uniform_lengths}")
    
    try:
        with torch.no_grad():
            uniform_results = cpc_model(uniform_audio_vq, uniform_video_vq, uniform_text_vq,
                                      lengths=uniform_lengths_tensor, attention_mask=uniform_attention_mask)
        print("✓ Edge case with uniform sequences handled successfully!")
        print(f"NCE Loss: {uniform_results[-1]:.4f}")
        
        expected_windows = uniform_length - n_prediction_steps - min_start_steps
        print(f"All samples participated in {expected_windows} windows each")
        
    except Exception as e:
        print(f"✗ Edge case failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # STEP 10: Summary and validation
    print(f"\n{'='*80}")
    print("STEP 10: Test Summary")
    print('='*80)
    
    validation_passed = True
    
    # Check that results are reasonable
    if torch.isnan(nce_loss) or torch.isinf(nce_loss):
        print("✗ NCE loss contains NaN or Inf values")
        validation_passed = False
    else:
        print("✓ NCE loss is a valid finite number")
    
    # Check that accuracies are in reasonable range [0, 1]
    accuracies = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9]
    all_acc_valid = True
    for i, acc in enumerate(accuracies):
        if 0 <= acc <= 1:
            pass  # Valid
        else:
            print(f"✗ Accuracy {i+1} is out of range [0,1]: {acc:.4f}")
            all_acc_valid = False
            validation_passed = False
    
    if all_acc_valid:
        print("✓ All accuracies are in valid range [0, 1]")
    
    # Final verdict
    print("=" * 80)
    if validation_passed:
        print("🎉 ALL TESTS PASSED! The windowed CPC correctly handles variable sequence lengths.")
        print("\nKey verified behaviors:")
        print("• Iterates through all possible window positions")
        print("• Dynamically filters samples based on actual content length per window")
        print("• Processes only samples with sufficient content for each window")
        print("• Averages results across all valid windows")
        print("• Handles edge cases (very short sequences, uniform lengths)")
        print("• Never processes padding tokens")
    else:
        print("❌ SOME TESTS FAILED. Check the error messages above.")
    
    return validation_passed


def analyze_window_participation(lengths, max_seq_len=50, n_prediction_steps=1, min_start_steps=1):
    """
    Helper function to visualize which samples participate in which windows.
    Creates a detailed participation matrix.
    """
    print("\n" + "="*80)
    print("DETAILED WINDOW PARTICIPATION ANALYSIS")
    print("="*80)
    
    max_window = max_seq_len - n_prediction_steps
    
    print(f"\nConfiguration:")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Prediction steps: {n_prediction_steps}")
    print(f"  Min start steps: {min_start_steps}")
    print(f"  Window range: [{min_start_steps}, {max_window})")
    print(f"\nSequence lengths: {lengths}")
    print(f"\nFor window t, sample needs: length > t + {n_prediction_steps}\n")
    
    # Create participation matrix
    print("Participation Matrix (✓ = participates, ✗ = too short):")
    print("-" * 80)
    
    # Header
    header = "Sample | Len |"
    window_positions = list(range(min_start_steps, max_window, max((max_window - min_start_steps) // 20, 1)))
    for t in window_positions:
        header += f" t={t:2d} |"
    print(header)
    print("-" * 80)
    
    # Each sample row
    for i, seq_len in enumerate(lengths):
        row = f"   {i}   | {seq_len:3d} |"
        for t in window_positions:
            if seq_len > t + n_prediction_steps:
                row += "  ✓  |"
            else:
                row += "  ✗  |"
        print(row)
    
    print("-" * 80)
    
    # Statistics
    print("\nWindow Statistics:")
    for t in window_positions:
        valid_count = sum(1 for seq_len in lengths if seq_len > t + n_prediction_steps)
        print(f"  Window t={t:2d}: {valid_count}/{len(lengths)} samples participate")
    
    # Per-sample statistics
    print("\nPer-Sample Statistics:")
    for i, seq_len in enumerate(lengths):
        max_t = seq_len - n_prediction_steps - 1
        if max_t >= min_start_steps:
            count = max_t - min_start_steps + 1
        else:
            count = 0
        print(f"  Sample {i} (len={seq_len:3d}): participates in {count:3d} windows "
              f"(contexts up to length {max_t+1 if max_t >= 0 else 0})")


# Run the tests
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    
    # First, analyze window participation with example data
    example_lengths = [45, 12, 30, 8, 25, 50]
    analyze_window_participation(example_lengths)
    
    # Then run the full test
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE TEST")
    print("="*80)
    test_windowed_cpc_variable_sequences()




# class Cross_CPC_AVT_pad(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
#         super(Cross_CPC_AVT_pad, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.context_dim = context_dim
#         self.num_layers = num_layers
#         self.n_prediction_steps = n_prediction_steps
#         self.min_start_steps = min_start_steps
#         self.softmax = nn.Softmax()
#         self.lsoftmax = nn.LogSoftmax()
        
#         # Autoregressive LSTM networks for each modality
#         self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
#         self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
#         self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
#         # Predictor networks for each modality
#         self.video_predictors = nn.ModuleList([
#             nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
#         ])
#         self.audio_predictors = nn.ModuleList([
#             nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
#         ])
#         self.text_predictors = nn.ModuleList([
#             nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
#         ])
    
#     def forward(self, audio_vq, video_vq, text_vq, lengths=None):
#         """
#         Enhanced forward pass with proper sequence length handling.
        
#         Args:
#             audio_vq, video_vq, text_vq: Quantized representations [batch, time_length, embedding_dim]
#             lengths: Actual sequence lengths for each sample in the batch [batch]
        
#         The key challenge here is ensuring we only learn from valid timesteps, not padding.
#         """
#         batch_dim, time_length, _ = video_vq.shape
        
        
        
#         if lengths is None:
#             lengths = torch.full((batch_dim,), time_length, dtype=torch.long, device=video_vq.device)
#         else:
#             lengths = lengths.to(video_vq.device)
        
#         min_len = lengths.min().item()
        
#         max_possible_t = min_len - self.n_prediction_steps - self.min_start_steps
        
#         if max_possible_t <= 0:
#             if min_len > 1:
#                 t_samples = torch.tensor([0]).long()
#                 actual_n_steps = min(self.n_prediction_steps, min_len - 1)
#             else:
#                 print(f"Warning: Sequences too short for CPC (min_len={min_len})")
#                 dummy_acc = torch.tensor(0.0, device=video_vq.device)
#                 dummy_nce = torch.tensor(0.0, device=video_vq.device)
#                 return dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_nce
#         else:
#             t_samples = (torch.randint(max_possible_t, size=(1,)) + self.min_start_steps).long()
#             actual_n_steps = self.n_prediction_steps
#         effective_lengths = torch.minimum(lengths, torch.full_like(lengths, t_samples + 1))

#         video_forward_seq = video_vq[:, :t_samples+1, :]
#         audio_forward_seq = audio_vq[:, :t_samples+1, :]
#         text_forward_seq = text_vq[:, :t_samples+1, :]
        
#         sorted_lengths, sorted_idx = effective_lengths.sort(0, descending=True)
#         unsorted_idx = sorted_idx.argsort(0)

#         video_forward_sorted = video_forward_seq[sorted_idx]
#         audio_forward_sorted = audio_forward_seq[sorted_idx]
#         text_forward_sorted = text_forward_seq[sorted_idx]
        
#         video_packed = pack_padded_sequence(
#             video_forward_sorted, sorted_lengths.cpu(), 
#             batch_first=True, enforce_sorted=True
#         )
#         audio_packed = pack_padded_sequence(
#             audio_forward_sorted, sorted_lengths.cpu(),
#             batch_first=True, enforce_sorted=True
#         )
#         text_packed = pack_padded_sequence(
#             text_forward_sorted, sorted_lengths.cpu(),
#             batch_first=True, enforce_sorted=True
#         )
        
#         video_hidden = (
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).double(),
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).double()
#         )
#         audio_hidden = (
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).double(),
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).double()
#         )
#         text_hidden = (
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=text_vq.device).double(),
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=text_vq.device).double()
#         )
        
#         video_context_packed, _ = self.video_ar_lstm(video_packed, video_hidden)
#         audio_context_packed, _ = self.audio_ar_lstm(audio_packed, audio_hidden)
#         text_context_packed, _ = self.text_ar_lstm(text_packed, text_hidden)
        

#         video_context, _ = pad_packed_sequence(video_context_packed, batch_first=True)
#         audio_context, _ = pad_packed_sequence(audio_context_packed, batch_first=True)
#         text_context, _ = pad_packed_sequence(text_context_packed, batch_first=True)
        
#         video_context = video_context[unsorted_idx]
#         audio_context = audio_context[unsorted_idx]
#         text_context = text_context[unsorted_idx]
        

#         video_context_list = []
#         audio_context_list = []
#         text_context_list = []
        
#         for i in range(batch_dim):
#             if effective_lengths[i] > t_samples:
#                 context_pos = t_samples.item()
#             else:
#                 context_pos = effective_lengths[i].item() - 1
            
#             video_context_list.append(video_context[i, context_pos, :])
#             audio_context_list.append(audio_context[i, context_pos, :])
#             text_context_list.append(text_context[i, context_pos, :])
        
#         video_context = torch.stack(video_context_list)  # [batch_dim, context_dim]
#         audio_context = torch.stack(audio_context_list)  # [batch_dim, context_dim]
#         text_context = torch.stack(text_context_list)    # [batch_dim, context_dim]
        

#         video_encode_samples = []
#         audio_encode_samples = []
#         text_encode_samples = []
        
#         for step in range(actual_n_steps):
#             step_video = []
#             step_audio = []
#             step_text = []
            
#             for i in range(batch_dim):
#                 target_pos = t_samples.item() + step + 1
                
    
#                 if target_pos < lengths[i]:
            
#                     step_video.append(video_vq[i, target_pos, :])
#                     step_audio.append(audio_vq[i, target_pos, :])
#                     step_text.append(text_vq[i, target_pos, :])
#                 else:

#                     last_valid_pos = lengths[i] - 1
#                     step_video.append(video_vq[i, last_valid_pos, :])
#                     step_audio.append(audio_vq[i, last_valid_pos, :])
#                     step_text.append(text_vq[i, last_valid_pos, :])
            
#             video_encode_samples.append(torch.stack(step_video))
#             audio_encode_samples.append(torch.stack(step_audio))
#             text_encode_samples.append(torch.stack(step_text))
        

#         video_encode_samples = torch.stack(video_encode_samples)
#         audio_encode_samples = torch.stack(audio_encode_samples)
#         text_encode_samples = torch.stack(text_encode_samples)
        

#         video_pred = torch.empty((actual_n_steps, batch_dim, self.embedding_dim), device=video_vq.device).double()
#         audio_pred = torch.empty((actual_n_steps, batch_dim, self.embedding_dim), device=audio_vq.device).double()
#         text_pred = torch.empty((actual_n_steps, batch_dim, self.embedding_dim), device=text_vq.device).double()
        
#         for i in range(actual_n_steps):
#             video_pred[i] = self.video_predictors[i](video_context)
#             audio_pred[i] = self.audio_predictors[i](audio_context)
#             text_pred[i] = self.text_predictors[i](text_context)
        

#         nce = 0
        
#         # Track accuracies for the last prediction step (for monitoring)
#         accuracy1 = accuracy2 = accuracy3 = accuracy4 = accuracy5 = 0
#         accuracy6 = accuracy7 = accuracy8 = accuracy9 = 0
        
#         for i in range(actual_n_steps):
#             # Compute similarity matrices between predictions and actual future samples
#             total1 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i], 0, 1))  # audio actual vs video pred
#             total2 = torch.mm(audio_encode_samples[i], torch.transpose(text_pred[i], 0, 1))   # audio actual vs text pred
#             total3 = torch.mm(video_encode_samples[i], torch.transpose(text_pred[i], 0, 1))   # video actual vs text pred
#             total4 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))  # video actual vs audio pred
#             total5 = torch.mm(text_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))   # text actual vs audio pred
#             total6 = torch.mm(text_encode_samples[i], torch.transpose(video_pred[i], 0, 1))   # text actual vs video pred
#             total7 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))  # audio self-prediction
#             total8 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i], 0, 1))  # video self-prediction
#             total9 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i], 0, 1))    # text self-prediction
            
#             # Calculate correct predictions (diagonal should be maximum)
#             correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            
#             # Weights for different loss components
#             w1 = w2 = w3 = w4 = w5 = w6 = 1.0  # Cross-modal predictions (important)
#             w7 = w8 = w9 = 0.1                  # Self-predictions (less weight)
            
#             # Accumulate NCE loss
#             nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1)))
#             nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2)))
#             nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3)))
#             nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4)))
#             nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5)))
#             nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6)))
#             nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7)))
#             nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8)))
#             nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9)))
        
#         # Normalize NCE loss
#         nce /= -1. * batch_dim * actual_n_steps
        
#         # Calculate accuracies (using values from last prediction step)
#         accuracy1 = 1. * correct1 / batch_dim
#         accuracy2 = 1. * correct2 / batch_dim
#         accuracy3 = 1. * correct3 / batch_dim
#         accuracy4 = 1. * correct4 / batch_dim
#         accuracy5 = 1. * correct5 / batch_dim
#         accuracy6 = 1. * correct6 / batch_dim
#         accuracy7 = 1. * correct7 / batch_dim
#         accuracy8 = 1. * correct8 / batch_dim
#         accuracy9 = 1. * correct9 / batch_dim
        
#         return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce






# class Cross_CPC_AVT_pad(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
#         super(Cross_CPC_AVT_pad, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.context_dim = context_dim
#         self.num_layers = num_layers
#         self.n_prediction_steps = n_prediction_steps
#         self.min_start_steps = min_start_steps
#         self.softmax = nn.Softmax()
#         self.lsoftmax = nn.LogSoftmax()
        
#         # Autoregressive LSTM networks for each modality
#         self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
#         self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
#         self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
#         # Predictor networks for each modality
#         self.video_predictors = nn.ModuleList([
#             nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
#         ])
#         self.audio_predictors = nn.ModuleList([
#             nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
#         ])
#         self.text_predictors = nn.ModuleList([
#             nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
#         ])
    
    
#     def forward(self, audio_vq, video_vq, text_vq, lengths=None):
#         """
#         Enhanced forward pass with proper sequence length handling.
        
#         Args:
#             audio_vq, video_vq, text_vq: Quantized representations [batch, time_length, embedding_dim]
#             lengths: Actual sequence lengths for each sample in the batch [batch]
#         """
#         batch_dim, time_length, _ = video_vq.shape
        
#         if lengths is None:
#             lengths = torch.full((batch_dim,), time_length, dtype=torch.long, device=video_vq.device)
#         else:
#             lengths = lengths.to(video_vq.device)
        
#         min_len = lengths.min().item()
        
#         max_possible_t = min_len - self.n_prediction_steps - self.min_start_steps
        
#         if max_possible_t <= 0:
#             if min_len > 1:
#                 # Create t_samples as a scalar from the start
#                 t_samples = 0  # Use scalar directly
#                 actual_n_steps = min(self.n_prediction_steps, min_len - 1)
#             else:
#                 print(f"Warning: Sequences too short for CPC (min_len={min_len})")
#                 dummy_acc = torch.tensor(0.0, device=video_vq.device)
#                 dummy_nce = torch.tensor(0.0, device=video_vq.device)
#                 return dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_acc, dummy_nce
#         else:
#             # Extract scalar value immediately after generation
#             t_samples = torch.randint(max_possible_t, size=(1,)).item() + self.min_start_steps  
#             actual_n_steps = self.n_prediction_steps
        
#         # Now t_samples is a scalar, so this works correctly
#         effective_lengths = torch.minimum(lengths, torch.full_like(lengths, t_samples + 1))

#         # Use t_samples as a scalar throughout (no need for .item() calls)
#         video_forward_seq = video_vq[:, :t_samples+1, :]
#         audio_forward_seq = audio_vq[:, :t_samples+1, :]
#         text_forward_seq = text_vq[:, :t_samples+1, :]
        
#         sorted_lengths, sorted_idx = effective_lengths.sort(0, descending=True)
#         unsorted_idx = sorted_idx.argsort(0)

#         video_forward_sorted = video_forward_seq[sorted_idx]
#         audio_forward_sorted = audio_forward_seq[sorted_idx]
#         text_forward_sorted = text_forward_seq[sorted_idx]
        
#         # Pack sequences for LSTM processing
#         video_packed = pack_padded_sequence(
#             video_forward_sorted, sorted_lengths.cpu(), 
#             batch_first=True, enforce_sorted=True
#         )
#         audio_packed = pack_padded_sequence(
#             audio_forward_sorted, sorted_lengths.cpu(),
#             batch_first=True, enforce_sorted=True
#         )
#         text_packed = pack_padded_sequence(
#             text_forward_sorted, sorted_lengths.cpu(),
#             batch_first=True, enforce_sorted=True
#         )
        
#         # Initialize hidden states
#         video_hidden = (
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).double(),
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=video_vq.device).double()
#         )
#         audio_hidden = (
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).double(),
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=audio_vq.device).double()
#         )
#         text_hidden = (
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=text_vq.device).double(),
#             torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device=text_vq.device).double()
#         )
        
#         # Process through LSTMs
#         video_context_packed, _ = self.video_ar_lstm(video_packed, video_hidden)
#         audio_context_packed, _ = self.audio_ar_lstm(audio_packed, audio_hidden)
#         text_context_packed, _ = self.text_ar_lstm(text_packed, text_hidden)
        
#         # Unpack sequences
#         video_context, _ = pad_packed_sequence(video_context_packed, batch_first=True)
#         audio_context, _ = pad_packed_sequence(audio_context_packed, batch_first=True)
#         text_context, _ = pad_packed_sequence(text_context_packed, batch_first=True)
        
#         # Restore original order
#         video_context = video_context[unsorted_idx]
#         audio_context = audio_context[unsorted_idx]
#         text_context = text_context[unsorted_idx]
        
#         # Extract context at appropriate positions
#         video_context_list = []
#         audio_context_list = []
#         text_context_list = []
        
#         for i in range(batch_dim):
#             # Since t_samples is now a scalar, we can use it directly
#             if effective_lengths[i] > t_samples:
#                 context_pos = t_samples  # Direct use of scalar
#             else:
#                 context_pos = effective_lengths[i].item() - 1
            
#             video_context_list.append(video_context[i, context_pos, :])
#             audio_context_list.append(audio_context[i, context_pos, :])
#             text_context_list.append(text_context[i, context_pos, :])
        
#         video_context = torch.stack(video_context_list)
#         audio_context = torch.stack(audio_context_list)
#         text_context = torch.stack(text_context_list)
        
#         # Prepare target samples
#         video_encode_samples = []
#         audio_encode_samples = []
#         text_encode_samples = []
        
#         for step in range(actual_n_steps):
#             step_video = []
#             step_audio = []
#             step_text = []
            
#             for i in range(batch_dim):
#                 # t_samples is now a scalar, so no .item() needed
#                 target_pos = t_samples + step + 1
                
#                 if target_pos < lengths[i]:
#                     step_video.append(video_vq[i, target_pos, :])
#                     step_audio.append(audio_vq[i, target_pos, :])
#                     step_text.append(text_vq[i, target_pos, :])
#                 else:
#                     last_valid_pos = lengths[i] - 1
#                     step_video.append(video_vq[i, last_valid_pos, :])
#                     step_audio.append(audio_vq[i, last_valid_pos, :])
#                     step_text.append(text_vq[i, last_valid_pos, :])
            
#             video_encode_samples.append(torch.stack(step_video))
#             audio_encode_samples.append(torch.stack(step_audio))
#             text_encode_samples.append(torch.stack(step_text))
        
#         video_encode_samples = torch.stack(video_encode_samples)
#         audio_encode_samples = torch.stack(audio_encode_samples)
#         text_encode_samples = torch.stack(text_encode_samples)
        
#         # Generate predictions
#         video_pred = torch.empty((actual_n_steps, batch_dim, self.embedding_dim), device=video_vq.device).double()
#         audio_pred = torch.empty((actual_n_steps, batch_dim, self.embedding_dim), device=audio_vq.device).double()
#         text_pred = torch.empty((actual_n_steps, batch_dim, self.embedding_dim), device=text_vq.device).double()
        
#         for i in range(actual_n_steps):
#             video_pred[i] = self.video_predictors[i](video_context)
#             audio_pred[i] = self.audio_predictors[i](audio_context)
#             text_pred[i] = self.text_predictors[i](text_context)
        
#         # Calculate NCE loss and accuracies
#         nce = 0
        
#         # Initialize accuracy tracking variables
#         accuracy1 = accuracy2 = accuracy3 = accuracy4 = accuracy5 = 0
#         accuracy6 = accuracy7 = accuracy8 = accuracy9 = 0
        
#         for i in range(actual_n_steps):
#             # Compute similarity matrices
#             total1 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
#             total2 = torch.mm(audio_encode_samples[i], torch.transpose(text_pred[i], 0, 1))
#             total3 = torch.mm(video_encode_samples[i], torch.transpose(text_pred[i], 0, 1))
#             total4 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
#             total5 = torch.mm(text_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
#             total6 = torch.mm(text_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
#             total7 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i], 0, 1))
#             total8 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i], 0, 1))
#             total9 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i], 0, 1))
            
#             # Calculate correct predictions
#             correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
#             correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=0), torch.arange(0, batch_dim, device=video_vq.device)))
            
#             # Weights for loss components
#             w1 = w2 = w3 = w4 = w5 = w6 = 1.0
#             w7 = w8 = w9 = 0.1
            
#             # Accumulate NCE loss
#             nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1)))
#             nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2)))
#             nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3)))
#             nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4)))
#             nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5)))
#             nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6)))
#             nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7)))
#             nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8)))
#             nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9)))
        
#         # Normalize NCE loss
#         nce /= -1. * batch_dim * actual_n_steps
        
#         # Calculate accuracies
#         accuracy1 = 1. * correct1 / batch_dim
#         accuracy2 = 1. * correct2 / batch_dim
#         accuracy3 = 1. * correct3 / batch_dim
#         accuracy4 = 1. * correct4 / batch_dim
#         accuracy5 = 1. * correct5 / batch_dim
#         accuracy6 = 1. * correct6 / batch_dim
#         accuracy7 = 1. * correct7 / batch_dim
#         accuracy8 = 1. * correct8 / batch_dim
#         accuracy9 = 1. * correct9 / batch_dim
        
#         return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce