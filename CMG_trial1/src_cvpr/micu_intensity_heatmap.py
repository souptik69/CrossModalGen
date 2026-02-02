# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from collections import Counter
# import pickle

# sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')
# from model.main_model_2 import AVT_VQVAE_Encoder
# # from transformers import BertTokenizer, BertModel
# from bert_embedding import BertEmbedding

# # Configuration
# N_EMBEDDINGS = 400
# BATCH_SIZE = 128
# OUTPUT_DIR = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'
# CHECKPOINT = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/MICU/40k/checkpoint/MICU-step2400.pt'

# # Load BERT and text processing utilities
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # bert_model = BertModel.from_pretrained('bert-base-uncased')

# bert_embedding = BertEmbedding()
# with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl', 'rb') as fp:
#     id2idx = pickle.load(fp)

# # with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl', 'rb') as fp:
# #     id2idx = pickle.load(fp)

# # def collate_func_AVT(samples):
# #     bsz = len(samples)
# #     text_prompts = [sample['text_fea'] for sample in samples]
# #     query = []
# #     query_words = []
    
# #     for text in text_prompts:
# #         inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
# #         with torch.no_grad():
# #             outputs = bert_model(**inputs)
# #             embeddings = outputs.last_hidden_state.squeeze(0).numpy()
# #         token_ids = inputs.input_ids[0].tolist()
# #         tokens = tokenizer.convert_ids_to_tokens(token_ids)
# #         non_special_tokens = tokens[1:-1]
# #         non_special_embeddings = embeddings[1:-1]
# #         words = []
# #         words_emb = [] 
# #         for token, emb in zip(non_special_tokens, non_special_embeddings):
# #             idx = tokenizer.convert_tokens_to_ids(token)
# #             if idx in id2idx and idx != 0:
# #                 words_emb.append(emb)
# #                 words.append(id2idx[idx])
# #         query.append(np.asarray(words_emb) if len(words_emb) > 0 else np.zeros((1, 768)))
# #         query_words.append(words)

# #     query_len = [10 for _ in query]
# #     query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
# #     query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
# #     for i, sample in enumerate(query):
# #         keep = min(sample.shape[0], query1.shape[1])
# #         if keep > 0:
# #             query1[i, :keep] = sample[:keep]
# #             query_idx[i, :keep] = query_words[i][:keep]
    
# #     query_len = np.asarray(query_len)
# #     query = torch.from_numpy(query1).float()

# #     return {
# #         'query': query,
# #         'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
# #         'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
# #     }

# def collate_func_AVT(samples):
#         bsz = len(samples)
#         result = bert_embedding([sample['text_fea'] for sample in samples])
#         query = []
#         query_words = []
#         for a, b in result:
#             words = []
#             words_emb = []
#             for word, emb in zip(a, b):
#                 idx = bert_embedding.vocab.token_to_idx[word]
#                 if idx in id2idx and idx != 0:
#                     words_emb.append(emb)
#                     words.append(id2idx[idx])
#             query.append(np.asarray(words_emb))
#             query_words.append(words)

#         query_len = []
#         for i, sample in enumerate(query):
#             # query_len.append(min(len(sample), 10))#max_num_words:10
#             query_len.append(10)#max_num_words:10
#         query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
#         query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
#         for i, sample in enumerate(query):
#             keep = min(sample.shape[0], query1.shape[1])
#             """
#             There may be cases where the sample length is 0, 
#             for example if your text happens to not be seen before in this BERT model. 
#             If that happens, you can 
#             1) clean the text before it enters BERT, 
#             2) add an if statement here, 
#             3) discard idx and directly import all embeddings after.
#             """
#             query1[i, :keep] = sample[:keep]
#             query_idx[i, :keep] = query_words[i][:keep]
#         query_len = np.asarray(query_len)
#         query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
#         query_idx = torch.from_numpy(query_idx).long()
    
#         return {
#             'query': query,
#             'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
#             'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
#         }


# def extract_video_text_indices(encoder, text_lstm, video_feat, text_feat, device):
#     """Extract video and text indices from AVT encoder"""
#     encoder.eval()
#     text_lstm.eval()
    
#     with torch.no_grad():
#         video_feat = video_feat.to(device).double()
#         text_feat = text_feat.to(device).double()
        
#         # Process text through LSTM
#         batch_dim = text_feat.size()[0]
#         text_hidden = (
#             torch.zeros(4, batch_dim, 128).double().to(device),
#             torch.zeros(4, batch_dim, 128).double().to(device)
#         )
#         text_processed, _ = text_lstm(text_feat, text_hidden)
        
#         # Video processing
#         video_semantic_result, _ = encoder.video_semantic_encoder(video_feat)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
#         video_semantic_result = encoder.video_self_att(video_semantic_result)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
#         # Text processing
#         text_semantic_result = text_processed.transpose(0, 1).contiguous()
#         text_semantic_result = encoder.text_self_att(text_semantic_result)
#         text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        
#         B, T, D = text_semantic_result.size()
        
#         # Extract codebook embeddings - note the structure for AVT
#         video_embedding = encoder.Cross_quantizer.embedding[:, :D]
#         audio_embedding = encoder.Cross_quantizer.embedding[:, D:2*D]  # This is actually for audio
#         text_embedding = encoder.Cross_quantizer.embedding[:, 2*D:]     # This is for text
        
#         # Compute distances for video
#         v_flat = video_semantic_result.reshape(-1, D)
#         v_distances = torch.addmm(
#             torch.sum(video_embedding**2, dim=1) + torch.sum(v_flat**2, dim=1, keepdim=True),
#             v_flat, video_embedding.t(), alpha=-2.0, beta=1.0
#         )
        
#         # Compute distances for text
#         t_flat = text_semantic_result.reshape(-1, D)
#         t_distances = torch.addmm(
#             torch.sum(text_embedding**2, dim=1) + torch.sum(t_flat**2, dim=1, keepdim=True),
#             t_flat, text_embedding.t(), alpha=-2.0, beta=1.0
#         )
        
#         v_indices = torch.argmin(v_distances, dim=-1).reshape(B, T)
#         t_indices = torch.argmin(t_distances, dim=-1).reshape(B, T)
        
#         return v_indices, t_indices

# def analyze_and_plot():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Load MICU AVT model
#     print("Loading MICU AVT model...")
#     encoder = AVT_VQVAE_Encoder(
#         audio_dim=128, video_dim=512, text_dim=256,
#         audio_output_dim=256, video_output_dim=2048, text_output_dim=256,
#         n_embeddings=N_EMBEDDINGS, embedding_dim=256
#     )
#     text_lstm = torch.nn.LSTM(768, 128, num_layers=2, batch_first=True, bidirectional=True)
    
#     encoder.double().to(device).eval()
#     text_lstm.double().to(device).eval()
    
#     # Load checkpoint
#     checkpoint = torch.load(CHECKPOINT, map_location=device)
#     encoder.load_state_dict(checkpoint['Encoder_parameters'])
#     text_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
#     print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
#     # Load dataset
#     sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
#     from dataset.VGGSOUND_dataset import VGGSoundDataset_AVT
    
#     dataset = VGGSoundDataset_AVT(
#         meta_csv_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv',
#         audio_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip',
#         video_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip',
#         split='train'
#     )
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
#                            num_workers=8, collate_fn=collate_func_AVT)
    
#     # Collect usage statistics (video and text, but we'll call text "audio")
#     print("Analyzing codebook usage...")
#     video_counter = Counter()
#     text_counter = Counter()  # This will be plotted as "audio"
    
#     for batch_data in tqdm(dataloader, desc="Processing"):
#         video_feat = batch_data['video_fea']
#         text_feat = batch_data['query']
        
#         v_idx, t_idx = extract_video_text_indices(encoder, text_lstm, video_feat, 
#                                                    text_feat, device)
#         video_counter.update(v_idx.cpu().numpy().flatten().tolist())
#         text_counter.update(t_idx.cpu().numpy().flatten().tolist())
    
#     # Convert to arrays
#     video_counts = np.zeros(N_EMBEDDINGS)
#     text_counts = np.zeros(N_EMBEDDINGS)
    
#     for idx, count in video_counter.items():
#         video_counts[idx] = count
#     for idx, count in text_counter.items():
#         text_counts[idx] = count
    
#     print(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)")
#     print(f"Text (plotted as Audio): {np.sum(text_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(text_counts>0)/N_EMBEDDINGS:.1f}%)")
    
#     # Create heatmap (plotting text as "Audio")
#     print("Creating intensity heatmap...")
#     plt.figure(figsize=(14, 6))
    
#     # Stack: top row = video, bottom row = text (but label as "Audio")
#     data = np.vstack([video_counts, text_counts])
#     data_normalized = data / (data.max() + 1e-10)
    
#     ax = sns.heatmap(data_normalized, cmap='YlOrRd',
#                      cbar_kws={'label': 'Normalized Usage Intensity'},
#                      xticklabels=False, yticklabels=['Video', 'Audio'],  # Text labeled as "Audio"
#                      linewidths=0)
    
#     plt.xlabel('Codebook Vector Index', fontsize=15, fontweight='bold')
#     plt.ylabel('Modality', fontsize=15, fontweight='bold')
#     plt.title('MICU AV Model: Codebook Usage Intensity Across Vectors',  # Title says "AV Model"
#              fontsize=20, fontweight='bold')
    
#     # Add x-axis ticks
#     xticks_pos = np.linspace(0, N_EMBEDDINGS, 11)
#     plt.xticks(xticks_pos, [f'{int(x)}' for x in xticks_pos], fontsize=10)
#     plt.yticks(fontsize=15, rotation=0)
    
#     plt.tight_layout()
#     output_path = os.path.join(OUTPUT_DIR, 'MICU_AV_intensity_heatmap.png')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"✓ Heatmap saved: {output_path}")
    
#     # Save statistics
#     with open(os.path.join(OUTPUT_DIR, 'MICU_statistics.txt'), 'w') as f:
#         f.write("MICU AVT Model Analysis (Video + Text plotted as Video + Audio)\n")
#         f.write("="*70 + "\n\n")
#         f.write(f"Total codebook vectors: {N_EMBEDDINGS}\n\n")
#         f.write(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)\n")
#         f.write(f"Text (shown as Audio): {np.sum(text_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(text_counts>0)/N_EMBEDDINGS:.1f}%)\n")

# if __name__ == '__main__':
#     torch.manual_seed(43)
#     np.random.seed(43)
#     analyze_and_plot()






# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from collections import Counter
# import pickle

# # sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')
# from model.main_model_2 import AVT_VQVAE_Encoder
# # from transformers import BertTokenizer, BertModel
# from bert_embedding import BertEmbedding

# # Configuration
# N_EMBEDDINGS = 400
# BATCH_SIZE = 80
# OUTPUT_DIR = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'
# CHECKPOINT = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/MICU/40k/checkpoint/MICU-step2400.pt'

# # Load BERT and text processing utilities
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # bert_model = BertModel.from_pretrained('bert-base-uncased')

# # with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt_new.pkl', 'rb') as fp:
# #     id2idx = pickle.load(fp)

# bert_embedding = BertEmbedding()
# with open('/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl', 'rb') as fp:
#     id2idx = pickle.load(fp)


# def collate_func_AVT(samples):
#         bsz = len(samples)
#         result = bert_embedding([sample['text_fea'] for sample in samples])
#         query = []
#         query_words = []
#         for a, b in result:
#             words = []
#             words_emb = []
#             for word, emb in zip(a, b):
#                 idx = bert_embedding.vocab.token_to_idx[word]
#                 if idx in id2idx and idx != 0:
#                     words_emb.append(emb)
#                     words.append(id2idx[idx])
#             query.append(np.asarray(words_emb))
#             query_words.append(words)

#         query_len = []
#         for i, sample in enumerate(query):
#             # query_len.append(min(len(sample), 10))#max_num_words:10
#             query_len.append(10)#max_num_words:10
#         query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
#         query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
#         for i, sample in enumerate(query):
#             keep = min(sample.shape[0], query1.shape[1])
#             """
#             There may be cases where the sample length is 0, 
#             for example if your text happens to not be seen before in this BERT model. 
#             If that happens, you can 
#             1) clean the text before it enters BERT, 
#             2) add an if statement here, 
#             3) discard idx and directly import all embeddings after.
#             """
#             query1[i, :keep] = sample[:keep]
#             query_idx[i, :keep] = query_words[i][:keep]
#         query_len = np.asarray(query_len)
#         query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
#         query_idx = torch.from_numpy(query_idx).long()
    
#         return {
#             'query': query,
#             'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
#             'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
#         }

# # def collate_func_AVT(samples):
# #     bsz = len(samples)
# #     text_prompts = [sample['text_fea'] for sample in samples]
# #     query = []
# #     query_words = []
    
# #     for text in text_prompts:
# #         inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
# #         with torch.no_grad():
# #             outputs = bert_model(**inputs)
# #             embeddings = outputs.last_hidden_state.squeeze(0).numpy()
# #         token_ids = inputs.input_ids[0].tolist()
# #         tokens = tokenizer.convert_ids_to_tokens(token_ids)
# #         non_special_tokens = tokens[1:-1]
# #         non_special_embeddings = embeddings[1:-1]
# #         words = []
# #         words_emb = [] 
# #         for token, emb in zip(non_special_tokens, non_special_embeddings):
# #             idx = tokenizer.convert_tokens_to_ids(token)
# #             if idx in id2idx and idx != 0:
# #                 words_emb.append(emb)
# #                 words.append(id2idx[idx])
# #         query.append(np.asarray(words_emb) if len(words_emb) > 0 else np.zeros((1, 768)))
# #         query_words.append(words)

# #     query_len = [10 for _ in query]
# #     query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
# #     query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
    
# #     for i, sample in enumerate(query):
# #         keep = min(sample.shape[0], query1.shape[1])
# #         if keep > 0:
# #             query1[i, :keep] = sample[:keep]
# #             query_idx[i, :keep] = query_words[i][:keep]
    
# #     query_len = np.asarray(query_len)
# #     query = torch.from_numpy(query1).float()

# #     return {
# #         'query': query,
# #         'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
# #         'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
# #     }

# def extract_video_text_indices(encoder, text_lstm, video_feat, text_feat, device):
#     """Extract video and text indices from AVT encoder
    
#     CRITICAL: In MICU AVT model, all modalities share the SAME codebook!
#     The codebook is [n_embeddings, embedding_dim] = [400, 256]
#     NOT split into segments for different modalities.
#     """
#     encoder.eval()
#     text_lstm.eval()
    
#     with torch.no_grad():
#         video_feat = video_feat.to(device).double()
#         text_feat = text_feat.to(device).double()
        
#         # Process text through LSTM
#         batch_dim = text_feat.size()[0]
#         text_hidden = (
#             torch.zeros(4, batch_dim, 128).double().to(device),
#             torch.zeros(4, batch_dim, 128).double().to(device)
#         )
#         text_processed, _ = text_lstm(text_feat, text_hidden)
        
#         # Video processing
#         video_semantic_result = encoder.video_semantic_encoder(video_feat)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
#         video_semantic_result = encoder.video_self_att(video_semantic_result)
#         video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
#         # Text processing
#         text_semantic_result = text_processed.transpose(0, 1).contiguous()
#         text_semantic_result = encoder.text_self_att(text_semantic_result)
#         text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        
#         B, T, D = text_semantic_result.size()  # D should be 256
        
#         # SHARED CODEBOOK - all modalities use the same [400, 256] codebook
#         embedding = encoder.Cross_quantizer.embedding  # [400, 256]
        
#         # Compute distances for video (exactly as in Cross_VQEmbeddingEMA_AVT.Video_vq_embedding)
#         v_flat = video_semantic_result.reshape(-1, D)  # [BxT, 256]
#         v_distances = torch.addmm(
#             torch.sum(embedding**2, dim=1) + torch.sum(v_flat**2, dim=1, keepdim=True),
#             v_flat, embedding.t(), alpha=-2.0, beta=1.0
#         )  # [BxT, 400]
        
#         # Compute distances for text (exactly as in Cross_VQEmbeddingEMA_AVT.Text_vq_embedding)
#         t_flat = text_semantic_result.reshape(-1, D)  # [BxT, 256]
#         t_distances = torch.addmm(
#             torch.sum(embedding**2, dim=1) + torch.sum(t_flat**2, dim=1, keepdim=True),
#             t_flat, embedding.t(), alpha=-2.0, beta=1.0
#         )  # [BxT, 400]
        
#         v_indices = torch.argmin(v_distances, dim=-1).reshape(B, T)  # [B, T]
#         t_indices = torch.argmin(t_distances, dim=-1).reshape(B, T)  # [B, T]
        
#         return v_indices, t_indices

# def analyze_and_plot():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Load MICU AVT model
#     print("Loading MICU AVT model...")
#     print("IMPORTANT: MICU uses a SHARED codebook for all modalities!")
#     print("  - Codebook shape: [400, 256]")
#     print("  - All modalities (audio, video, text) compete for the same 400 vectors")
#     print("  - This is different from split-codebook architectures")
#     print()
    
#     encoder = AVT_VQVAE_Encoder(
#         audio_dim=128, video_dim=512, text_dim=256,
#         audio_output_dim=256, video_output_dim=2048, text_output_dim=256,
#         n_embeddings=N_EMBEDDINGS, embedding_dim=256
#     )
#     text_lstm = torch.nn.LSTM(768, 128, num_layers=2, batch_first=True, bidirectional=True)
    
#     encoder.double().to(device).eval()
#     text_lstm.double().to(device).eval()
    
#     # Load checkpoint
#     checkpoint = torch.load(CHECKPOINT, map_location=device)
#     encoder.load_state_dict(checkpoint['Encoder_parameters'])
#     text_lstm.load_state_dict(checkpoint['Text_ar_lstm_parameters'])
#     print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
#     # Load dataset
#     # sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
#     from dataset.VGGSOUND_dataset import VGGSoundDataset_AVT
    
#     dataset = VGGSoundDataset_AVT(
#         meta_csv_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv',
#         audio_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip',
#         video_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip',
#         split='train'
#     )
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
#                            num_workers=8, collate_fn=collate_func_AVT)
    
#     # Collect usage statistics (video and text, but we'll call text "audio")
#     print("Analyzing codebook usage...")
#     video_counter = Counter()
#     text_counter = Counter()  # This will be plotted as "audio"
    
#     for batch_data in tqdm(dataloader, desc="Processing"):
#         video_feat = batch_data['video_fea']
#         text_feat = batch_data['query']
        
#         v_idx, t_idx = extract_video_text_indices(encoder, text_lstm, video_feat, 
#                                                    text_feat, device)
#         video_counter.update(v_idx.cpu().numpy().flatten().tolist())
#         text_counter.update(t_idx.cpu().numpy().flatten().tolist())
    
#     # Convert to arrays
#     video_counts = np.zeros(N_EMBEDDINGS)
#     text_counts = np.zeros(N_EMBEDDINGS)
    
#     for idx, count in video_counter.items():
#         video_counts[idx] = count
#     for idx, count in text_counter.items():
#         text_counts[idx] = count
    
#     print(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)")
#     print(f"Text (plotted as Audio): {np.sum(text_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(text_counts>0)/N_EMBEDDINGS:.1f}%)")
    
#     # Create heatmap (plotting text as "Audio")
#     print("Creating intensity heatmap...")
#     plt.figure(figsize=(14, 6))
    
#     # Stack: top row = video, bottom row = text (but label as "Audio")
#     data = np.vstack([video_counts, text_counts])
#     data_normalized = data / (data.max() + 1e-10)
    
#     ax = sns.heatmap(data_normalized, cmap='YlOrRd',
#                      cbar_kws={'label': 'Normalized Usage Intensity'},
#                      xticklabels=False, yticklabels=['Video', 'Audio'],  # Text labeled as "Audio"
#                      linewidths=0)
    
#     plt.xlabel('Codebook Vector Index', fontsize=15, fontweight='bold')
#     plt.ylabel('Modality', fontsize=15, fontweight='bold')
#     plt.title('MICU AV Model: Codebook Usage Intensity Across Vectors',  # Title says "AV Model"
#              fontsize=20, fontweight='bold')
    
#     # Add x-axis ticks
#     xticks_pos = np.linspace(0, N_EMBEDDINGS, 11)
#     plt.xticks(xticks_pos, [f'{int(x)}' for x in xticks_pos], fontsize=10)
#     plt.yticks(fontsize=15, rotation=0)
    
#     plt.tight_layout()
#     output_path = os.path.join(OUTPUT_DIR, 'MICU_AV_intensity_heatmap.png')
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"✓ Heatmap saved: {output_path}")
    
#     # Save statistics
#     with open(os.path.join(OUTPUT_DIR, 'MICU_statistics.txt'), 'w') as f:
#         f.write("MICU AVT Model Analysis (Video + Text plotted as Video + Audio)\n")
#         f.write("="*70 + "\n\n")
#         f.write(f"Total codebook vectors: {N_EMBEDDINGS}\n\n")
#         f.write(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)\n")
#         f.write(f"Text (shown as Audio): {np.sum(text_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(text_counts>0)/N_EMBEDDINGS:.1f}%)\n")

# if __name__ == '__main__':
#     torch.manual_seed(43)
#     np.random.seed(43)
#     analyze_and_plot()


import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr')
from model.main_model_2 import AV_VQVAE_Encoder

# Configuration
N_EMBEDDINGS = 400
BATCH_SIZE = 300
OUTPUT_DIR = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization'
CHECKPOINT = '/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AV/MICU/40k/checkpoint/MICU-step2400.pt'

def collate_func_AV(samples):
    """Simple collate function for audio-video only"""
    return {
        'audio_fea': torch.from_numpy(np.asarray([s['audio_fea'] for s in samples])).float(),
        'video_fea': torch.from_numpy(np.asarray([s['video_fea'] for s in samples])).float()
    }

def extract_audio_video_indices(encoder, audio_feat, video_feat, device):
    """Extract ACTUAL audio and video indices from AVT encoder
    
    This extracts the real audio modality (not text).
    Both audio and video use the SHARED codebook [400, 256].
    """
    encoder.eval()
    
    with torch.no_grad():
        audio_feat = audio_feat.to(device).double()
        video_feat = video_feat.to(device).double()
        
        # Video processing (same as before)
        video_semantic_result = encoder.video_semantic_encoder(video_feat)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        video_semantic_result = encoder.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()
        
        # Audio processing (actual audio modality)
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = encoder.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        
        B, T, D = audio_semantic_result.size()  # D should be 256
        
        # SHARED CODEBOOK - all modalities use the same [400, 256] codebook
        embedding = encoder.Cross_quantizer.embedding  # [400, 256]
        
        # Compute distances for video
        v_flat = video_semantic_result.reshape(-1, D)  # [BxT, 256]
        v_distances = torch.addmm(
            torch.sum(embedding**2, dim=1) + torch.sum(v_flat**2, dim=1, keepdim=True),
            v_flat, embedding.t(), alpha=-2.0, beta=1.0
        )  # [BxT, 400]
        
        # Compute distances for audio
        a_flat = audio_semantic_result.reshape(-1, D)  # [BxT, 256]
        a_distances = torch.addmm(
            torch.sum(embedding**2, dim=1) + torch.sum(a_flat**2, dim=1, keepdim=True),
            a_flat, embedding.t(), alpha=-2.0, beta=1.0
        )  # [BxT, 400]
        
        v_indices = torch.argmin(v_distances, dim=-1).reshape(B, T)  # [B, T]
        a_indices = torch.argmin(a_distances, dim=-1).reshape(B, T)  # [B, T]
        
        return v_indices, a_indices

def analyze_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load MICU AVT model
    print("Loading MICU AVT model...")
    print("Extracting ACTUAL Audio + Video modalities")
    print("IMPORTANT: MICU uses a SHARED codebook for all modalities!")
    print("  - Codebook shape: [400, 256]")
    print("  - All modalities (audio, video, text) compete for the same 400 vectors")
    print()
    
    encoder = AV_VQVAE_Encoder(
        audio_dim=128, video_dim=512,
        audio_output_dim=256, video_output_dim=2048,
        n_embeddings=N_EMBEDDINGS, embedding_dim=256
    )
    
    encoder.double().to(device).eval()
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load dataset
    sys.path.append('/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/dataset')
    from dataset.VGGSOUND_dataset import VGGSoundDataset_AV_1
    
    dataset = VGGSoundDataset_AV_1(
        meta_csv_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsound-avel40k.csv',
        audio_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/audio/zip',
        video_fea_base_path='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/feature/video/zip',
        split='train'
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=8, collate_fn=collate_func_AV)
    
    # Collect usage statistics for ACTUAL audio and video
    print("Analyzing codebook usage for Audio and Video modalities...")
    video_counter = Counter()
    audio_counter = Counter()
    
    for batch_data in tqdm(dataloader, desc="Processing"):
        video_feat = batch_data['video_fea']
        audio_feat = batch_data['audio_fea']
        
        v_idx, a_idx = extract_audio_video_indices(encoder, audio_feat, video_feat, device)
        video_counter.update(v_idx.cpu().numpy().flatten().tolist())
        audio_counter.update(a_idx.cpu().numpy().flatten().tolist())
    
    # Convert to arrays
    video_counts = np.zeros(N_EMBEDDINGS)
    audio_counts = np.zeros(N_EMBEDDINGS)
    
    for idx, count in video_counter.items():
        video_counts[idx] = count
    for idx, count in audio_counter.items():
        audio_counts[idx] = count
    
    print(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)")
    print(f"Audio: {np.sum(audio_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(audio_counts>0)/N_EMBEDDINGS:.1f}%)")
    
    # Create heatmap
    print("Creating intensity heatmap...")
    plt.figure(figsize=(14, 6))
    
    # Stack: top row = video, bottom row = audio
    data = np.vstack([video_counts, audio_counts])
    data_normalized = data / (data.max() + 1e-10)
    
    ax = sns.heatmap(data_normalized, cmap='YlOrRd',
                     cbar_kws={'label': 'Normalized Usage Intensity'},
                     xticklabels=False, yticklabels=['Video', 'Audio'],
                     linewidths=0)
    
    plt.xlabel('Codebook Vector Index', fontsize=15, fontweight='bold')
    plt.ylabel('Modality', fontsize=15, fontweight='bold')
    plt.title('MICU AV Model: Codebook Usage Intensity Across Vectors',
             fontsize=20, fontweight='bold')
    
    # Add x-axis ticks
    xticks_pos = np.linspace(0, N_EMBEDDINGS, 11)
    plt.xticks(xticks_pos, [f'{int(x)}' for x in xticks_pos], fontsize=10)
    plt.yticks(fontsize=15, rotation=0)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'MICU_AV_actual_audio_video_heatmap_1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Heatmap saved: {output_path}")
    
    # Save statistics
    with open(os.path.join(OUTPUT_DIR, 'MICU_actual_AV_statistics.txt'), 'w') as f:
        f.write("MICU AVT Model Analysis (ACTUAL Audio + Video)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total codebook vectors: {N_EMBEDDINGS}\n\n")
        f.write(f"Video: {np.sum(video_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(video_counts>0)/N_EMBEDDINGS:.1f}%)\n")
        f.write(f"Audio (actual modality): {np.sum(audio_counts > 0)}/{N_EMBEDDINGS} vectors used ({100*np.sum(audio_counts>0)/N_EMBEDDINGS:.1f}%)\n")
        f.write(f"\nNote: This uses the ACTUAL audio modality, not text.\n")

if __name__ == '__main__':
    torch.manual_seed(43)
    np.random.seed(43)
    analyze_and_plot()