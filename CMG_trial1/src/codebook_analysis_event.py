# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict, Counter
# import os
# import sys
# from datetime import datetime
# import pandas as pd
# import h5py
# import zipfile
# from io import BytesIO
# import pickle

# # Add your project paths
# sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/dataset")
# sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src")

# from model.main_model_novel import AV_VQVAE_Encoder, AVT_VQVAE_Encoder

# def print_analysis_progress(message):
#     """Helper function for logging analysis progress"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# def generate_ave_category_list():
#     """Generate AVE category list - 28 classes"""
#     # You'll need to provide the actual AVE categories
#     return [f"event_{i}" for i in range(28)]  # Placeholder

# def generate_avvp_category_list():
#     """Generate AVVP category list from file"""
#     file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_Categories.txt'
#     category_list = []
#     try:
#         with open(file_path, 'r') as fr:
#             for line in fr.readlines():
#                 category_list.append(line.strip())
#         category_list.append("background")  # Add background class
#     except FileNotFoundError:
#         print_analysis_progress(f"Warning: Category file not found, using placeholder categories")
#         category_list = [f"avvp_event_{i}" for i in range(25)] + ["background"]
#     return category_list

# class AVEDatasetAnalysis:
#     def __init__(self, data_root, split='test'):
#         self.split = split
#         self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
#         self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
#         self.labels_path = os.path.join(data_root, 'labels.h5')
#         self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
#         self.categories = generate_ave_category_list()
#         self.h5_isOpen = False

#     def __getitem__(self, index):
#         if not self.h5_isOpen:
#             self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
#             self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
#             self.labels = h5py.File(self.labels_path, 'r')['avadataset']
#             self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
#             self.h5_isOpen = True
        
#         sample_index = self.sample_order[index]
#         visual_feat = self.visual_feature[sample_index]
#         audio_feat = self.audio_feature[sample_index]
#         label = self.labels[sample_index]
        
#         return {
#             'audio_fea': audio_feat,
#             'video_fea': visual_feat, 
#             'labels': label,
#             'video_ids': f"ave_sample_{sample_index}"
#         }

#     def __len__(self):
#         f = h5py.File(self.sample_order_path, 'r')
#         sample_num = len(f['order'])
#         f.close()
#         return sample_num

# class AVVPDatasetAnalysis:
#     def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, split='test'):
#         self.audio_fea_base_path = audio_fea_base_path
#         self.video_fea_base_path = video_fea_base_path
#         self.split_df = pd.read_csv(meta_csv_path, sep='\t')
#         self.categories = generate_avvp_category_list()
        
#     def _load_fea(self, fea_base_path, video_id):
#         fea_path = os.path.join(fea_base_path, f"{video_id}.zip")
#         try:
#             with zipfile.ZipFile(fea_path, mode='r') as zfile:
#                 for name in zfile.namelist():
#                     if '.pkl' not in name:
#                         continue
#                     with zfile.open(name, mode='r') as fea_file:
#                         content = BytesIO(fea_file.read())
#                         fea = pickle.load(content)
#             return fea
#         except Exception as e:
#             print_analysis_progress(f"Error loading feature {fea_path}: {e}")
#             return None
    
#     def _obtain_avel_label(self, onsets, offsets, categorys):
#         T, category_num = 10, len(self.categories) - 1  # Exclude background for now
#         label = np.zeros((T, category_num + 1))  # Add background
#         label[:, -1] = np.ones(T)  # Background default
        
#         iter_num = len(categorys)
#         for i in range(iter_num):
#             avc_label = np.zeros(T)
#             avc_label[onsets[i]:offsets[i]] = 1
#             if categorys[i] in self.categories[:-1]:  # Exclude background
#                 class_id = self.categories.index(categorys[i])
#                 bg_flag = 1 - avc_label
#                 for j in range(10):
#                     label[j, class_id] = int(label[j, class_id]) | int(avc_label[j])
#                 for j in range(10):
#                     label[j, -1] = int(label[j, -1]) & int(bg_flag[j])
#         return label

#     def __getitem__(self, index):
#         one_video_df = self.split_df.iloc[index]
#         categorys, video_id = one_video_df['event_labels'].split(','), one_video_df['filename']
#         onsets, offsets = one_video_df['onset'].split(','), one_video_df['offset'].split(',')
#         onsets = list(map(int, onsets))
#         offsets = list(map(int, offsets))
        
#         audio_fea = self._load_fea(self.audio_fea_base_path, video_id[:11])
#         video_fea = self._load_fea(self.video_fea_base_path, video_id[:11])
        
#         if audio_fea is None or video_fea is None:
#             return None
            
#         # Handle audio feature padding/truncation
#         if audio_fea.shape[0] < 10:
#             cur_t = audio_fea.shape[0]
#             add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
#             audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
#         elif audio_fea.shape[0] > 10:
#             audio_fea = audio_fea[:10, :]
            
#         avel_label = self._obtain_avel_label(onsets, offsets, categorys)
        
#         return {
#             'audio_fea': audio_fea,
#             'video_fea': video_fea,
#             'labels': avel_label,
#             'video_ids': video_id,
#             'event_categories': categorys
#         }

#     def __len__(self):
#         return len(self.split_df)

# def stratify_samples_by_events_ave(dataset, samples_per_class=2):
#     """
#     Sample data points from different AVE event classes.
#     """
#     print_analysis_progress(f"Stratifying AVE samples by event classes...")
    
#     # Collect samples by event class
#     class_samples = defaultdict(list)
#     categories = generate_ave_category_list()
    
#     for i in range(len(dataset)):
#         try:
#             sample = dataset[i]
#             label = sample['labels']
            
#             # AVE has single-label classification
#             # Assume label is the class index or one-hot encoded
#             if isinstance(label, np.ndarray):
#                 if len(label.shape) == 1 and len(label) > 1:  # One-hot
#                     class_id = np.argmax(label)
#                 else:  # Single value
#                     class_id = int(label.item()) if hasattr(label, 'item') else int(label)
#             else:
#                 class_id = int(label)
            
#             if 0 <= class_id < len(categories):
#                 sample['class_id'] = class_id
#                 sample['class_name'] = categories[class_id]
#                 class_samples[class_id].append(sample)
                
#         except Exception as e:
#             print_analysis_progress(f"Error processing sample {i}: {e}")
#             continue
    
#     # Sample from each class
#     sampled_data = {
#         'samples': {},
#         'class_info': {
#             'categories': categories,
#             'dataset_name': 'AVE'
#         }
#     }
    
#     for class_id, samples in class_samples.items():
#         if len(samples) >= samples_per_class:
#             selected_samples = np.random.choice(samples, samples_per_class, replace=False)
#         else:
#             selected_samples = samples
#             print_analysis_progress(f"Warning: Only {len(samples)} samples for class {class_id}")
        
#         sampled_data['samples'][class_id] = list(selected_samples)
#         print_analysis_progress(f"Class {class_id} ({categories[class_id]}): {len(selected_samples)} samples")
    
#     return sampled_data

# def stratify_samples_by_events_avvp(dataset, samples_per_class=2, focus_on_primary=True):
#     """
#     Sample data points from AVVP dataset by event classes.
#     Since AVVP is multi-label, we can focus on primary events or sample by individual classes.
#     """
#     print_analysis_progress(f"Stratifying AVVP samples by event classes...")
    
#     class_samples = defaultdict(list)
#     categories = generate_avvp_category_list()
    
#     for i in range(len(dataset)):
#         try:
#             sample = dataset[i]
#             if sample is None:
#                 continue
                
#             labels = sample['labels']  # Shape: [10, 26]
#             event_categories = sample.get('event_categories', [])
            
#             if focus_on_primary and event_categories:
#                 # Use the first mentioned event as primary
#                 primary_event = event_categories[0]
#                 if primary_event in categories[:-1]:  # Exclude background
#                     class_id = categories.index(primary_event)
#                     sample['class_id'] = class_id
#                     sample['class_name'] = primary_event
#                     sample['is_primary'] = True
#                     class_samples[class_id].append(sample)
#             else:
#                 # Sample based on all active classes in the temporal sequence
#                 active_classes = []
#                 for t in range(labels.shape[0]):  # For each timestep
#                     for c in range(labels.shape[1] - 1):  # Exclude background
#                         if labels[t, c] > 0:
#                             active_classes.append(c)
                
#                 # Add sample to each active class
#                 for class_id in set(active_classes):
#                     sample_copy = sample.copy()
#                     sample_copy['class_id'] = class_id
#                     sample_copy['class_name'] = categories[class_id]
#                     sample_copy['is_primary'] = False
#                     class_samples[class_id].append(sample_copy)
                    
#         except Exception as e:
#             print_analysis_progress(f"Error processing AVVP sample {i}: {e}")
#             continue
    
#     # Sample from each class
#     sampled_data = {
#         'samples': {},
#         'class_info': {
#             'categories': categories,
#             'dataset_name': 'AVVP'
#         }
#     }
    
#     for class_id, samples in class_samples.items():
#         if len(samples) >= samples_per_class:
#             selected_samples = np.random.choice(samples, samples_per_class, replace=False)
#         else:
#             selected_samples = samples
#             print_analysis_progress(f"Warning: Only {len(samples)} samples for class {class_id}")
        
#         sampled_data['samples'][class_id] = list(selected_samples)
#         print_analysis_progress(f"Class {class_id} ({categories[class_id]}): {len(selected_samples)} samples")
    
#     return sampled_data

# def load_pretrained_encoder(checkpoint_path, model_config, model_type='AVT'):
#     """
#     Load a pretrained encoder from checkpoint.
#     """
#     print_analysis_progress(f"Loading pretrained {model_type} encoder from: {checkpoint_path}")
    
#     if model_type == 'AVT':
#         encoder = AVT_VQVAE_Encoder(
#             audio_dim=model_config['audio_dim'],
#             video_dim=model_config['video_dim'], 
#             text_dim=model_config['text_dim'],
#             video_output_dim=model_config['video_output_dim'],
#             n_embeddings=model_config['n_embeddings'],
#             embedding_dim=model_config['embedding_dim']
#         )
#     else:  # AV
#         encoder = AV_VQVAE_Encoder(
#             audio_dim=model_config['audio_dim'],
#             video_dim=model_config['video_dim'],
#             video_output_dim=model_config['video_output_dim'], 
#             n_embeddings=model_config['n_embeddings'],
#             embedding_dim=model_config['embedding_dim']
#         )
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     encoder.load_state_dict(checkpoint['Encoder_parameters'])
#     encoder = encoder.double().to(device)
#     encoder.eval()
    
#     print_analysis_progress(f"{model_type} Encoder loaded successfully on {device}")
#     return encoder

# def analyze_sample_quantization_events(encoder, sample, class_id, class_name, dataset_name, model_type='AVT'):
#     """
#     Analyze how a single sample gets quantized by the VQ-VAE encoder for event detection.
#     """
#     print_analysis_progress(f"\n{'='*100}")
#     print_analysis_progress(f"EVENT QUANTIZATION ANALYSIS - {dataset_name} Sample")
#     print_analysis_progress(f"Event Class: {class_id} ({class_name})")
#     print_analysis_progress(f"Video ID: {sample.get('video_ids', 'unknown')}")
#     print_analysis_progress(f"{'='*100}")
    
#     device = next(encoder.parameters()).device
    
#     # Prepare input tensors
#     audio_feat = torch.from_numpy(sample['audio_fea']).unsqueeze(0).double().to(device)
#     video_feat = torch.from_numpy(sample['video_fea']).unsqueeze(0).double().to(device)
    
#     with torch.no_grad():
#         if model_type == 'AVT':
#             # For AVT model, we need text features - use zero padding or skip text analysis
#             text_feat = torch.zeros(1, audio_feat.shape[1], 256).double().to(device)  # Dummy text
            
#             # Get quantized representations using individual VQ encoders
#             out_vq_audio, audio_vq = encoder.Audio_VQ_Encoder(audio_feat)
#             out_vq_video, video_vq = encoder.Video_VQ_Encoder(video_feat)
#             # out_vq_text, text_vq = encoder.Text_VQ_Encoder(text_feat)  # Skip for event analysis
            
#         else:  # AV model
#             out_vq_audio, audio_vq = encoder.Audio_VQ_Encoder(audio_feat)
#             out_vq_video, video_vq = encoder.Video_VQ_Encoder(video_feat)
    
#     # Remove batch dimension for analysis
#     audio_quantized = audio_vq.squeeze(0).cpu().numpy()
#     video_quantized = video_vq.squeeze(0).cpu().numpy()
#     out_vq_audio_full = out_vq_audio.squeeze(0).cpu().numpy()
#     out_vq_video_full = out_vq_video.squeeze(0).cpu().numpy()
    
#     print_analysis_progress("Quantized representations extracted successfully!")
    
#     # Analyze codebook usage patterns
#     analysis_results = {
#         'sample_info': {
#             'video_id': sample.get('video_ids', 'unknown'),
#             'class_id': class_id,
#             'class_name': class_name,
#             'dataset_name': dataset_name
#         },
#         'quantized_representations': {
#             'audio': audio_quantized,
#             'video': video_quantized
#         },
#         'full_quantized_vectors': {
#             'audio': out_vq_audio_full,
#             'video': out_vq_video_full
#         },
#         'quantization_indices': {},
#         'codebook_matches': {}
#     }
    
#     # Get the quantizer to analyze codebook usage
#     quantizer = encoder.Cross_quantizer
#     codebook_embedding = quantizer.embedding.detach().cpu().numpy()
    
#     if model_type == 'AVT':
#         # For AVT: codebook is split into [video, audio, text] segments
#         D = audio_quantized.shape[-1]  # Should be 256
#         video_embedding = codebook_embedding[:, :D]
#         audio_embedding = codebook_embedding[:, D:2*D]
#     else:
#         # For AV: codebook is split into [video, audio] segments  
#         D = audio_quantized.shape[-1]
#         video_embedding = codebook_embedding[:, :D]
#         audio_embedding = codebook_embedding[:, D:]
    
#     # Find quantization indices for each modality
#     modalities = [
#         ('audio', audio_quantized, audio_embedding),
#         ('video', video_quantized, video_embedding)
#     ]
    
#     for mod_name, quantized, embedding_segment in modalities:
#         print_analysis_progress(f"\n{mod_name.upper()} Quantization Analysis:")
        
#         seq_len = quantized.shape[0]
#         quantization_indices = []
        
#         for t in range(seq_len):
#             # Find closest codebook vector
#             distances = np.sum((embedding_segment - quantized[t])**2, axis=1)
#             closest_idx = np.argmin(distances)
#             quantization_indices.append((t, closest_idx, distances[closest_idx]))
            
#             if t < 3 or t == seq_len - 1:  # Show first few and last timestep
#                 print_analysis_progress(f"  Timestep {t}: Codebook vector {closest_idx} (distance: {distances[closest_idx]:.6f})")
        
#         analysis_results['quantization_indices'][mod_name] = quantization_indices
        
#         # Analyze usage patterns
#         used_indices = [idx for _, idx, _ in quantization_indices]
#         unique_indices = set(used_indices)
#         usage_counter = Counter(used_indices)
        
#         print_analysis_progress(f"  Total timesteps: {len(quantization_indices)}")
#         print_analysis_progress(f"  Unique vectors used: {len(unique_indices)}")
#         print_analysis_progress(f"  Most frequent vectors: {usage_counter.most_common(3)}")
    
#     return analysis_results

# def create_event_class_visualizations(sampled_datasets, checkpoint_path, model_config, output_dir, model_type='AVT'):
#     """
#     Create visualizations showing how codebook indices are distributed across event classes.
#     """
#     print_analysis_progress(f"\n{'='*120}")
#     print_analysis_progress("CREATING EVENT-CODEBOOK VISUALIZATIONS")
#     print_analysis_progress(f"{'='*120}")
    
#     # Create visualization subdirectory
#     viz_dir = os.path.join(output_dir, 'visualizations')
#     os.makedirs(viz_dir, exist_ok=True)
    
#     # Load pretrained encoder
#     encoder = load_pretrained_encoder(checkpoint_path, model_config, model_type)
    
#     # Set up plot style
#     plt.style.use('default')
#     sns.set_palette("husl")
    
#     # Collect all quantization data
#     all_data = {}
    
#     for dataset_name, dataset_samples in sampled_datasets.items():
#         print_analysis_progress(f"Processing {dataset_name} for visualization...")
        
#         class_info = dataset_samples['class_info']
#         samples = dataset_samples['samples']
        
#         dataset_data = {
#             'codebook_indices': {'audio': {}, 'video': {}},
#             'class_info': class_info
#         }
        
#         for class_id, class_samples in samples.items():
#             # Initialize storage for this class
#             for modality in ['audio', 'video']:
#                 dataset_data['codebook_indices'][modality][class_id] = []
            
#             for sample in class_samples:
#                 try:
#                     # Analyze quantization for this sample
#                     analysis = analyze_sample_quantization_events(
#                         encoder, sample, class_id, 
#                         class_info['categories'][class_id], dataset_name, model_type
#                     )
                    
#                     # Extract indices for each modality
#                     for modality in ['audio', 'video']:
#                         indices = [idx for t, idx, dist in analysis['quantization_indices'][modality]]
#                         dataset_data['codebook_indices'][modality][class_id].extend(indices)
                        
#                 except Exception as e:
#                     print_analysis_progress(f"Error analyzing sample: {e}")
#                     continue
        
#         all_data[dataset_name] = dataset_data
    
#     # Create visualizations
#     print_analysis_progress("Creating event class visualizations...")
    
#     # 1. Event Class Codebook Heatmaps
#     create_event_class_heatmaps(all_data, viz_dir)
    
#     # 2. Top Codebook Indices Bar Plots by Event Class
#     create_event_top_indices_barplots(all_data, viz_dir)
    
#     # 3. Event Class Diversity Analysis
#     create_event_diversity_analysis(all_data, viz_dir)
    
#     print_analysis_progress(f"Event visualizations saved to: {viz_dir}")

# def create_event_class_heatmaps(all_data, viz_dir):
#     """Create heatmaps showing codebook usage across event classes."""
#     print_analysis_progress("Creating event class codebook heatmaps...")
    
#     for dataset_name, dataset_data in all_data.items():
#         class_info = dataset_data['class_info']
#         categories = class_info['categories']
        
#         fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#         fig.suptitle(f'{dataset_name} - Codebook Usage by Event Class', fontsize=16, fontweight='bold')
        
#         modalities = ['audio', 'video']
        
#         for i, modality in enumerate(modalities):
#             ax = axes[i]
            
#             # Collect usage data for heatmap
#             codebook_usage = defaultdict(lambda: defaultdict(int))
#             all_indices = set()
            
#             for class_id, indices in dataset_data['codebook_indices'][modality].items():
#                 counter = Counter(indices)
#                 for idx, count in counter.items():
#                     codebook_usage[class_id][idx] = count
#                     all_indices.add(idx)
            
#             if all_indices:
#                 # Create matrix for heatmap
#                 sorted_indices = sorted(all_indices)
#                 sorted_classes = sorted(codebook_usage.keys())
#                 matrix = np.zeros((len(sorted_classes), len(sorted_indices)))
                
#                 for class_idx, class_id in enumerate(sorted_classes):
#                     for idx_pos, codebook_idx in enumerate(sorted_indices):
#                         matrix[class_idx, idx_pos] = codebook_usage[class_id][codebook_idx]
                
#                 # Plot heatmap
#                 sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=True,
#                            xticklabels=[f'{idx}' for idx in sorted_indices[::max(1, len(sorted_indices)//10)]], 
#                            yticklabels=[f'{categories[i] if i < len(categories) else f"Class_{i}"}' for i in sorted_classes])
                
#                 ax.set_title(f'{modality.title()} Modality')
#                 ax.set_xlabel('Codebook Index')
#                 ax.set_ylabel('Event Class')
#             else:
#                 ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
#                 ax.set_title(f'{modality.title()} Modality')
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(viz_dir, f'{dataset_name}_event_codebook_heatmaps.png'), 
#                    dpi=300, bbox_inches='tight')
#         plt.close()

# def create_event_top_indices_barplots(all_data, viz_dir):
#     """Create bar plots showing top codebook indices for each event class."""
#     print_analysis_progress("Creating event class top indices bar plots...")
    
#     for dataset_name, dataset_data in all_data.items():
#         class_info = dataset_data['class_info']
#         categories = class_info['categories']
        
#         modalities = ['audio', 'video']
        
#         for modality in modalities:
#             class_ids = list(dataset_data['codebook_indices'][modality].keys())
#             if not class_ids:
#                 continue
                
#             n_classes = len(class_ids)
#             cols = min(4, n_classes)
#             rows = (n_classes + cols - 1) // cols
            
#             fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
#             if n_classes == 1:
#                 axes = [axes]
#             elif rows == 1:
#                 axes = axes if n_classes > 1 else [axes]
#             else:
#                 axes = axes.flatten()
            
#             fig.suptitle(f'{dataset_name} - Top {modality.title()} Codebook Indices by Event Class', 
#                         fontsize=14, fontweight='bold')
            
#             for i, class_id in enumerate(class_ids):
#                 ax = axes[i]
                
#                 indices = dataset_data['codebook_indices'][modality][class_id]
                
#                 if indices:
#                     counter = Counter(indices)
#                     top_indices = counter.most_common(10)
                    
#                     if top_indices:
#                         indices_list, counts_list = zip(*top_indices)
                        
#                         bars = ax.bar(range(len(indices_list)), counts_list, alpha=0.7)
#                         ax.set_xlabel('Codebook Index')
#                         ax.set_ylabel('Usage Count')
                        
#                         class_name = categories[class_id] if class_id < len(categories) else f'Class_{class_id}'
#                         ax.set_title(f'{class_name}')
#                         ax.set_xticks(range(len(indices_list)))
#                         ax.set_xticklabels([str(idx) for idx in indices_list], rotation=45)
                        
#                         # Add value labels on bars
#                         for bar, count in zip(bars, counts_list):
#                             height = bar.get_height()
#                             ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                                    f'{count}', ha='center', va='bottom', fontsize=8)
#                 else:
#                     ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
#                     ax.set_title(f'Class {class_id}')
            
#             # Hide empty subplots
#             for i in range(n_classes, len(axes)):
#                 axes[i].set_visible(False)
            
#             plt.tight_layout()
#             plt.savefig(os.path.join(viz_dir, f'{dataset_name}_{modality}_event_top_indices.png'), 
#                        dpi=300, bbox_inches='tight')
#             plt.close()

# def create_event_diversity_analysis(all_data, viz_dir):
#     """Create plots analyzing codebook diversity across event classes."""
#     print_analysis_progress("Creating event diversity analysis plots...")
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#     fig.suptitle('Event Class Codebook Diversity Analysis', fontsize=16, fontweight='bold')
    
#     # Plot 1: Number of unique indices per event class
#     ax1 = axes[0, 0]
    
#     for dataset_name, dataset_data in all_data.items():
#         class_info = dataset_data['class_info']
#         categories = class_info['categories']
        
#         modalities = ['audio', 'video']
        
#         for modality in modalities:
#             class_ids = list(dataset_data['codebook_indices'][modality].keys())
#             unique_counts = []
#             class_labels = []
            
#             for class_id in sorted(class_ids):
#                 indices = dataset_data['codebook_indices'][modality][class_id]
#                 unique_counts.append(len(set(indices)) if indices else 0)
#                 class_name = categories[class_id] if class_id < len(categories) else f'C{class_id}'
#                 class_labels.append(class_name)
            
#             if unique_counts:
#                 ax1.plot(range(len(class_ids)), unique_counts, marker='o', 
#                         label=f'{dataset_name}-{modality}', alpha=0.7)
    
#     ax1.set_xlabel('Event Class')
#     ax1.set_ylabel('Number of Unique Codebook Indices')
#     ax1.set_title('Codebook Diversity by Event Class')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # Plot 2: Total usage per event class
#     ax2 = axes[0, 1]
    
#     for dataset_name, dataset_data in all_data.items():
#         for modality in modalities:
#             class_ids = list(dataset_data['codebook_indices'][modality].keys())
#             total_counts = []
            
#             for class_id in sorted(class_ids):
#                 indices = dataset_data['codebook_indices'][modality][class_id]
#                 total_counts.append(len(indices) if indices else 0)
            
#             if total_counts:
#                 ax2.plot(range(len(class_ids)), total_counts, marker='s', 
#                         label=f'{dataset_name}-{modality}', alpha=0.7)
    
#     ax2.set_xlabel('Event Class')
#     ax2.set_ylabel('Total Codebook Usage')
#     ax2.set_title('Total Codebook Usage by Event Class')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
    
#     # Plot 3: Usage distribution comparison
#     ax3 = axes[1, 0]
    
#     for dataset_name, dataset_data in all_data.items():
#         for modality in modalities:
#             all_indices = []
#             for class_id, indices in dataset_data['codebook_indices'][modality].items():
#                 all_indices.extend(indices)
            
#             if all_indices:
#                 unique_indices = len(set(all_indices))
#                 total_usage = len(all_indices)
#                 diversity_ratio = unique_indices / total_usage if total_usage > 0 else 0
                
#                 ax3.bar(f'{dataset_name}\n{modality}', diversity_ratio, alpha=0.7, 
#                        label=f'{dataset_name}-{modality}')
    
#     ax3.set_ylabel('Diversity Ratio (Unique/Total)')
#     ax3.set_title('Codebook Diversity Ratio by Dataset')
#     ax3.tick_params(axis='x', rotation=45)
    
#     # Plot 4: Event class distribution
#     ax4 = axes[1, 1]
    
#     for dataset_name, dataset_data in all_data.items():
#         class_info = dataset_data['class_info']
#         categories = class_info['categories']
        
#         # Count samples per class across modalities
#         class_sample_counts = defaultdict(int)
#         for modality in ['audio', 'video']:
#             for class_id, indices in dataset_data['codebook_indices'][modality].items():
#                 if indices:  # Only count classes with data
#                     class_sample_counts[class_id] += 1
        
#         if class_sample_counts:
#             class_ids = sorted(class_sample_counts.keys())
#             counts = [class_sample_counts[cid] for cid in class_ids]
            
#             bars = ax4.bar([f'{categories[cid] if cid < len(categories) else f"C{cid}"}' 
#                            for cid in class_ids], counts, alpha=0.7, label=dataset_name)
    
#     ax4.set_ylabel('Number of Classes with Data')
#     ax4.set_title('Event Class Data Availability')
#     ax4.tick_params(axis='x', rotation=45)
#     ax4.legend()
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(viz_dir, 'event_diversity_analysis.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     """
#     Main function to run the comprehensive event-codebook analysis.
#     """
#     import argparse
    
#     # Set up command-line argument parsing
#     parser = argparse.ArgumentParser(description='Comprehensive Event-Codebook Analysis for VQ-VAE Models')
    
#     # Dataset arguments
#     parser.add_argument('--dataset', type=str, choices=['ave', 'avvp'], required=True,
#                         help='Dataset to analyze: ave or avvp')
#     parser.add_argument('--ave_data_root', type=str, 
#                         default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVE/data',
#                         help='Root directory for AVE data')
#     parser.add_argument('--avvp_csv_path', type=str,
#                         default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_multimodal_simplified.csv',
#                         help='CSV file path for AVVP data')
#     parser.add_argument('--avvp_audio_path', type=str,
#                         default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/audio/zip',
#                         help='AVVP audio features path')
#     parser.add_argument('--avvp_video_path', type=str, 
#                         default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/video/zip',
#                         help='AVVP video features path')
    
#     # Model and checkpoint arguments
#     parser.add_argument('--checkpoint_path', type=str, required=True,
#                         help='Path to the pretrained model checkpoint')
#     parser.add_argument('--output_dir', type=str, required=True,
#                         help='Directory to save analysis results')
#     parser.add_argument('--model_type', type=str, choices=['AV', 'AVT'], default='AVT',
#                         help='Model type: AV or AVT')
    
#     # Analysis configuration arguments
#     parser.add_argument('--samples_per_class', type=int, default=3,
#                         help='Number of samples to collect per event class (default: 3)')
    
#     # Model configuration arguments
#     parser.add_argument('--audio_dim', type=int, default=128,
#                         help='Audio feature dimension (default: 128)')
#     parser.add_argument('--video_dim', type=int, default=512,
#                         help='Video feature dimension (default: 512)')
#     parser.add_argument('--text_dim', type=int, default=256,
#                         help='Text feature dimension (default: 256)')
#     parser.add_argument('--video_output_dim', type=int, default=2048,
#                         help='Video output dimension (default: 2048)')
#     parser.add_argument('--n_embeddings', type=int, default=400,
#                         help='Number of codebook embeddings (default: 400)')
#     parser.add_argument('--embedding_dim', type=int, default=256,
#                         help='Embedding dimension (default: 256)')
    
#     # Parse arguments
#     args = parser.parse_args()
    
#     # Create model configuration dictionary
#     MODEL_CONFIG = {
#         'audio_dim': args.audio_dim,
#         'video_dim': args.video_dim,
#         'text_dim': args.text_dim,
#         'video_output_dim': args.video_output_dim,
#         'n_embeddings': args.n_embeddings,
#         'embedding_dim': args.embedding_dim
#     }
    
#     print_analysis_progress("Starting comprehensive event-codebook analysis...")
#     print_analysis_progress(f"Configuration:")
#     print_analysis_progress(f"  Dataset: {args.dataset}")
#     print_analysis_progress(f"  Checkpoint: {args.checkpoint_path}")
#     print_analysis_progress(f"  Output directory: {args.output_dir}")
#     print_analysis_progress(f"  Model type: {args.model_type}")
#     print_analysis_progress(f"  Samples per class: {args.samples_per_class}")
#     print_analysis_progress(f"  Model config: {MODEL_CONFIG}")
    
#     # Validate checkpoint path
#     if not os.path.exists(args.checkpoint_path):
#         print_analysis_progress(f"ERROR: Checkpoint file not found at {args.checkpoint_path}")
#         exit(1)
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
#     print_analysis_progress(f"Output directory created: {args.output_dir}")
    
#     # Load and stratify datasets
#     print_analysis_progress(f"Loading {args.dataset.upper()} dataset...")
#     sampled_datasets = {}
    
#     if args.dataset == 'ave':
#         ave_dataset = AVEDatasetAnalysis(args.ave_data_root, split='test')
#         ave_samples = stratify_samples_by_events_ave(ave_dataset, args.samples_per_class)
#         sampled_datasets['AVE'] = ave_samples
        
#     elif args.dataset == 'avvp':
#         # Load AVVP test split - you may need to filter the CSV for test samples
#         avvp_dataset = AVVPDatasetAnalysis(args.avvp_csv_path, args.avvp_audio_path, 
#                                           args.avvp_video_path, split='test')
#         avvp_samples = stratify_samples_by_events_avvp(avvp_dataset, args.samples_per_class)
#         sampled_datasets['AVVP'] = avvp_samples
    
#     # Create visualizations
#     print_analysis_progress("\nCreating event-codebook visualizations...")
#     create_event_class_visualizations(
#         sampled_datasets, args.checkpoint_path, MODEL_CONFIG, args.output_dir, args.model_type
#     )
    
#     print_analysis_progress("Event analysis pipeline completed successfully!")
#     print_analysis_progress(f"Check results in: {args.output_dir}")
#     print_analysis_progress(f"Check visualizations in: {os.path.join(args.output_dir, 'visualizations')}")

# if __name__ == "__main__":
#     main()



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import os
import sys
from datetime import datetime
import pandas as pd
import h5py
import zipfile
from io import BytesIO
import pickle

# Add your project paths
sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/dataset")
sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src")

from model.main_model_novel import AV_VQVAE_Encoder, AVT_VQVAE_Encoder

def print_analysis_progress(message):
    """Helper function for logging analysis progress"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def generate_ave_category_list():
    """Generate AVE category list - 28 classes"""
    # You'll need to provide the actual AVE categories
    return [f"event_{i}" for i in range(28)]  # Placeholder

def generate_avvp_category_list():
    """Generate AVVP category list from file"""
    file_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_Categories.txt'
    category_list = []
    try:
        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                category_list.append(line.strip())
        category_list.append("background")  # Add background class
    except FileNotFoundError:
        print_analysis_progress(f"Warning: Category file not found, using placeholder categories")
        category_list = [f"avvp_event_{i}" for i in range(25)] + ["background"]
    return category_list

class AVEDatasetAnalysis:
    def __init__(self, data_root, split='test'):
        self.split = split
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')
        self.labels_path = os.path.join(data_root, 'labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.categories = generate_ave_category_list()
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True
        
        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]
        
        return {
            'audio_fea': audio_feat,
            'video_fea': visual_feat, 
            'labels': label,
            'video_ids': f"ave_sample_{sample_index}"
        }

    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num

class AVVPDatasetAnalysis:
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, split='test'):
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        self.split_df = pd.read_csv(meta_csv_path, sep='\t')
        self.categories = generate_avvp_category_list()
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, f"{video_id}.zip")
        try:
            with zipfile.ZipFile(fea_path, mode='r') as zfile:
                for name in zfile.namelist():
                    if '.pkl' not in name:
                        continue
                    with zfile.open(name, mode='r') as fea_file:
                        content = BytesIO(fea_file.read())
                        fea = pickle.load(content)
            return fea
        except Exception as e:
            print_analysis_progress(f"Error loading feature {fea_path}: {e}")
            return None
    
    def _obtain_avel_label(self, onsets, offsets, categorys):
        T, category_num = 10, len(self.categories) - 1  # Exclude background for now
        label = np.zeros((T, category_num + 1))  # Add background
        label[:, -1] = np.ones(T)  # Background default
        
        iter_num = len(categorys)
        for i in range(iter_num):
            avc_label = np.zeros(T)
            avc_label[onsets[i]:offsets[i]] = 1
            if categorys[i] in self.categories[:-1]:  # Exclude background
                class_id = self.categories.index(categorys[i])
                bg_flag = 1 - avc_label
                for j in range(10):
                    label[j, class_id] = int(label[j, class_id]) | int(avc_label[j])
                for j in range(10):
                    label[j, -1] = int(label[j, -1]) & int(bg_flag[j])
        return label

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        categorys, video_id = one_video_df['event_labels'].split(','), one_video_df['filename']
        onsets, offsets = one_video_df['onset'].split(','), one_video_df['offset'].split(',')
        onsets = list(map(int, onsets))
        offsets = list(map(int, offsets))
        
        audio_fea = self._load_fea(self.audio_fea_base_path, video_id[:11])
        video_fea = self._load_fea(self.video_fea_base_path, video_id[:11])
        
        if audio_fea is None or video_fea is None:
            return None
            
        # Handle audio feature padding/truncation
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]
            
        avel_label = self._obtain_avel_label(onsets, offsets, categorys)
        
        return {
            'audio_fea': audio_fea,
            'video_fea': video_fea,
            'labels': avel_label,
            'video_ids': video_id,
            'event_categories': categorys
        }

    def __len__(self):
        return len(self.split_df)

def stratify_samples_by_events_ave(dataset, samples_per_class=2):
    """
    Sample data points from different AVE event classes.
    """
    print_analysis_progress(f"Stratifying AVE samples by event classes...")
    
    # Collect samples by event class
    class_samples = defaultdict(list)
    categories = generate_ave_category_list()
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            label = sample['labels']
            
            # Debug: print first few labels to understand structure
            if i < 3:
                print_analysis_progress(f"Sample {i} label shape: {label.shape if hasattr(label, 'shape') else 'no shape'}, "
                                      f"type: {type(label)}, value: {label}")
            
            # Handle different label formats in AVE dataset
            if isinstance(label, np.ndarray):
                # Flatten the label array to handle multi-dimensional cases
                label_flat = label.flatten()
                
                if len(label_flat) == 1:
                    # Single value in array
                    class_id = int(label_flat[0])
                elif len(label_flat) > 1:
                    # Check if it's one-hot encoded (exactly one 1, rest 0s)
                    unique_vals = np.unique(label_flat)
                    if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals and np.sum(label_flat) == 1:
                        # One-hot encoded
                        class_id = int(np.argmax(label_flat))
                    else:
                        # Multi-label or other format - use the first non-zero class or argmax
                        nonzero_indices = np.nonzero(label_flat)[0]
                        if len(nonzero_indices) > 0:
                            class_id = int(nonzero_indices[0])
                        else:
                            class_id = int(np.argmax(label_flat))
                else:
                    # Empty array - skip
                    print_analysis_progress(f"Warning: Empty label array for sample {i}")
                    continue
                    
            elif isinstance(label, (int, np.integer)):
                class_id = int(label)
            elif isinstance(label, (float, np.floating)):
                class_id = int(round(label))
            else:
                # Try to convert to int
                try:
                    class_id = int(label)
                except (ValueError, TypeError):
                    print_analysis_progress(f"Warning: Cannot convert label {label} (type: {type(label)}) to class_id for sample {i}")
                    continue
            
            # Validate class_id is within expected range
            if 0 <= class_id < len(categories):
                sample['class_id'] = class_id
                sample['class_name'] = categories[class_id]
                class_samples[class_id].append(sample)
            else:
                print_analysis_progress(f"Warning: Class ID {class_id} out of range [0, {len(categories)-1}] for sample {i}")
                
        except Exception as e:
            print_analysis_progress(f"Error processing sample {i}: {e}")
            if i < 5:  # Print more details for first few errors
                print_analysis_progress(f"  Label details: shape={getattr(label, 'shape', 'N/A')}, "
                                      f"dtype={getattr(label, 'dtype', 'N/A')}, value={label}")
            continue
    
    print_analysis_progress(f"Successfully processed {sum(len(samples) for samples in class_samples.values())} samples across {len(class_samples)} classes")
    
    # Sample from each class
    sampled_data = {
        'samples': {},
        'class_info': {
            'categories': categories,
            'dataset_name': 'AVE'
        }
    }
    
    for class_id, samples in class_samples.items():
        if len(samples) >= samples_per_class:
            selected_samples = np.random.choice(samples, samples_per_class, replace=False)
        else:
            selected_samples = samples
            print_analysis_progress(f"Warning: Only {len(samples)} samples for class {class_id}")
        
        sampled_data['samples'][class_id] = list(selected_samples)
        print_analysis_progress(f"Class {class_id} ({categories[class_id]}): {len(selected_samples)} samples")
    
    return sampled_data

def stratify_samples_by_events_avvp(dataset, samples_per_class=2, focus_on_primary=True):
    """
    Sample data points from AVVP dataset by event classes.
    Since AVVP is multi-label, we can focus on primary events or sample by individual classes.
    """
    print_analysis_progress(f"Stratifying AVVP samples by event classes...")
    
    class_samples = defaultdict(list)
    categories = generate_avvp_category_list()
    
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            if sample is None:
                continue
                
            labels = sample['labels']  # Shape: [10, 26]
            event_categories = sample.get('event_categories', [])
            
            if focus_on_primary and event_categories:
                # Use the first mentioned event as primary
                primary_event = event_categories[0]
                if primary_event in categories[:-1]:  # Exclude background
                    class_id = categories.index(primary_event)
                    sample['class_id'] = class_id
                    sample['class_name'] = primary_event
                    sample['is_primary'] = True
                    class_samples[class_id].append(sample)
            else:
                # Sample based on all active classes in the temporal sequence
                active_classes = []
                for t in range(labels.shape[0]):  # For each timestep
                    for c in range(labels.shape[1] - 1):  # Exclude background
                        if labels[t, c] > 0:
                            active_classes.append(c)
                
                # Add sample to each active class
                for class_id in set(active_classes):
                    sample_copy = sample.copy()
                    sample_copy['class_id'] = class_id
                    sample_copy['class_name'] = categories[class_id]
                    sample_copy['is_primary'] = False
                    class_samples[class_id].append(sample_copy)
                    
        except Exception as e:
            print_analysis_progress(f"Error processing AVVP sample {i}: {e}")
            continue
    
    # Sample from each class
    sampled_data = {
        'samples': {},
        'class_info': {
            'categories': categories,
            'dataset_name': 'AVVP'
        }
    }
    
    for class_id, samples in class_samples.items():
        if len(samples) >= samples_per_class:
            selected_samples = np.random.choice(samples, samples_per_class, replace=False)
        else:
            selected_samples = samples
            print_analysis_progress(f"Warning: Only {len(samples)} samples for class {class_id}")
        
        sampled_data['samples'][class_id] = list(selected_samples)
        print_analysis_progress(f"Class {class_id} ({categories[class_id]}): {len(selected_samples)} samples")
    
    return sampled_data

def load_pretrained_encoder(checkpoint_path, model_config, model_type='AVT'):
    """
    Load a pretrained encoder from checkpoint.
    """
    print_analysis_progress(f"Loading pretrained {model_type} encoder from: {checkpoint_path}")
    
    if model_type == 'AVT':
        encoder = AVT_VQVAE_Encoder(
            audio_dim=model_config['audio_dim'],
            video_dim=model_config['video_dim'], 
            text_dim=model_config['text_dim'],
            video_output_dim=model_config['video_output_dim'],
            n_embeddings=model_config['n_embeddings'],
            embedding_dim=model_config['embedding_dim']
        )
    else:  # AV
        encoder = AV_VQVAE_Encoder(
            audio_dim=model_config['audio_dim'],
            video_dim=model_config['video_dim'],
            video_output_dim=model_config['video_output_dim'], 
            n_embeddings=model_config['n_embeddings'],
            embedding_dim=model_config['embedding_dim']
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['Encoder_parameters'])
    encoder = encoder.double().to(device)
    encoder.eval()
    
    print_analysis_progress(f"{model_type} Encoder loaded successfully on {device}")
    return encoder

def analyze_sample_quantization_events(encoder, sample, class_id, class_name, dataset_name, model_type='AVT'):
    """
    Analyze how a single sample gets quantized by the VQ-VAE encoder for event detection.
    """
    print_analysis_progress(f"\n{'='*100}")
    print_analysis_progress(f"EVENT QUANTIZATION ANALYSIS - {dataset_name} Sample")
    print_analysis_progress(f"Event Class: {class_id} ({class_name})")
    print_analysis_progress(f"Video ID: {sample.get('video_ids', 'unknown')}")
    print_analysis_progress(f"{'='*100}")
    
    device = next(encoder.parameters()).device
    
    # Prepare input tensors
    audio_feat = torch.from_numpy(sample['audio_fea']).unsqueeze(0).double().to(device)
    video_feat = torch.from_numpy(sample['video_fea']).unsqueeze(0).double().to(device)
    
    with torch.no_grad():
        if model_type == 'AVT':
            # For AVT model, we need text features - use zero padding or skip text analysis
            text_feat = torch.zeros(1, audio_feat.shape[1], 256).double().to(device)  # Dummy text
            
            # Get quantized representations using individual VQ encoders
            out_vq_audio, audio_vq = encoder.Audio_VQ_Encoder(audio_feat)
            out_vq_video, video_vq = encoder.Video_VQ_Encoder(video_feat)
            # out_vq_text, text_vq = encoder.Text_VQ_Encoder(text_feat)  # Skip for event analysis
            
        else:  # AV model
            out_vq_audio, audio_vq = encoder.Audio_VQ_Encoder(audio_feat)
            out_vq_video, video_vq = encoder.Video_VQ_Encoder(video_feat)
    
    # Remove batch dimension for analysis
    audio_quantized = audio_vq.squeeze(0).cpu().numpy()
    video_quantized = video_vq.squeeze(0).cpu().numpy()
    out_vq_audio_full = out_vq_audio.squeeze(0).cpu().numpy()
    out_vq_video_full = out_vq_video.squeeze(0).cpu().numpy()
    
    print_analysis_progress("Quantized representations extracted successfully!")
    
    # Analyze codebook usage patterns
    analysis_results = {
        'sample_info': {
            'video_id': sample.get('video_ids', 'unknown'),
            'class_id': class_id,
            'class_name': class_name,
            'dataset_name': dataset_name
        },
        'quantized_representations': {
            'audio': audio_quantized,
            'video': video_quantized
        },
        'full_quantized_vectors': {
            'audio': out_vq_audio_full,
            'video': out_vq_video_full
        },
        'quantization_indices': {},
        'codebook_matches': {}
    }
    
    # Get the quantizer to analyze codebook usage
    quantizer = encoder.Cross_quantizer
    codebook_embedding = quantizer.embedding.detach().cpu().numpy()
    
    if model_type == 'AVT':
        # For AVT: codebook is split into [video, audio, text] segments
        D = audio_quantized.shape[-1]  # Should be 256
        video_embedding = codebook_embedding[:, :D]
        audio_embedding = codebook_embedding[:, D:2*D]
    else:
        # For AV: codebook is split into [video, audio] segments  
        D = audio_quantized.shape[-1]
        video_embedding = codebook_embedding[:, :D]
        audio_embedding = codebook_embedding[:, D:]
    
    # Find quantization indices for each modality
    modalities = [
        ('audio', audio_quantized, audio_embedding),
        ('video', video_quantized, video_embedding)
    ]
    
    for mod_name, quantized, embedding_segment in modalities:
        print_analysis_progress(f"\n{mod_name.upper()} Quantization Analysis:")
        
        seq_len = quantized.shape[0]
        quantization_indices = []
        
        for t in range(seq_len):
            # Find closest codebook vector
            distances = np.sum((embedding_segment - quantized[t])**2, axis=1)
            closest_idx = np.argmin(distances)
            quantization_indices.append((t, closest_idx, distances[closest_idx]))
            
            if t < 3 or t == seq_len - 1:  # Show first few and last timestep
                print_analysis_progress(f"  Timestep {t}: Codebook vector {closest_idx} (distance: {distances[closest_idx]:.6f})")
        
        analysis_results['quantization_indices'][mod_name] = quantization_indices
        
        # Analyze usage patterns
        used_indices = [idx for _, idx, _ in quantization_indices]
        unique_indices = set(used_indices)
        usage_counter = Counter(used_indices)
        
        print_analysis_progress(f"  Total timesteps: {len(quantization_indices)}")
        print_analysis_progress(f"  Unique vectors used: {len(unique_indices)}")
        print_analysis_progress(f"  Most frequent vectors: {usage_counter.most_common(3)}")
    
    return analysis_results

def create_event_class_visualizations(sampled_datasets, checkpoint_path, model_config, output_dir, model_type='AVT'):
    """
    Create visualizations showing how codebook indices are distributed across event classes.
    """
    print_analysis_progress(f"\n{'='*120}")
    print_analysis_progress("CREATING EVENT-CODEBOOK VISUALIZATIONS")
    print_analysis_progress(f"{'='*120}")
    
    # Create visualization subdirectory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(checkpoint_path, model_config, model_type)
    
    # Set up plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Collect all quantization data
    all_data = {}
    
    for dataset_name, dataset_samples in sampled_datasets.items():
        print_analysis_progress(f"Processing {dataset_name} for visualization...")
        
        class_info = dataset_samples['class_info']
        samples = dataset_samples['samples']
        
        dataset_data = {
            'codebook_indices': {'audio': {}, 'video': {}},
            'class_info': class_info
        }
        
        for class_id, class_samples in samples.items():
            # Initialize storage for this class
            for modality in ['audio', 'video']:
                dataset_data['codebook_indices'][modality][class_id] = []
            
            for sample in class_samples:
                try:
                    # Analyze quantization for this sample
                    analysis = analyze_sample_quantization_events(
                        encoder, sample, class_id, 
                        class_info['categories'][class_id], dataset_name, model_type
                    )
                    
                    # Extract indices for each modality
                    for modality in ['audio', 'video']:
                        indices = [idx for t, idx, dist in analysis['quantization_indices'][modality]]
                        dataset_data['codebook_indices'][modality][class_id].extend(indices)
                        
                except Exception as e:
                    print_analysis_progress(f"Error analyzing sample: {e}")
                    continue
        
        all_data[dataset_name] = dataset_data
    
    # Create visualizations
    print_analysis_progress("Creating event class visualizations...")
    
    # 1. Event Class Codebook Heatmaps
    create_event_class_heatmaps(all_data, viz_dir)
    
    # 2. Top Codebook Indices Bar Plots by Event Class
    create_event_top_indices_barplots(all_data, viz_dir)
    
    # 3. Event Class Diversity Analysis
    create_event_diversity_analysis(all_data, viz_dir)
    
    print_analysis_progress(f"Event visualizations saved to: {viz_dir}")

def create_event_class_heatmaps(all_data, viz_dir):
    """Create heatmaps showing codebook usage across event classes."""
    print_analysis_progress("Creating event class codebook heatmaps...")
    
    for dataset_name, dataset_data in all_data.items():
        class_info = dataset_data['class_info']
        categories = class_info['categories']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle(f'{dataset_name} - Codebook Usage by Event Class', fontsize=16, fontweight='bold')
        
        modalities = ['audio', 'video']
        
        for i, modality in enumerate(modalities):
            ax = axes[i]
            
            # Collect usage data for heatmap
            codebook_usage = defaultdict(lambda: defaultdict(int))
            all_indices = set()
            
            for class_id, indices in dataset_data['codebook_indices'][modality].items():
                counter = Counter(indices)
                for idx, count in counter.items():
                    codebook_usage[class_id][idx] = count
                    all_indices.add(idx)
            
            if all_indices:
                # Create matrix for heatmap
                sorted_indices = sorted(all_indices)
                sorted_classes = sorted(codebook_usage.keys())
                matrix = np.zeros((len(sorted_classes), len(sorted_indices)))
                
                for class_idx, class_id in enumerate(sorted_classes):
                    for idx_pos, codebook_idx in enumerate(sorted_indices):
                        matrix[class_idx, idx_pos] = codebook_usage[class_id][codebook_idx]
                
                # Plot heatmap
                sns.heatmap(matrix, ax=ax, cmap='viridis', cbar=True,
                           xticklabels=[f'{idx}' for idx in sorted_indices[::max(1, len(sorted_indices)//10)]], 
                           yticklabels=[f'{categories[i] if i < len(categories) else f"Class_{i}"}' for i in sorted_classes])
                
                ax.set_title(f'{modality.title()} Modality')
                ax.set_xlabel('Codebook Index')
                ax.set_ylabel('Event Class')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{modality.title()} Modality')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'{dataset_name}_event_codebook_heatmaps.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_event_top_indices_barplots(all_data, viz_dir):
    """Create bar plots showing top codebook indices for each event class."""
    print_analysis_progress("Creating event class top indices bar plots...")
    
    for dataset_name, dataset_data in all_data.items():
        class_info = dataset_data['class_info']
        categories = class_info['categories']
        
        modalities = ['audio', 'video']
        
        for modality in modalities:
            class_ids = list(dataset_data['codebook_indices'][modality].keys())
            if not class_ids:
                continue
                
            n_classes = len(class_ids)
            cols = min(4, n_classes)
            rows = (n_classes + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            if n_classes == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if n_classes > 1 else [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'{dataset_name} - Top {modality.title()} Codebook Indices by Event Class', 
                        fontsize=14, fontweight='bold')
            
            for i, class_id in enumerate(class_ids):
                ax = axes[i]
                
                indices = dataset_data['codebook_indices'][modality][class_id]
                
                if indices:
                    counter = Counter(indices)
                    top_indices = counter.most_common(10)
                    
                    if top_indices:
                        indices_list, counts_list = zip(*top_indices)
                        
                        bars = ax.bar(range(len(indices_list)), counts_list, alpha=0.7)
                        ax.set_xlabel('Codebook Index')
                        ax.set_ylabel('Usage Count')
                        
                        class_name = categories[class_id] if class_id < len(categories) else f'Class_{class_id}'
                        ax.set_title(f'{class_name}')
                        ax.set_xticks(range(len(indices_list)))
                        ax.set_xticklabels([str(idx) for idx in indices_list], rotation=45)
                        
                        # Add value labels on bars
                        for bar, count in zip(bars, counts_list):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{count}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Class {class_id}')
            
            # Hide empty subplots
            for i in range(n_classes, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'{dataset_name}_{modality}_event_top_indices.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def create_event_diversity_analysis(all_data, viz_dir):
    """Create plots analyzing codebook diversity across event classes."""
    print_analysis_progress("Creating event diversity analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Event Class Codebook Diversity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of unique indices per event class
    ax1 = axes[0, 0]
    
    for dataset_name, dataset_data in all_data.items():
        class_info = dataset_data['class_info']
        categories = class_info['categories']
        
        modalities = ['audio', 'video']
        
        for modality in modalities:
            class_ids = list(dataset_data['codebook_indices'][modality].keys())
            unique_counts = []
            class_labels = []
            
            for class_id in sorted(class_ids):
                indices = dataset_data['codebook_indices'][modality][class_id]
                unique_counts.append(len(set(indices)) if indices else 0)
                class_name = categories[class_id] if class_id < len(categories) else f'C{class_id}'
                class_labels.append(class_name)
            
            if unique_counts:
                ax1.plot(range(len(class_ids)), unique_counts, marker='o', 
                        label=f'{dataset_name}-{modality}', alpha=0.7)
    
    ax1.set_xlabel('Event Class')
    ax1.set_ylabel('Number of Unique Codebook Indices')
    ax1.set_title('Codebook Diversity by Event Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total usage per event class
    ax2 = axes[0, 1]
    
    for dataset_name, dataset_data in all_data.items():
        for modality in modalities:
            class_ids = list(dataset_data['codebook_indices'][modality].keys())
            total_counts = []
            
            for class_id in sorted(class_ids):
                indices = dataset_data['codebook_indices'][modality][class_id]
                total_counts.append(len(indices) if indices else 0)
            
            if total_counts:
                ax2.plot(range(len(class_ids)), total_counts, marker='s', 
                        label=f'{dataset_name}-{modality}', alpha=0.7)
    
    ax2.set_xlabel('Event Class')
    ax2.set_ylabel('Total Codebook Usage')
    ax2.set_title('Total Codebook Usage by Event Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Usage distribution comparison
    ax3 = axes[1, 0]
    
    for dataset_name, dataset_data in all_data.items():
        for modality in modalities:
            all_indices = []
            for class_id, indices in dataset_data['codebook_indices'][modality].items():
                all_indices.extend(indices)
            
            if all_indices:
                unique_indices = len(set(all_indices))
                total_usage = len(all_indices)
                diversity_ratio = unique_indices / total_usage if total_usage > 0 else 0
                
                ax3.bar(f'{dataset_name}\n{modality}', diversity_ratio, alpha=0.7, 
                       label=f'{dataset_name}-{modality}')
    
    ax3.set_ylabel('Diversity Ratio (Unique/Total)')
    ax3.set_title('Codebook Diversity Ratio by Dataset')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Event class distribution
    ax4 = axes[1, 1]
    
    for dataset_name, dataset_data in all_data.items():
        class_info = dataset_data['class_info']
        categories = class_info['categories']
        
        # Count samples per class across modalities
        class_sample_counts = defaultdict(int)
        for modality in ['audio', 'video']:
            for class_id, indices in dataset_data['codebook_indices'][modality].items():
                if indices:  # Only count classes with data
                    class_sample_counts[class_id] += 1
        
        if class_sample_counts:
            class_ids = sorted(class_sample_counts.keys())
            counts = [class_sample_counts[cid] for cid in class_ids]
            
            bars = ax4.bar([f'{categories[cid] if cid < len(categories) else f"C{cid}"}' 
                           for cid in class_ids], counts, alpha=0.7, label=dataset_name)
    
    ax4.set_ylabel('Number of Classes with Data')
    ax4.set_title('Event Class Data Availability')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'event_diversity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the comprehensive event-codebook analysis.
    """
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Comprehensive Event-Codebook Analysis for VQ-VAE Models')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['ave', 'avvp'], required=True,
                        help='Dataset to analyze: ave or avvp')
    parser.add_argument('--ave_data_root', type=str, 
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVE/data',
                        help='Root directory for AVE data')
    parser.add_argument('--avvp_csv_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_multimodal_simplified.csv',
                        help='CSV file path for AVVP data')
    parser.add_argument('--avvp_audio_path', type=str,
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/audio/zip',
                        help='AVVP audio features path')
    parser.add_argument('--avvp_video_path', type=str, 
                        default='/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/video/zip',
                        help='AVVP video features path')
    
    # Model and checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save analysis results')
    parser.add_argument('--model_type', type=str, choices=['AV', 'AVT'], default='AVT',
                        help='Model type: AV or AVT')
    
    # Analysis configuration arguments
    parser.add_argument('--samples_per_class', type=int, default=3,
                        help='Number of samples to collect per event class (default: 3)')
    
    # Model configuration arguments
    parser.add_argument('--audio_dim', type=int, default=128,
                        help='Audio feature dimension (default: 128)')
    parser.add_argument('--video_dim', type=int, default=512,
                        help='Video feature dimension (default: 512)')
    parser.add_argument('--text_dim', type=int, default=256,
                        help='Text feature dimension (default: 256)')
    parser.add_argument('--video_output_dim', type=int, default=2048,
                        help='Video output dimension (default: 2048)')
    parser.add_argument('--n_embeddings', type=int, default=400,
                        help='Number of codebook embeddings (default: 400)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension (default: 256)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create model configuration dictionary
    MODEL_CONFIG = {
        'audio_dim': args.audio_dim,
        'video_dim': args.video_dim,
        'text_dim': args.text_dim,
        'video_output_dim': args.video_output_dim,
        'n_embeddings': args.n_embeddings,
        'embedding_dim': args.embedding_dim
    }
    
    print_analysis_progress("Starting comprehensive event-codebook analysis...")
    print_analysis_progress(f"Configuration:")
    print_analysis_progress(f"  Dataset: {args.dataset}")
    print_analysis_progress(f"  Checkpoint: {args.checkpoint_path}")
    print_analysis_progress(f"  Output directory: {args.output_dir}")
    print_analysis_progress(f"  Model type: {args.model_type}")
    print_analysis_progress(f"  Samples per class: {args.samples_per_class}")
    print_analysis_progress(f"  Model config: {MODEL_CONFIG}")
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print_analysis_progress(f"ERROR: Checkpoint file not found at {args.checkpoint_path}")
        exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print_analysis_progress(f"Output directory created: {args.output_dir}")
    
    # Load and stratify datasets
    print_analysis_progress(f"Loading {args.dataset.upper()} dataset...")
    sampled_datasets = {}
    
    if args.dataset == 'ave':
        ave_dataset = AVEDatasetAnalysis(args.ave_data_root, split='test')
        ave_samples = stratify_samples_by_events_ave(ave_dataset, args.samples_per_class)
        sampled_datasets['AVE'] = ave_samples
        
    elif args.dataset == 'avvp':
        # Load AVVP test split - you may need to filter the CSV for test samples
        avvp_dataset = AVVPDatasetAnalysis(args.avvp_csv_path, args.avvp_audio_path, 
                                          args.avvp_video_path, split='test')
        avvp_samples = stratify_samples_by_events_avvp(avvp_dataset, args.samples_per_class)
        sampled_datasets['AVVP'] = avvp_samples
    
    # Create visualizations
    print_analysis_progress("\nCreating event-codebook visualizations...")
    create_event_class_visualizations(
        sampled_datasets, args.checkpoint_path, MODEL_CONFIG, args.output_dir, args.model_type
    )
    
    print_analysis_progress("Event analysis pipeline completed successfully!")
    print_analysis_progress(f"Check results in: {args.output_dir}")
    print_analysis_progress(f"Check visualizations in: {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()