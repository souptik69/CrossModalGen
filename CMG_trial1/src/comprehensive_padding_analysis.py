# #!/usr/bin/env python3
# """
# Comprehensive Padding Analysis for MOSEI and MOSI Datasets
# Analyzes padding statistics across different sentiment bins for all dataset splits.
# """

# import os
# import sys
# import torch
# import numpy as np
# import json
# import csv
# from datetime import datetime
# from collections import defaultdict, Counter
# import pandas as pd

# # Add your project paths
# sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/dataset")
# sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src")

# from dataset.MOSEI_MOSI import (
#     get_mosei_supervised_dataloaders,
#     get_mosi_dataloaders,
#     get_mosei_unsupervised_split_dataloaders
# )

# def print_analysis_progress(message):
#     """Helper function for logging analysis progress"""
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")
#     return timestamp

# def discretize_sentiment_labels(labels, num_bins=7):
#     """
#     Discretize continuous sentiment labels into bins.
    
#     Args:
#         labels: Array of continuous sentiment labels
#         num_bins: Number of discrete bins to create
    
#     Returns:
#         discrete_labels: Discretized labels (0 to num_bins-1)
#         bin_edges: The edges used for binning
#         label_mapping: Mapping from discrete labels to sentiment ranges
#     """
#     # Convert to numpy if needed
#     if isinstance(labels, torch.Tensor):
#         labels_np = labels.numpy()
#     else:
#         labels_np = np.array(labels)
    
#     # Flatten if needed
#     if len(labels_np.shape) > 1:
#         labels_np = labels_np.flatten()
    
#     # Create bins from min to max label values
#     min_label = np.min(labels_np)
#     max_label = np.max(labels_np)
#     bin_edges = np.linspace(min_label, max_label, num_bins + 1)
    
#     # Discretize labels
#     discrete_labels = np.digitize(labels_np, bin_edges[1:-1])  # Exclude the last edge for digitize
    
#     # Create label mapping for interpretation
#     label_mapping = {}
#     for i in range(num_bins):
#         label_mapping[i] = {
#             'range': f"[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})",
#             'min': bin_edges[i],
#             'max': bin_edges[i+1]
#         }
    
#     return discrete_labels, bin_edges, label_mapping

# def analyze_sample_padding(features, tolerance=1e-7):
#     """
#     Analyze padding in a single sample's features.
    
#     Args:
#         features: Feature array of shape [seq_len, feat_dim]
#         tolerance: Tolerance for identifying zero-padded timesteps
        
#     Returns:
#         padding_info: Dictionary with padding statistics
#     """
#     seq_len, feat_dim = features.shape
    
#     # Identify padding timesteps (those with all near-zero values)
#     padding_timesteps = []
#     content_timesteps = []
    
#     for t in range(seq_len):
#         if np.allclose(features[t], 0, atol=tolerance):
#             padding_timesteps.append(t)
#         else:
#             content_timesteps.append(t)
    
#     # Calculate statistics
#     num_padding = len(padding_timesteps)
#     num_content = len(content_timesteps)
#     padding_ratio = num_padding / seq_len if seq_len > 0 else 0.0
    
#     # Check for leading vs trailing padding
#     leading_padding = 0
#     trailing_padding = 0
    
#     if num_padding > 0:
#         # Count leading padding
#         for t in range(seq_len):
#             if t in padding_timesteps:
#                 leading_padding += 1
#             else:
#                 break
                
#         # Count trailing padding
#         for t in range(seq_len - 1, -1, -1):
#             if t in padding_timesteps:
#                 trailing_padding += 1
#             else:
#                 break
    
#     padding_info = {
#         'seq_len': seq_len,
#         'num_padding': num_padding,
#         'num_content': num_content,
#         'padding_ratio': padding_ratio,
#         'leading_padding': leading_padding,
#         'trailing_padding': trailing_padding,
#         'has_padding': num_padding > 0,
#         'is_fully_padded': num_padding == seq_len,
#         'padding_timesteps': padding_timesteps,
#         'content_timesteps': content_timesteps
#     }
    
#     return padding_info

# def analyze_batch_padding(batch_data, batch_idx, split_name, dataset_name, num_bins=7, tolerance=1e-7):
#     """
#     Analyze padding for an entire batch of data.
    
#     Args:
#         batch_data: Dictionary containing 'audio_fea', 'video_fea', 'text_fea', 'labels'
#         batch_idx: Batch index for identification
#         split_name: Split name (train/val/test)
#         dataset_name: Dataset name (MOSEI/MOSI)
#         num_bins: Number of sentiment bins
#         tolerance: Tolerance for padding detection
        
#     Returns:
#         batch_analysis: Dictionary containing detailed analysis results
#     """
#     audio_features = batch_data['audio_fea'].numpy()  # [batch_size, seq_len, 74]
#     video_features = batch_data['video_fea'].numpy()  # [batch_size, seq_len, 35]
#     text_features = batch_data['text_fea'].numpy()    # [batch_size, seq_len, 300]
#     labels = batch_data['labels'].numpy()             # [batch_size, 1]
    
#     batch_size = audio_features.shape[0]
    
#     # Discretize sentiment labels
#     discrete_labels, bin_edges, label_mapping = discretize_sentiment_labels(labels, num_bins)
    
#     batch_analysis = {
#         'batch_info': {
#             'batch_idx': batch_idx,
#             'batch_size': batch_size,
#             'split_name': split_name,
#             'dataset_name': dataset_name,
#             'seq_len': audio_features.shape[1]
#         },
#         'sentiment_info': {
#             'bin_edges': bin_edges,
#             'label_mapping': label_mapping,
#             'discrete_labels': discrete_labels
#         },
#         'samples': [],
#         'by_bin_stats': defaultdict(list),
#         'modality_stats': {
#             'audio': defaultdict(list),
#             'video': defaultdict(list),
#             'text': defaultdict(list)
#         }
#     }
    
#     # Analyze each sample in the batch
#     for i in range(batch_size):
#         sample_label = labels[i]
#         sample_bin = discrete_labels[i]
        
#         # Analyze padding for each modality
#         audio_padding = analyze_sample_padding(audio_features[i], tolerance)
#         video_padding = analyze_sample_padding(video_features[i], tolerance)
#         text_padding = analyze_sample_padding(text_features[i], tolerance)
        
#         sample_analysis = {
#             'sample_idx': i,
#             'global_sample_idx': batch_idx * batch_size + i,
#             'label': sample_label.item() if hasattr(sample_label, 'item') else float(sample_label),
#             'sentiment_bin': sample_bin,
#             'padding_analysis': {
#                 'audio': audio_padding,
#                 'video': video_padding,
#                 'text': text_padding
#             }
#         }
        
#         batch_analysis['samples'].append(sample_analysis)
        
#         # Aggregate statistics by sentiment bin
#         batch_analysis['by_bin_stats'][sample_bin].append({
#             'has_audio_padding': audio_padding['has_padding'],
#             'has_video_padding': video_padding['has_padding'],
#             'has_text_padding': text_padding['has_padding'],
#             'audio_padding_ratio': audio_padding['padding_ratio'],
#             'video_padding_ratio': video_padding['padding_ratio'],
#             'text_padding_ratio': text_padding['padding_ratio'],
#             'any_modality_padding': any([
#                 audio_padding['has_padding'],
#                 video_padding['has_padding'], 
#                 text_padding['has_padding']
#             ]),
#             'all_modalities_padding': all([
#                 audio_padding['has_padding'],
#                 video_padding['has_padding'],
#                 text_padding['has_padding']
#             ])
#         })
        
#         # Aggregate by modality and bin
#         for modality, padding_info in [('audio', audio_padding), ('video', video_padding), ('text', text_padding)]:
#             batch_analysis['modality_stats'][modality][sample_bin].append(padding_info)
    
#     return batch_analysis

# def analyze_complete_split(dataloader, split_name, dataset_name, num_bins=7, tolerance=1e-7, max_batches=None):
#     """
#     Analyze padding for a complete dataset split.
    
#     Args:
#         dataloader: PyTorch DataLoader
#         split_name: Split name
#         dataset_name: Dataset name
#         num_bins: Number of sentiment bins
#         tolerance: Tolerance for padding detection
#         max_batches: Maximum number of batches to process (None for all)
        
#     Returns:
#         split_analysis: Complete analysis for the split
#     """
#     print_analysis_progress(f"Analyzing {dataset_name} {split_name} split...")
    
#     split_analysis = {
#         'split_info': {
#             'split_name': split_name,
#             'dataset_name': dataset_name,
#             'total_batches': len(dataloader),
#             'total_samples': len(dataloader.dataset)
#         },
#         'batches': [],
#         'aggregate_stats': {
#             'by_bin': defaultdict(lambda: {
#                 'sample_count': 0,
#                 'samples_with_padding': 0,
#                 'samples_without_padding': 0,
#                 'total_padding_ratio': {'audio': 0, 'video': 0, 'text': 0},
#                 'samples_with_any_padding': 0,
#                 'samples_with_all_padding': 0
#             }),
#             'overall': {
#                 'total_samples': 0,
#                 'samples_with_padding': 0,
#                 'samples_without_padding': 0,
#                 'average_padding_ratio': {'audio': 0, 'video': 0, 'text': 0}
#             }
#         }
#     }
    
#     batches_processed = 0
    
#     for batch_idx, batch_data in enumerate(dataloader):
#         if max_batches is not None and batches_processed >= max_batches:
#             print_analysis_progress(f"Reached max_batches limit ({max_batches}), stopping...")
#             break
            
#         batch_analysis = analyze_batch_padding(
#             batch_data, batch_idx, split_name, dataset_name, num_bins, tolerance
#         )
        
#         split_analysis['batches'].append(batch_analysis)
        
#         # Aggregate statistics
#         for sample in batch_analysis['samples']:
#             bin_id = sample['sentiment_bin']
#             audio_padding = sample['padding_analysis']['audio']
#             video_padding = sample['padding_analysis']['video']
#             text_padding = sample['padding_analysis']['text']
            
#             # Update bin-specific stats
#             bin_stats = split_analysis['aggregate_stats']['by_bin'][bin_id]
#             bin_stats['sample_count'] += 1
#             bin_stats['total_padding_ratio']['audio'] += audio_padding['padding_ratio']
#             bin_stats['total_padding_ratio']['video'] += video_padding['padding_ratio']
#             bin_stats['total_padding_ratio']['text'] += text_padding['padding_ratio']
            
#             has_any_padding = any([
#                 audio_padding['has_padding'],
#                 video_padding['has_padding'],
#                 text_padding['has_padding']
#             ])
            
#             if has_any_padding:
#                 bin_stats['samples_with_padding'] += 1
#                 bin_stats['samples_with_any_padding'] += 1
#             else:
#                 bin_stats['samples_without_padding'] += 1
                
#             if all([audio_padding['has_padding'], video_padding['has_padding'], text_padding['has_padding']]):
#                 bin_stats['samples_with_all_padding'] += 1
            
#             # Update overall stats
#             overall = split_analysis['aggregate_stats']['overall']
#             overall['total_samples'] += 1
#             overall['average_padding_ratio']['audio'] += audio_padding['padding_ratio']
#             overall['average_padding_ratio']['video'] += video_padding['padding_ratio']
#             overall['average_padding_ratio']['text'] += text_padding['padding_ratio']
            
#             if has_any_padding:
#                 overall['samples_with_padding'] += 1
#             else:
#                 overall['samples_without_padding'] += 1
        
#         batches_processed += 1
        
#         if batch_idx % 10 == 0:
#             print_analysis_progress(f"Processed batch {batch_idx}/{len(dataloader)} ({batches_processed} total)")
    
#     # Calculate final averages
#     for bin_id, bin_stats in split_analysis['aggregate_stats']['by_bin'].items():
#         if bin_stats['sample_count'] > 0:
#             for modality in ['audio', 'video', 'text']:
#                 bin_stats['total_padding_ratio'][modality] /= bin_stats['sample_count']
    
#     overall = split_analysis['aggregate_stats']['overall']
#     if overall['total_samples'] > 0:
#         for modality in ['audio', 'video', 'text']:
#             overall['average_padding_ratio'][modality] /= overall['total_samples']
    
#     print_analysis_progress(f"Completed analysis of {dataset_name} {split_name}: {overall['total_samples']} samples")
    
#     return split_analysis

# def generate_summary_report(all_analyses, output_dir):
#     """
#     Generate comprehensive summary reports and statistics.
    
#     Args:
#         all_analyses: Dictionary containing all analysis results
#         output_dir: Directory to save reports
#     """
#     print_analysis_progress("Generating comprehensive summary reports...")
    
#     # Create summary statistics CSV
#     csv_data = []
    
#     for dataset_name, dataset_analyses in all_analyses.items():
#         for split_name, split_analysis in dataset_analyses.items():
#             overall_stats = split_analysis['aggregate_stats']['overall']
            
#             # Overall statistics
#             csv_data.append({
#                 'Dataset': dataset_name,
#                 'Split': split_name,
#                 'Bin': 'ALL',
#                 'Total_Samples': overall_stats['total_samples'],
#                 'Samples_With_Padding': overall_stats['samples_with_padding'],
#                 'Samples_Without_Padding': overall_stats['samples_without_padding'],
#                 'Percent_With_Padding': (overall_stats['samples_with_padding'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0,
#                 'Percent_Without_Padding': (overall_stats['samples_without_padding'] / overall_stats['total_samples'] * 100) if overall_stats['total_samples'] > 0 else 0,
#                 'Avg_Audio_Padding_Ratio': overall_stats['average_padding_ratio']['audio'],
#                 'Avg_Video_Padding_Ratio': overall_stats['average_padding_ratio']['video'],
#                 'Avg_Text_Padding_Ratio': overall_stats['average_padding_ratio']['text']
#             })
            
#             # Per-bin statistics
#             for bin_id, bin_stats in split_analysis['aggregate_stats']['by_bin'].items():
#                 if bin_stats['sample_count'] > 0:
#                     csv_data.append({
#                         'Dataset': dataset_name,
#                         'Split': split_name,
#                         'Bin': bin_id,
#                         'Total_Samples': bin_stats['sample_count'],
#                         'Samples_With_Padding': bin_stats['samples_with_padding'],
#                         'Samples_Without_Padding': bin_stats['samples_without_padding'],
#                         'Percent_With_Padding': (bin_stats['samples_with_padding'] / bin_stats['sample_count'] * 100),
#                         'Percent_Without_Padding': (bin_stats['samples_without_padding'] / bin_stats['sample_count'] * 100),
#                         'Avg_Audio_Padding_Ratio': bin_stats['total_padding_ratio']['audio'],
#                         'Avg_Video_Padding_Ratio': bin_stats['total_padding_ratio']['video'],
#                         'Avg_Text_Padding_Ratio': bin_stats['total_padding_ratio']['text']
#                     })
    
#     # Save CSV
#     csv_path = os.path.join(output_dir, 'padding_analysis_summary.csv')
#     with open(csv_path, 'w', newline='') as csvfile:
#         if csv_data:
#             fieldnames = csv_data[0].keys()
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(csv_data)
    
#     print_analysis_progress(f"CSV summary saved to: {csv_path}")
    
#     # Generate detailed text report
#     report_path = os.path.join(output_dir, 'padding_analysis_detailed_report.txt')
#     with open(report_path, 'w') as f:
#         f.write("COMPREHENSIVE PADDING ANALYSIS REPORT\n")
#         f.write("=" * 80 + "\n\n")
#         f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
#         for dataset_name, dataset_analyses in all_analyses.items():
#             f.write(f"\n{dataset_name.upper()} DATASET ANALYSIS\n")
#             f.write("-" * 50 + "\n")
            
#             for split_name, split_analysis in dataset_analyses.items():
#                 f.write(f"\n{split_name.upper()} Split:\n")
                
#                 overall = split_analysis['aggregate_stats']['overall']
#                 f.write(f"  Total samples: {overall['total_samples']}\n")
#                 f.write(f"  Samples with padding: {overall['samples_with_padding']} ({overall['samples_with_padding']/overall['total_samples']*100:.1f}%)\n")
#                 f.write(f"  Samples without padding: {overall['samples_without_padding']} ({overall['samples_without_padding']/overall['total_samples']*100:.1f}%)\n")
#                 f.write(f"  Average padding ratios:\n")
#                 f.write(f"    Audio: {overall['average_padding_ratio']['audio']:.3f}\n")
#                 f.write(f"    Video: {overall['average_padding_ratio']['video']:.3f}\n")
#                 f.write(f"    Text: {overall['average_padding_ratio']['text']:.3f}\n")
                
#                 f.write(f"\n  Sentiment Bin Analysis:\n")
#                 for bin_id in sorted(split_analysis['aggregate_stats']['by_bin'].keys()):
#                     bin_stats = split_analysis['aggregate_stats']['by_bin'][bin_id]
#                     if bin_stats['sample_count'] > 0:
#                         f.write(f"    Bin {bin_id}: {bin_stats['sample_count']} samples\n")
#                         f.write(f"      With padding: {bin_stats['samples_with_padding']} ({bin_stats['samples_with_padding']/bin_stats['sample_count']*100:.1f}%)\n")
#                         f.write(f"      Without padding: {bin_stats['samples_without_padding']} ({bin_stats['samples_without_padding']/bin_stats['sample_count']*100:.1f}%)\n")
#                         f.write(f"      Avg padding ratios - Audio: {bin_stats['total_padding_ratio']['audio']:.3f}, ")
#                         f.write(f"Video: {bin_stats['total_padding_ratio']['video']:.3f}, ")
#                         f.write(f"Text: {bin_stats['total_padding_ratio']['text']:.3f}\n")
                
#                 f.write("\n")
    
#     print_analysis_progress(f"Detailed report saved to: {report_path}")
    
#     # Save complete analysis as JSON
#     json_path = os.path.join(output_dir, 'complete_padding_analysis.json')
    
#     # Convert numpy types to native Python types for JSON serialization
#     def convert_for_json(obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, dict):
#             return {key: convert_for_json(value) for key, value in obj.items()}
#         elif isinstance(obj, list):
#             return [convert_for_json(item) for item in obj]
#         elif isinstance(obj, defaultdict):
#             return dict(obj)
#         else:
#             return obj
    
#     json_serializable = convert_for_json(all_analyses)
    
#     with open(json_path, 'w') as f:
#         json.dump(json_serializable, f, indent=2, default=str)
    
#     print_analysis_progress(f"Complete analysis JSON saved to: {json_path}")

# def main():
#     """
#     Main function to run comprehensive padding analysis.
#     """
#     print_analysis_progress("Starting comprehensive padding analysis for MOSEI and MOSI datasets")
    
#     # Configuration
#     num_bins = 7
#     tolerance = 1e-7
#     batch_size = 64
#     max_seq_len = 10
#     num_workers = 8
#     max_batches = None  # Set to a number to limit analysis for testing
    
#     # Create output directory
#     output_dir = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Results/padding_analysis_10"
#     os.makedirs(output_dir, exist_ok=True)
    
#     print_analysis_progress(f"Results will be saved to: {output_dir}")
#     print_analysis_progress(f"Configuration: bins={num_bins}, tolerance={tolerance}, max_batches={max_batches}")
    
#     all_analyses = {}
    
#     # Analyze MOSEI dataset
#     print_analysis_progress("\n" + "="*60)
#     print_analysis_progress("ANALYZING MOSEI DATASET")
#     print_analysis_progress("="*60)
    
#     try:
#         # MOSEI supervised splits
#         train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(
#             batch_size=batch_size, max_seq_len=max_seq_len, num_workers=num_workers
#         )
        
#         mosei_analyses = {}
#         mosei_analyses['train'] = analyze_complete_split(
#             train_loader, 'train', 'MOSEI', num_bins, tolerance, max_batches
#         )
#         mosei_analyses['val'] = analyze_complete_split(
#             val_loader, 'val', 'MOSEI', num_bins, tolerance, max_batches  
#         )
#         mosei_analyses['test'] = analyze_complete_split(
#             test_loader, 'test', 'MOSEI', num_bins, tolerance, max_batches
#         )
        
#         all_analyses['MOSEI'] = mosei_analyses
#         print_analysis_progress("✅ MOSEI analysis completed successfully!")
        
#     except Exception as e:
#         print_analysis_progress(f"❌ Error analyzing MOSEI: {e}")
#         import traceback
#         print_analysis_progress(f"Traceback: {traceback.format_exc()}")
    
#     # Analyze MOSI dataset
#     print_analysis_progress("\n" + "="*60)
#     print_analysis_progress("ANALYZING MOSI DATASET")
#     print_analysis_progress("="*60)
    
#     try:
#         mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(
#             batch_size=batch_size, max_seq_len=max_seq_len, num_workers=num_workers
#         )
        
#         mosi_analyses = {}
#         mosi_analyses['train'] = analyze_complete_split(
#             mosi_train, 'train', 'MOSI', num_bins, tolerance, max_batches
#         )
#         mosi_analyses['val'] = analyze_complete_split(
#             mosi_val, 'val', 'MOSI', num_bins, tolerance, max_batches
#         )
#         mosi_analyses['test'] = analyze_complete_split(
#             mosi_test, 'test', 'MOSI', num_bins, tolerance, max_batches
#         )
        
#         all_analyses['MOSI'] = mosi_analyses
#         print_analysis_progress("✅ MOSI analysis completed successfully!")
        
#     except Exception as e:
#         print_analysis_progress(f"❌ Error analyzing MOSI: {e}")
#         import traceback
#         print_analysis_progress(f"Traceback: {traceback.format_exc()}")
    
#     # Generate summary reports
#     if all_analyses:
#         print_analysis_progress("\n" + "="*60)
#         print_analysis_progress("GENERATING SUMMARY REPORTS")
#         print_analysis_progress("="*60)
        
#         generate_summary_report(all_analyses, output_dir)
        
#         # Print quick summary to console
#         print_analysis_progress("\n" + "="*60)
#         print_analysis_progress("QUICK SUMMARY")
#         print_analysis_progress("="*60)
        
#         for dataset_name, dataset_analyses in all_analyses.items():
#             print_analysis_progress(f"\n{dataset_name}:")
#             for split_name, split_analysis in dataset_analyses.items():
#                 overall = split_analysis['aggregate_stats']['overall']
#                 print_analysis_progress(f"  {split_name}: {overall['total_samples']} samples, "
#                                       f"{overall['samples_without_padding']} ({overall['samples_without_padding']/overall['total_samples']*100:.1f}%) without padding")
    
#     print_analysis_progress("\n" + "="*60)
#     print_analysis_progress("ANALYSIS COMPLETE!")
#     print_analysis_progress(f"Results saved to: {output_dir}")
#     print_analysis_progress("="*60)

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Content Length Analysis for MOSEI and MOSI Datasets
Calculates average content length (unpadded timesteps) for each modality 
in each sentiment bin per split.
"""

import os
import sys
import torch
import numpy as np
import csv
from datetime import datetime
from collections import defaultdict

# Add your project paths
sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/dataset")
sys.path.append("/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src")

from dataset.MOSEI_MOSI import (
    get_mosei_supervised_dataloaders,
    get_mosi_dataloaders,
    get_mosei_unsupervised_split_dataloaders
)

def print_analysis_progress(message):
    """Helper function for logging analysis progress"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    return timestamp

def discretize_sentiment_labels(labels, num_bins=7):
    """
    Discretize continuous sentiment labels into bins.
    """
    # Convert to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels_np = labels.numpy()
    else:
        labels_np = np.array(labels)
    
    # Flatten if needed
    if len(labels_np.shape) > 1:
        labels_np = labels_np.flatten()
    
    # Create bins from min to max label values
    min_label = np.min(labels_np)
    max_label = np.max(labels_np)
    bin_edges = np.linspace(min_label, max_label, num_bins + 1)
    
    # Discretize labels
    discrete_labels = np.digitize(labels_np, bin_edges[1:-1])
    
    return discrete_labels

def count_unpadded_timesteps(features, tolerance=1e-7):
    """
    Count the number of unpadded timesteps in a sequence.
    
    Args:
        features: Feature array of shape [seq_len, feat_dim]
        tolerance: Tolerance for identifying zero-padded timesteps
        
    Returns:
        unpadded_count: Number of unpadded timesteps
    """
    seq_len = features.shape[0]
    unpadded_count = 0
    
    for t in range(seq_len):
        if not np.allclose(features[t], 0, atol=tolerance):
            unpadded_count += 1
    
    return unpadded_count

def analyze_split_content_lengths(dataloader, split_name, dataset_name, num_bins=7, tolerance=1e-7, max_batches=None):
    """
    Analyze content lengths for a complete dataset split.
    
    Returns:
        content_lengths_by_bin: Dictionary with average content lengths per bin
    """
    print_analysis_progress(f"Analyzing content lengths for {dataset_name} {split_name} split...")
    
    # Store content lengths grouped by sentiment bin
    content_lengths_by_bin = defaultdict(lambda: {
        'audio_lengths': [],
        'video_lengths': [], 
        'text_lengths': []
    })
    
    batches_processed = 0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        if max_batches is not None and batches_processed >= max_batches:
            print_analysis_progress(f"Reached max_batches limit ({max_batches}), stopping...")
            break
            
        audio_features = batch_data['audio_fea'].numpy()  # [batch_size, seq_len, 74]
        video_features = batch_data['video_fea'].numpy()  # [batch_size, seq_len, 35]
        text_features = batch_data['text_fea'].numpy()    # [batch_size, seq_len, 300]
        labels = batch_data['labels'].numpy()             # [batch_size, 1]
        
        batch_size = audio_features.shape[0]
        
        # Discretize sentiment labels
        discrete_labels = discretize_sentiment_labels(labels, num_bins)
        
        # Process each sample in the batch
        for i in range(batch_size):
            sentiment_bin = discrete_labels[i]
            
            # Count unpadded timesteps for each modality
            audio_unpadded = count_unpadded_timesteps(audio_features[i], tolerance)
            video_unpadded = count_unpadded_timesteps(video_features[i], tolerance)
            text_unpadded = count_unpadded_timesteps(text_features[i], tolerance)
            
            # Store in the appropriate bin
            content_lengths_by_bin[sentiment_bin]['audio_lengths'].append(audio_unpadded)
            content_lengths_by_bin[sentiment_bin]['video_lengths'].append(video_unpadded)
            content_lengths_by_bin[sentiment_bin]['text_lengths'].append(text_unpadded)
            
            total_samples += 1
        
        batches_processed += 1
        
        if batch_idx % 10 == 0:
            print_analysis_progress(f"Processed batch {batch_idx}/{len(dataloader)} ({batches_processed} total)")
    
    print_analysis_progress(f"Completed analysis: {total_samples} samples processed")
    return content_lengths_by_bin

def calculate_average_content_lengths(content_lengths_by_bin):
    """
    Calculate average content lengths from the collected data.
    
    Returns:
        avg_lengths: Dictionary with average lengths per bin
    """
    avg_lengths = {}
    
    for bin_id, lengths in content_lengths_by_bin.items():
        avg_lengths[bin_id] = {
            'sample_count': len(lengths['audio_lengths']),
            'audio_avg': np.mean(lengths['audio_lengths']) if lengths['audio_lengths'] else 0.0,
            'video_avg': np.mean(lengths['video_lengths']) if lengths['video_lengths'] else 0.0,
            'text_avg': np.mean(lengths['text_lengths']) if lengths['text_lengths'] else 0.0
        }
    
    return avg_lengths

def display_content_length_summary(all_results):
    """
    Display a formatted summary of average content lengths.
    """
    print_analysis_progress("\n" + "="*80)
    print_analysis_progress("AVERAGE UNPADDED TIMESTEPS PER SEQUENCE BY MODALITY AND SENTIMENT BIN")
    print_analysis_progress("="*80)
    
    for dataset_name, dataset_results in all_results.items():
        print_analysis_progress(f"\n{dataset_name.upper()} DATASET:")
        print_analysis_progress("-" * 60)
        
        for split_name, avg_lengths in dataset_results.items():
            print_analysis_progress(f"\n  {split_name.upper()} Split:")
            print_analysis_progress(f"    Bin | Samples |  Audio Avg  |  Video Avg  |  Text Avg   ")
            print_analysis_progress(f"    ----|---------|-------------|-------------|-------------|")
            
            # Calculate overall averages across all bins
            total_samples = sum(stats['sample_count'] for stats in avg_lengths.values())
            
            if total_samples > 0:
                # Weight averages by sample count
                weighted_audio = sum(stats['audio_avg'] * stats['sample_count'] for stats in avg_lengths.values()) / total_samples
                weighted_video = sum(stats['video_avg'] * stats['sample_count'] for stats in avg_lengths.values()) / total_samples  
                weighted_text = sum(stats['text_avg'] * stats['sample_count'] for stats in avg_lengths.values()) / total_samples
                
                print_analysis_progress(
                    f"    ALL | {total_samples:7d} | {weighted_audio:11.2f} | {weighted_video:11.2f} | {weighted_text:11.2f} |"
                )
                print_analysis_progress(f"    ----|---------|-------------|-------------|-------------|")
            
            # Display per-bin statistics
            for bin_id in sorted(avg_lengths.keys()):
                stats = avg_lengths[bin_id]
                print_analysis_progress(
                    f"    {bin_id:3d} | {stats['sample_count']:7d} | {stats['audio_avg']:11.2f} | {stats['video_avg']:11.2f} | {stats['text_avg']:11.2f} |"
                )

def save_content_length_results(all_results, output_dir):
    """
    Save content length analysis results to CSV.
    """
    print_analysis_progress("Saving content length analysis results...")
    
    csv_data = []
    
    for dataset_name, dataset_results in all_results.items():
        for split_name, avg_lengths in dataset_results.items():
            
            # Overall averages
            total_samples = sum(stats['sample_count'] for stats in avg_lengths.values())
            if total_samples > 0:
                weighted_audio = sum(stats['audio_avg'] * stats['sample_count'] for stats in avg_lengths.values()) / total_samples
                weighted_video = sum(stats['video_avg'] * stats['sample_count'] for stats in avg_lengths.values()) / total_samples
                weighted_text = sum(stats['text_avg'] * stats['sample_count'] for stats in avg_lengths.values()) / total_samples
                
                csv_data.append({
                    'Dataset': dataset_name,
                    'Split': split_name,
                    'Sentiment_Bin': 'ALL',
                    'Sample_Count': total_samples,
                    'Audio_Avg_Unpadded_Timesteps': round(weighted_audio, 2),
                    'Video_Avg_Unpadded_Timesteps': round(weighted_video, 2),
                    'Text_Avg_Unpadded_Timesteps': round(weighted_text, 2)
                })
            
            # Per-bin averages
            for bin_id in sorted(avg_lengths.keys()):
                stats = avg_lengths[bin_id]
                csv_data.append({
                    'Dataset': dataset_name,
                    'Split': split_name,
                    'Sentiment_Bin': bin_id,
                    'Sample_Count': stats['sample_count'],
                    'Audio_Avg_Unpadded_Timesteps': round(stats['audio_avg'], 2),
                    'Video_Avg_Unpadded_Timesteps': round(stats['video_avg'], 2),
                    'Text_Avg_Unpadded_Timesteps': round(stats['text_avg'], 2)
                })
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'average_content_lengths.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        if csv_data:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
    
    print_analysis_progress(f"Results saved to: {csv_path}")

def main():
    """
    Main function to run content length analysis.
    """
    print_analysis_progress("Starting average content length analysis for MOSEI and MOSI datasets")
    
    # Configuration
    num_bins = 7
    tolerance = 1e-7
    batch_size =128
    max_seq_len = 50
    num_workers = 8
    max_batches = None  # Set to a number to limit analysis for testing
    
    # Create output directory
    output_dir = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Results/content_length_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print_analysis_progress(f"Results will be saved to: {output_dir}")
    print_analysis_progress(f"Configuration: bins={num_bins}, tolerance={tolerance}, max_batches={max_batches}")
    
    all_results = {}
    
    # Analyze MOSEI dataset
    print_analysis_progress("\n" + "="*60)
    print_analysis_progress("ANALYZING MOSEI DATASET")
    print_analysis_progress("="*60)
    
    try:
        # MOSEI supervised splits
        train_loader, val_loader, test_loader = get_mosei_supervised_dataloaders(
            batch_size=batch_size, max_seq_len=max_seq_len, num_workers=num_workers
        )
        
        mosei_train_content = analyze_split_content_lengths(
            train_loader, 'train', 'MOSEI', num_bins, tolerance, max_batches
        )
        mosei_val_content = analyze_split_content_lengths(
            val_loader, 'val', 'MOSEI', num_bins, tolerance, max_batches  
        )
        mosei_test_content = analyze_split_content_lengths(
            test_loader, 'test', 'MOSEI', num_bins, tolerance, max_batches
        )
        
        all_results['MOSEI'] = {
            'train': calculate_average_content_lengths(mosei_train_content),
            'val': calculate_average_content_lengths(mosei_val_content),
            'test': calculate_average_content_lengths(mosei_test_content)
        }
        
        print_analysis_progress("✅ MOSEI analysis completed successfully!")
        
    except Exception as e:
        print_analysis_progress(f"❌ Error analyzing MOSEI: {e}")
        import traceback
        print_analysis_progress(f"Traceback: {traceback.format_exc()}")
    
    # Analyze MOSI dataset
    print_analysis_progress("\n" + "="*60)
    print_analysis_progress("ANALYZING MOSI DATASET")
    print_analysis_progress("="*60)
    
    try:
        mosi_train, mosi_val, mosi_test = get_mosi_dataloaders(
            batch_size=batch_size, max_seq_len=max_seq_len, num_workers=num_workers
        )
        
        mosi_train_content = analyze_split_content_lengths(
            mosi_train, 'train', 'MOSI', num_bins, tolerance, max_batches
        )
        mosi_val_content = analyze_split_content_lengths(
            mosi_val, 'val', 'MOSI', num_bins, tolerance, max_batches
        )
        mosi_test_content = analyze_split_content_lengths(
            mosi_test, 'test', 'MOSI', num_bins, tolerance, max_batches
        )
        
        all_results['MOSI'] = {
            'train': calculate_average_content_lengths(mosi_train_content),
            'val': calculate_average_content_lengths(mosi_val_content),
            'test': calculate_average_content_lengths(mosi_test_content)
        }
        
        print_analysis_progress("✅ MOSI analysis completed successfully!")
        
    except Exception as e:
        print_analysis_progress(f"❌ Error analyzing MOSI: {e}")
        import traceback
        print_analysis_progress(f"Traceback: {traceback.format_exc()}")
    
    # Display and save results
    if all_results:
        print_analysis_progress("\n" + "="*60)
        print_analysis_progress("GENERATING RESULTS")
        print_analysis_progress("="*60)
        
        display_content_length_summary(all_results)
        save_content_length_results(all_results, output_dir)
        
        # Print quick summary
        print_analysis_progress("\n" + "="*60)
        print_analysis_progress("ANALYSIS COMPLETE!")
        print_analysis_progress(f"Results saved to: {output_dir}")
        print_analysis_progress("="*60)

if __name__ == "__main__":
    main()