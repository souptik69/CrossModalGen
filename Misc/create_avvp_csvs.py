import pandas as pd
import numpy as np
from collections import defaultdict
import os

def parse_onset_offset(onset_str, offset_str):
    """Parse onset and offset strings that may contain comma-separated values"""
    if ',' in str(onset_str):
        onsets = [int(x) for x in str(onset_str).split(',')]
    else:
        onsets = [int(onset_str)]
    
    if ',' in str(offset_str):
        offsets = [int(x) for x in str(offset_str).split(',')]
    else:
        offsets = [int(offset_str)]
    
    return onsets, offsets

def parse_event_labels(labels_str):
    """Parse comma-separated event labels"""
    if pd.isna(labels_str):
        return []
    return [label.strip() for label in str(labels_str).split(',')]

def create_temporal_segments(onsets, offsets, labels):
    """Create individual temporal segments from multi-segment entries"""
    segments = []
    
    if len(onsets) != len(offsets):
        # If lengths don't match, pad the shorter one
        max_len = max(len(onsets), len(offsets))
        if len(onsets) < max_len:
            onsets.extend([onsets[-1]] * (max_len - len(onsets)))
        if len(offsets) < max_len:
            offsets.extend([offsets[-1]] * (max_len - len(offsets)))
    
    # Ensure we have corresponding labels for each segment
    if len(labels) < len(onsets):
        # Repeat the first label if we don't have enough
        labels.extend([labels[0]] * (len(onsets) - len(labels)))
    elif len(labels) > len(onsets):
        # Take only the first few labels if we have too many
        labels = labels[:len(onsets)]
    
    for i, (onset, offset) in enumerate(zip(onsets, offsets)):
        label = labels[i] if i < len(labels) else labels[0]
        segments.append({
            'onset': onset,
            'offset': offset,
            'label': label
        })
    
    return segments

def merge_temporal_annotations(visual_segments, audio_segments):
    """Merge visual and audio temporal annotations"""
    # Create a unified timeline
    all_times = set()
    
    # Collect all unique time points
    for seg in visual_segments:
        all_times.add(seg['onset'])
        all_times.add(seg['offset'])
    
    for seg in audio_segments:
        all_times.add(seg['onset'])
        all_times.add(seg['offset'])
    
    all_times = sorted(list(all_times))
    
    # Create merged segments
    merged_segments = []
    
    for i in range(len(all_times) - 1):
        start_time = all_times[i]
        end_time = all_times[i + 1]
        
        # Find overlapping visual and audio labels
        visual_labels = []
        audio_labels = []
        
        for seg in visual_segments:
            if seg['onset'] <= start_time and seg['offset'] >= end_time:
                visual_labels.append(seg['label'])
        
        for seg in audio_segments:
            if seg['onset'] <= start_time and seg['offset'] >= end_time:
                audio_labels.append(seg['label'])
        
        # Combine unique labels
        all_labels = list(set(visual_labels + audio_labels))
        
        if all_labels:  # Only add if there are labels
            merged_segments.append({
                'onset': start_time,
                'offset': end_time,
                'labels': all_labels,
                'visual_labels': visual_labels,
                'audio_labels': audio_labels
            })
    
    return merged_segments

def create_multimodal_csv(visual_df, audio_df, output_path, strategy='intersection'):
    """
    Create multimodal CSV combining visual and audio annotations
    
    Args:
        visual_df: DataFrame with visual annotations
        audio_df: DataFrame with audio annotations
        output_path: Path to save the combined CSV
        strategy: 'intersection' (only common files) or 'union' (all files)
    """
    
    # Find common filenames
    visual_filenames = set(visual_df['filename'])
    audio_filenames = set(audio_df['filename'])
    
    if strategy == 'intersection':
        target_filenames = visual_filenames.intersection(audio_filenames)
        print(f"Processing {len(target_filenames)} common files")
    elif strategy == 'union':
        target_filenames = visual_filenames.union(audio_filenames)
        print(f"Processing {len(target_filenames)} total files")
    else:
        raise ValueError("Strategy must be 'intersection' or 'union'")
    
    combined_entries = []
    
    for filename in target_filenames:
        # Get visual and audio entries
        visual_entries = visual_df[visual_df['filename'] == filename]
        audio_entries = audio_df[audio_df['filename'] == filename]
        
        if strategy == 'intersection':
            # Both must exist
            if len(visual_entries) == 0 or len(audio_entries) == 0:
                continue
                
            visual_entry = visual_entries.iloc[0]
            audio_entry = audio_entries.iloc[0]
            
            # Parse temporal information
            visual_onsets, visual_offsets = parse_onset_offset(visual_entry['onset'], visual_entry['offset'])
            audio_onsets, audio_offsets = parse_onset_offset(audio_entry['onset'], audio_entry['offset'])
            
            visual_labels = parse_event_labels(visual_entry['event_labels'])
            audio_labels = parse_event_labels(audio_entry['event_labels'])
            
            # Create segments
            visual_segments = create_temporal_segments(visual_onsets, visual_offsets, visual_labels)
            audio_segments = create_temporal_segments(audio_onsets, audio_offsets, audio_labels)
            
            # Merge annotations
            merged_segments = merge_temporal_annotations(visual_segments, audio_segments)
            
            # Create entries for each merged segment
            for seg in merged_segments:
                combined_entries.append({
                    'filename': filename,
                    'onset': seg['onset'],
                    'offset': seg['offset'],
                    'event_labels': ','.join(seg['labels']),
                    'visual_labels': ','.join(seg['visual_labels']) if seg['visual_labels'] else '',
                    'audio_labels': ','.join(seg['audio_labels']) if seg['audio_labels'] else '',
                    'modality': 'multimodal',
                    'has_visual': len(seg['visual_labels']) > 0,
                    'has_audio': len(seg['audio_labels']) > 0
                })
                
        elif strategy == 'union':
            # Handle cases where only one modality exists
            if len(visual_entries) > 0 and len(audio_entries) > 0:
                # Both exist - same as intersection case
                visual_entry = visual_entries.iloc[0]
                audio_entry = audio_entries.iloc[0]
                
                visual_onsets, visual_offsets = parse_onset_offset(visual_entry['onset'], visual_entry['offset'])
                audio_onsets, audio_offsets = parse_onset_offset(audio_entry['onset'], audio_entry['offset'])
                
                visual_labels = parse_event_labels(visual_entry['event_labels'])
                audio_labels = parse_event_labels(audio_entry['event_labels'])
                
                visual_segments = create_temporal_segments(visual_onsets, visual_offsets, visual_labels)
                audio_segments = create_temporal_segments(audio_onsets, audio_offsets, audio_labels)
                
                merged_segments = merge_temporal_annotations(visual_segments, audio_segments)
                
                for seg in merged_segments:
                    combined_entries.append({
                        'filename': filename,
                        'onset': seg['onset'],
                        'offset': seg['offset'],
                        'event_labels': ','.join(seg['labels']),
                        'visual_labels': ','.join(seg['visual_labels']) if seg['visual_labels'] else '',
                        'audio_labels': ','.join(seg['audio_labels']) if seg['audio_labels'] else '',
                        'modality': 'multimodal',
                        'has_visual': len(seg['visual_labels']) > 0,
                        'has_audio': len(seg['audio_labels']) > 0
                    })
                    
            elif len(visual_entries) > 0:
                # Only visual exists
                visual_entry = visual_entries.iloc[0]
                visual_onsets, visual_offsets = parse_onset_offset(visual_entry['onset'], visual_entry['offset'])
                visual_labels = parse_event_labels(visual_entry['event_labels'])
                visual_segments = create_temporal_segments(visual_onsets, visual_offsets, visual_labels)
                
                for seg in visual_segments:
                    combined_entries.append({
                        'filename': filename,
                        'onset': seg['onset'],
                        'offset': seg['offset'],
                        'event_labels': seg['label'],
                        'visual_labels': seg['label'],
                        'audio_labels': '',
                        'modality': 'visual_only',
                        'has_visual': True,
                        'has_audio': False
                    })
                    
            elif len(audio_entries) > 0:
                # Only audio exists
                audio_entry = audio_entries.iloc[0]
                audio_onsets, audio_offsets = parse_onset_offset(audio_entry['onset'], audio_entry['offset'])
                audio_labels = parse_event_labels(audio_entry['event_labels'])
                audio_segments = create_temporal_segments(audio_onsets, audio_offsets, audio_labels)
                
                for seg in audio_segments:
                    combined_entries.append({
                        'filename': filename,
                        'onset': seg['onset'],
                        'offset': seg['offset'],
                        'event_labels': seg['label'],
                        'visual_labels': '',
                        'audio_labels': seg['label'],
                        'modality': 'audio_only',
                        'has_visual': False,
                        'has_audio': True
                    })
    
    # Create DataFrame and save
    combined_df = pd.DataFrame(combined_entries)
    combined_df = combined_df.sort_values(['filename', 'onset'])
    combined_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Saved combined CSV with {len(combined_df)} entries to {output_path}")
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total entries: {len(combined_df)}")
    print(f"Multimodal entries: {len(combined_df[combined_df['modality'] == 'multimodal'])}")
    print(f"Visual-only entries: {len(combined_df[combined_df['modality'] == 'visual_only'])}")
    print(f"Audio-only entries: {len(combined_df[combined_df['modality'] == 'audio_only'])}")
    
    return combined_df

def create_simplified_multimodal_csv(visual_df, audio_df, output_path):
    """
    Create a simplified multimodal CSV that keeps the original format
    but only includes files that exist in both datasets
    """
    # Find common filenames
    visual_filenames = set(visual_df['filename'])
    audio_filenames = set(audio_df['filename'])
    common_filenames = visual_filenames.intersection(audio_filenames)
    
    print(f"Found {len(common_filenames)} common files")
    
    # Filter both datasets to only include common files
    visual_common = visual_df[visual_df['filename'].isin(common_filenames)].copy()
    audio_common = audio_df[audio_df['filename'].isin(common_filenames)].copy()
    
    # Add modality column
    visual_common['modality'] = 'visual'
    audio_common['modality'] = 'audio'
    
    # Combine both datasets
    combined_df = pd.concat([visual_common, audio_common], ignore_index=True)
    combined_df = combined_df.sort_values(['filename', 'modality'])
    
    # Save
    combined_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Saved simplified combined CSV with {len(combined_df)} entries to {output_path}")
    print(f"Visual entries: {len(visual_common)}")
    print(f"Audio entries: {len(audio_common)}")
    
    return combined_df

# Main execution
if __name__ == "__main__":
    # Setup output directory and logging
    from datetime import datetime
    
    output_dir = "multimodal_csv_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup summary file
    summary_file = os.path.join(output_dir, f"csv_creation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    class Tee:
        """Class to write to both file and console simultaneously"""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Open summary file and setup dual output
    import sys
    summary_f = open(summary_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, summary_f)
    
    print("="*80)
    print("MULTIMODAL CSV CREATION REPORT")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load the datasets with correct paths
    visual_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_eval_visual_checked_combined.csv'
    audio_csv_path = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_eval_audio_checked_combined.csv'
    
    print(f"Loading visual data from: {visual_csv_path}")
    print(f"Loading audio data from: {audio_csv_path}")
    
    try:
        visual_df = pd.read_csv(visual_csv_path, sep='\t')
        audio_df = pd.read_csv(audio_csv_path, sep='\t')
        print("✓ Successfully loaded both CSV files")
        print(f"  - Visual dataset: {visual_df.shape[0]} entries")
        print(f"  - Audio dataset: {audio_df.shape[0]} entries")
    except Exception as e:
        print(f"✗ Error loading CSV files: {e}")
        sys.exit(1)
    
    print("\nCreating multimodal datasets...")
    
    # Create different versions of combined datasets
    
    # 1. Intersection strategy - only common files with merged temporal annotations
    print("\n" + "="*60)
    print("1. INTERSECTION-BASED MULTIMODAL CSV")
    print("="*60)
    intersection_output = os.path.join(output_dir, 'AVVP_multimodal_intersection.csv')
    multimodal_intersection = create_multimodal_csv(
        visual_df, audio_df, 
        intersection_output, 
        strategy='intersection'
    )
    
    # 2. Union strategy - all files
    print("\n" + "="*60)
    print("2. UNION-BASED MULTIMODAL CSV")
    print("="*60)
    union_output = os.path.join(output_dir, 'AVVP_multimodal_union.csv')
    multimodal_union = create_multimodal_csv(
        visual_df, audio_df, 
        union_output, 
        strategy='union'
    )
    
    # 3. Simplified version - just common files with original format
    print("\n" + "="*60)
    print("3. SIMPLIFIED MULTIMODAL CSV")
    print("="*60)
    simplified_output = os.path.join(output_dir, 'AVVP_multimodal_simplified.csv')
    simplified_multimodal = create_simplified_multimodal_csv(
        visual_df, audio_df, 
        simplified_output
    )
    
    print("\n" + "="*80)
    print("SUMMARY OF CREATED FILES")
    print("="*80)
    print("Created three multimodal CSV files:")
    print(f"1. {intersection_output}")
    print(f"   - Description: Merged temporal annotations for common files")
    print(f"   - Entries: {len(multimodal_intersection) if 'multimodal_intersection' in locals() else 'N/A'}")
    print(f"   - Use case: Pure multimodal training with synchronized data")
    
    print(f"\n2. {union_output}")
    print(f"   - Description: All files with modality indicators")
    print(f"   - Entries: {len(multimodal_union) if 'multimodal_union' in locals() else 'N/A'}")
    print(f"   - Use case: Maximum data utilization with mixed modality training")
    
    print(f"\n3. {simplified_output}")
    print(f"   - Description: Common files in original format with modality labels")
    print(f"   - Entries: {len(simplified_multimodal) if 'simplified_multimodal' in locals() else 'N/A'}")
    print(f"   - Use case: Easy integration with existing training pipeline")
    
    # Display sample from intersection dataset
    if 'multimodal_intersection' in locals() and len(multimodal_intersection) > 0:
        print(f"\nSample from intersection dataset (first 10 entries):")
        print(multimodal_intersection.head(10).to_string())
    
    # File size analysis
    print(f"\nFile Size Analysis:")
    for file_path in [intersection_output, union_output, simplified_output]:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  - {os.path.basename(file_path)}: {size_mb:.2f} MB")
    
    print(f"\nRecommendations:")
    print(f"  - For multimodal training: Use 'AVVP_multimodal_simplified.csv'")
    print(f"  - For maximum data usage: Use 'AVVP_multimodal_union.csv'")
    print(f"  - For research on temporal alignment: Use 'AVVP_multimodal_intersection.csv'")
    
    print(f"\n" + "="*80)
    print("CSV CREATION COMPLETE")
    print("="*80)
    print(f"All files saved in: {output_dir}")
    print(f"Summary report: {summary_file}")
    
    # Restore original stdout and close file
    sys.stdout = original_stdout
    summary_f.close()
    
    print(f"\n✓ CSV creation complete! Check the summary file: {summary_file}")
    print(f"✓ Output files saved in: {output_dir}")