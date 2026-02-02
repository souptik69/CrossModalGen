import os
import sys
import numpy as np
import zipfile
from io import BytesIO
import pickle
from tqdm import tqdm
import argparse
import pandas as pd
import gc
from scipy.io import wavfile

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Add vggish_model directory to Python path
sys.path.append('/project/ag-jafra/Souptik/VGGSoundAVEL/vggish_model')
import vggish_input
import vggish_params
import vggish_slim

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def wav_to_logmel(wav_path, max_seconds=30):
    """Convert wav file to log-mel spectrogram"""
    sr, wav_data = wavfile.read(wav_path)
    assert wav_data.dtype == np.int16
    wav_data = wav_data / 32768.0
    
    # Handle stereo
    if len(wav_data.shape) > 1:
        wav_data = wav_data.mean(axis=1)
    
    available_seconds = min(max_seconds, len(wav_data) // sr)
    if available_seconds == 0:
        return None
    
    log_mel = np.zeros([available_seconds, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])
    for i in range(available_seconds):
        segment = wav_data[i * sr:(i + 1) * sr]
        segment_examples = vggish_input.waveform_to_examples(segment, sr)
        if segment_examples.shape[0] > 0:
            log_mel[i] = segment_examples
    
    return log_mel

def extract_features(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Load CSV
    df = pd.read_csv(args.csv_path)
    file_names = df['file_name'].unique().tolist()
    print(f"Found {len(file_names)} unique audio files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize VGGish model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, args.checkpoint_path)
    
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
    
    # Filter already processed
    to_process = []
    for fname in file_names:
        audio_name = fname[:-4]  # Remove .wav
        output_zip = os.path.join(args.output_dir, f"{audio_name}.zip")
        if not os.path.exists(output_zip):
            to_process.append(fname)
    
    print(f"Skipping {len(file_names) - len(to_process)} already processed, {len(to_process)} remaining")
    
    processed, errors = 0, 0
    
    # Process in batches (accumulate log-mels, then run inference)
    batch_data = []  # (audio_name, log_mel)
    
    for fname in tqdm(to_process, desc='Processing'):
        try:
            audio_name = fname[:-4]
            wav_path = os.path.join(args.audio_dir, fname)
            
            if not os.path.exists(wav_path):
                errors += 1
                continue
            
            log_mel = wav_to_logmel(wav_path)
            if log_mel is None:
                errors += 1
                continue
            
            batch_data.append((audio_name, log_mel))
            
            # Process batch when full
            if len(batch_data) >= args.batch_size:
                for name, mel in batch_data:
                    embeddings = sess.run(embedding_tensor, feed_dict={features_tensor: mel})
                    output_zip = os.path.join(args.output_dir, f"{name}.zip")
                    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        buffer = BytesIO()
                        pickle.dump(embeddings.astype(np.float32), buffer)
                        zf.writestr(f"{name}.pkl", buffer.getvalue())
                    processed += 1
                batch_data = []
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            errors += 1
    
    # Process remaining
    for name, mel in batch_data:
        try:
            embeddings = sess.run(embedding_tensor, feed_dict={features_tensor: mel})
            output_zip = os.path.join(args.output_dir, f"{name}.zip")
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                buffer = BytesIO()
                pickle.dump(embeddings.astype(np.float32), buffer)
                zf.writestr(f"{name}.pkl", buffer.getvalue())
            processed += 1
        except Exception as e:
            print(f"Error saving {name}: {e}")
            errors += 1
    
    sess.close()
    print(f"Done! Processed: {processed}, Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', required=True, help='Path to audio wav files')
    parser.add_argument('--csv_path', required=True, help='Path to clotho CSV')
    parser.add_argument('--output_dir', required=True, help='Output directory for feature zips')
    parser.add_argument('--checkpoint_path', required=True, help='Path to VGGish checkpoint')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    args = parser.parse_args()
    extract_features(args)