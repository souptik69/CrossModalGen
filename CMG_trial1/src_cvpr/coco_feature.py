import os
import json
import pickle
import zipfile
import argparse
import numpy as np
import cv2
from io import BytesIO
from tqdm import tqdm
import tensorflow as tf
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def extract_features(args):
    # Load VGG19 model
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    
    # Warmup
    model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
    
    # Load annotations
    with open(args.annotations_json, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    
    # Get unique image IDs
    image_ids = sorted(set(ann['image_id'] for ann in annotations))
    print(f"Found {len(image_ids)} unique images, batch size: {args.batch_size}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter out already processed
    to_process = []
    for img_id in image_ids:
        output_zip = os.path.join(args.output_dir, f"{img_id:012}.zip")
        if not os.path.exists(output_zip):
            to_process.append(img_id)
    
    print(f"Skipping {len(image_ids) - len(to_process)} already processed, {len(to_process)} remaining")
    
    processed, errors = 0, 0
    
    # Process in batches
    for batch_start in tqdm(range(0, len(to_process), args.batch_size), desc='Batches'):
        batch_ids = to_process[batch_start:batch_start + args.batch_size]
        batch_images = []
        valid_ids = []
        
        # Load batch of images
        for img_id in batch_ids:
            img_path = os.path.join(args.image_dir, f"{img_id:012}.jpg")
            if not os.path.exists(img_path):
                errors += 1
                continue
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                img = preprocess_input(img.astype(np.float32))
                batch_images.append(img)
                valid_ids.append(img_id)
            except Exception as e:
                print(f"Error loading {img_id}: {e}")
                errors += 1
        
        if not batch_images:
            continue
        
        # Extract features for batch
        batch_array = np.stack(batch_images, axis=0)
        batch_features = model.predict(batch_array, verbose=0).astype(np.float32)
        
        # Save each feature
        for idx, img_id in enumerate(valid_ids):
            try:
                output_zip = os.path.join(args.output_dir, f"{img_id:012}.zip")
                with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                    buffer = BytesIO()
                    pickle.dump(batch_features[idx], buffer)
                    zf.writestr(f"{img_id:012}.pkl", buffer.getvalue())
                processed += 1
            except Exception as e:
                print(f"Error saving {img_id}: {e}")
                errors += 1
    
    print(f"Done! Processed: {processed}, Errors: {errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help='Path to val2017 images')
    parser.add_argument('--annotations_json', required=True, help='Path to captions_val2017.json')
    parser.add_argument('--output_dir', required=True, help='Output directory for feature zips')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    args = parser.parse_args()
    extract_features(args)