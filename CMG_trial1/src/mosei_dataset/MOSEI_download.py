# from constants.paths import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
# import sys

# if SDK_PATH is None:
#     print("SDK path is not specified! Please specify first in constants/paths.py")
#     exit(0)
# else:
#     sys.path.append(SDK_PATH)

# import mmsdk
# import os
# import re
# import numpy as np
# from mmsdk import mmdatasdk as md
# from subprocess import check_call, CalledProcessError

# # create folders for storing the data
# if not os.path.exists(DATA_PATH):
#     check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)

# # download highlevel features, low-level (raw) data and labels for CMU-MOSEI dataset
# DATASET = md.cmu_mosei

# try:
#     md.mmdataset(DATASET.highlevel, DATA_PATH)
#     print("High-level features downloaded successfully.")
# except RuntimeError:
#     print("High-level features have been downloaded previously.")

# try:
#     md.mmdataset(DATASET.raw, DATA_PATH)
#     print("Raw data downloaded successfully.")
# except RuntimeError:
#     print("Raw data have been downloaded previously.")
    
# try:
#     md.mmdataset(DATASET.labels, DATA_PATH)
#     print("Labels downloaded successfully.")
# except RuntimeError:
#     print("Labels have been downloaded previously.")

# print("CMU-MOSEI download complete!")



# from constants.paths import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
# import sys
# from datetime import datetime

# if SDK_PATH is None:
#     print("SDK path is not specified! Please specify first in constants/paths.py")
#     exit(0)
# else:
#     sys.path.append(SDK_PATH)

# import mmsdk
# import os
# import re
# import numpy as np
# from mmsdk import mmdatasdk as md
# from subprocess import check_call, CalledProcessError
# from tqdm import tqdm

# def print_progress(message):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# # create folders for storing the data
# print_progress("Creating data directory...")
# if not os.path.exists(DATA_PATH):
#     check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)
#     print_progress(f"Created directory: {DATA_PATH}")
# else:
#     print_progress(f"Directory already exists: {DATA_PATH}")

# # download highlevel features, low-level (raw) data and labels for CMU-MOSEI dataset
# DATASET = md.cmu_mosei

# # Progress tracking for downloads
# downloads = [
#     ("High-level features", DATASET.highlevel, "high-level"),
#     ("Raw data", DATASET.raw, "raw"),
#     ("Labels", DATASET.labels, "labels")
# ]

# print_progress("Starting CMU-MOSEI dataset download...")
# print_progress("Note: Each download may take several hours due to large file sizes")

# for i, (desc, dataset_part, part_name) in enumerate(downloads, 1):
#     print_progress(f"[{i}/3] Starting download: {desc}")
    
#     try:
#         # Check if files already exist
#         existing_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csd')] if os.path.exists(DATA_PATH) else []
#         files_before = len(existing_files)
        
#         print_progress(f"Downloading {desc}... (this may take a while)")
#         md.mmdataset(dataset_part, DATA_PATH)
        
#         # Check files after download
#         existing_files_after = [f for f in os.listdir(DATA_PATH) if f.endswith('.csd')]
#         files_after = len(existing_files_after)
        
#         if files_after > files_before:
#             print_progress(f"✓ {desc} downloaded successfully! ({files_after - files_before} new files)")
#         else:
#             print_progress(f"✓ {desc} verified (files already present)")
            
#     except RuntimeError as e:
#         print_progress(f"✓ {desc} have been downloaded previously.")
#     except Exception as e:
#         print_progress(f"✗ Error downloading {desc}: {str(e)}")

# # Final summary
# print_progress("Checking final dataset...")
# final_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csd')] if os.path.exists(DATA_PATH) else []
# print_progress(f"Total .csd files in dataset: {len(final_files)}")

# for file in sorted(final_files):
#     file_path = os.path.join(DATA_PATH, file)
#     file_size = os.path.getsize(file_path) / (1024**3)  # GB
#     print_progress(f"  - {file} ({file_size:.2f} GB)")

# print_progress("CMU-MOSEI download complete!")




# import os
# import sys
# import subprocess
# from datetime import datetime

# # Add MultiBench to path
# MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
# DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'

# sys.path.append(MULTIBENCH_PATH)

# def print_progress(message):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{timestamp}] {message}")

# def download_mosei_pkl():
#     """Download mosei_raw.pkl from Google Drive"""
#     file_id = "1zFOBHijVppTiyteSsi0aTFYPEsda_AOk"
#     pkl_path = os.path.join(DATA_PATH, 'mosei_raw.pkl')
    
#     if os.path.exists(pkl_path):
#         print_progress("✓ mosei_raw.pkl already exists")
#         return pkl_path
    
#     print_progress("Downloading mosei_raw.pkl from Google Drive...")
    
#     try:
#         # Try using gdown
#         import gdown
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, pkl_path, quiet=False)
#         print_progress("✓ Downloaded using gdown")
        
#     except ImportError:
#         print_progress("gdown not found, installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
#         import gdown
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, pkl_path, quiet=False)
#         print_progress("✓ Downloaded using gdown")
        
#     except Exception as e:
#         print_progress(f"gdown failed: {e}")
#         print_progress("Trying wget...")
        
#         # Fallback to wget
#         wget_url = f"https://drive.google.com/uc?export=download&id={file_id}"
#         try:
#             subprocess.check_call(["wget", "--no-check-certificate", "-O", pkl_path, wget_url])
#             print_progress("✓ Downloaded using wget")
#         except Exception as e2:
#             print_progress(f"✗ Download failed: {e2}")
#             print_progress("Please manually download from: https://drive.google.com/file/d/1zFOBHijVppTiyteSsi0aTFYPEsda_AOk/view")
#             return None
    
#     return pkl_path

# # Create data directory
# os.makedirs(DATA_PATH, exist_ok=True)
# print_progress(f"Data directory: {DATA_PATH}")

# # Download mosei_raw.pkl first
# pkl_path = download_mosei_pkl()
# if pkl_path is None:
#     print_progress("✗ Failed to download mosei_raw.pkl")
#     sys.exit(1)

# try:
#     # Import MultiBench data loader
#     from datasets.affect.get_data import get_dataloader
#     print_progress("MultiBench modules imported successfully")
    
#     # Process CMU-MOSEI data using the downloaded pkl file
#     print_progress("Processing CMU-MOSEI data with MultiBench...")
    
#     # Get dataloaders using the downloaded file
#     traindata, validdata, testdata = get_dataloader(
#         pkl_path,
#         data_type='mosei',
#         max_pad=True,
#         max_seq_len=50
#     )
    
#     print_progress("✓ CMU-MOSEI dataloaders created successfully")
#     print_progress(f"  - Train samples: {len(traindata.dataset) if hasattr(traindata, 'dataset') else 'N/A'}")
#     print_progress(f"  - Valid samples: {len(validdata.dataset) if hasattr(validdata, 'dataset') else 'N/A'}")
#     print_progress(f"  - Test samples: {len(testdata.dataset) if hasattr(testdata, 'dataset') else 'N/A'}")
    
#     # Check file size
#     if os.path.exists(pkl_path):
#         file_size = os.path.getsize(pkl_path) / (1024**2)  # MB
#         print_progress(f"  - mosei_raw.pkl: {file_size:.2f} MB")
    
# except ImportError as e:
#     print_progress(f"✗ Import error: {e}")
#     print_progress("Make sure MultiBench is properly installed")
# except Exception as e:
#     print_progress(f"✗ Error: {e}")

# print_progress("CMU-MOSEI MultiBench setup complete!")


import os
import sys
import subprocess
from datetime import datetime

# Add MultiBench to path
MULTIBENCH_PATH = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/MultiBench"
DATA_PATH = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'

sys.path.append(MULTIBENCH_PATH)

def print_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def download_file(file_id, filename, description):
    """Download file from Google Drive with skip check"""
    file_path = os.path.join(DATA_PATH, filename)
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024**2)  # MB
        print_progress(f"✓ {filename} already exists ({file_size:.2f} MB)")
        return file_path
    
    print_progress(f"Downloading {description}...")
    
    try:
        # Try using gdown
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
        print_progress(f"✓ Downloaded {filename}")
        
    except ImportError:
        print_progress("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
        print_progress(f"✓ Downloaded {filename}")
        
    except Exception as e:
        print_progress(f"gdown failed for {filename}: {e}")
        print_progress("Trying wget...")
        
        wget_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        try:
            subprocess.check_call(["wget", "--no-check-certificate", "-O", file_path, wget_url])
            print_progress(f"✓ Downloaded {filename} using wget")
        except Exception as e2:
            print_progress(f"✗ Download failed for {filename}: {e2}")
            return None
    
    return file_path

def download_all_mosei_files():
    """Download all CMU-MOSEI files"""
    files_to_download = [
        ("1zFOBHijVppTiyteSsi0aTFYPEsda_AOk", "mosei_raw.pkl", "MOSEI raw data"),
        ("180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz", "mosei_senti_data.pkl", "MOSEI sentiment data"),
        ("1vvFSabZYvFeYU2ERyI3cM0vD2gkLOGte", "mosei_unalign.hdf5", "MOSEI unaligned data"),
        ("1clO058hj4GRQFPJAZNKilsbhgu-NzTXE", "mosei.hdf5", "MOSEI aligned data")
    ]
    
    downloaded_files = {}
    for file_id, filename, description in files_to_download:
        file_path = download_file(file_id, filename, description)
        downloaded_files[filename] = file_path
    
    return downloaded_files

# Create data directory
os.makedirs(DATA_PATH, exist_ok=True)
print_progress(f"Data directory: {DATA_PATH}")

# Download all MOSEI files
print_progress("Checking and downloading CMU-MOSEI files...")
downloaded_files = download_all_mosei_files()

# Check if raw pkl downloaded successfully
pkl_path = downloaded_files.get('mosei_raw.pkl')
if pkl_path is None or not os.path.exists(pkl_path):
    print_progress("✗ Failed to download mosei_raw.pkl")
    sys.exit(1)

try:
    # Import MultiBench data loader
    from datasets.affect.get_data import get_dataloader
    print_progress("MultiBench modules imported successfully")
    
    # Process CMU-MOSEI data using the downloaded pkl file
    print_progress("Processing CMU-MOSEI data with MultiBench...")
    
    # Get dataloaders using the downloaded file
    traindata, validdata, testdata = get_dataloader(
        pkl_path,
        data_type='mosei',
        max_pad=True,
        max_seq_len=50
    )
    
    print_progress("✓ CMU-MOSEI dataloaders created successfully")
    print_progress(f"  - Train samples: {len(traindata.dataset) if hasattr(traindata, 'dataset') else 'N/A'}")
    print_progress(f"  - Valid samples: {len(validdata.dataset) if hasattr(validdata, 'dataset') else 'N/A'}")
    print_progress(f"  - Test samples: {len(testdata.dataset) if hasattr(testdata, 'dataset') else 'N/A'}")
    
    # Report all downloaded files
    print_progress("Downloaded files summary:")
    for filename, filepath in downloaded_files.items():
        if filepath and os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024**2)  # MB
            print_progress(f"  - {filename}: {file_size:.2f} MB")
    
except ImportError as e:
    print_progress(f"✗ Import error: {e}")
    print_progress("Make sure MultiBench is properly installed")
except Exception as e:
    print_progress(f"✗ Error: {e}")

print_progress("CMU-MOSEI MultiBench setup complete!")