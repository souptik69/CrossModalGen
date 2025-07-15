from typing import Optional

# path to the SDK folder
SDK_PATH: Optional[str] = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src/mosei_dataset/CMU-MultimodalSDK/"

# path to the folder where you want to store data
DATA_PATH: Optional[str] = '/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMU_MOSEI/'

# path to a pretrained word embedding file
WORD_EMB_PATH: Optional[str] = None

# path to loaded word embedding matrix and corresponding word2id mapping
CACHE_PATH: Optional[str] = './data/embedding_and_mapping.pt'
