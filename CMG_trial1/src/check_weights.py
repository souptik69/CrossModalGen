from model.main_model_novel import AV_VQVAE_Encoder
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

video_dim = 512
audio_dim = 128
video_output_dim = 2048
n_embeddings = 400
embedding_dim = 256
model_resume = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Encoder = AV_VQVAE_Encoder(audio_dim, video_dim, video_output_dim, n_embeddings, embedding_dim)
Encoder.double()
Encoder.to(device)

path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/Novel_AV_final_2/Meta_Hier_Softmax_CPC/checkpoint/DCID-model-5.pt"
# path_checkpoints = "/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/Novel_AV_final_2/Meta_Hier_Softmax/checkpoint/DCID-model-5.pt"
print(f"Loading checkpoint from: {path_checkpoints}")

checkpoints = torch.load(path_checkpoints)
Encoder.load_state_dict(checkpoints['Encoder_parameters'])

quantizer = Encoder.Cross_quantizer
modal_weights = quantizer.modal_weights
hier_weights = quantizer.hier_weights
modal_weights_video = modal_weights[:, 0:2]
modal_weights_audio = modal_weights[:, 2:4]
hier_weights_video = hier_weights[:, 0:2]
hier_weights_audio = hier_weights[:, 2:4]

print("="*60)
print("Model Weights and Hierarchical Analysis - First 10 Vectors")
print("="*60)

for i in range(400):  # Print first 10 vectors
    print(f"Vector {i:2d}:")
    print(f"  Video modality weights (v → v), (a → v) : [{modal_weights_video[i,0]:.4f}, {modal_weights_video[i,1]:.4f}]")
    print(f"  Audio modality weights (v → a), (a  → a): [{modal_weights_audio[i,0]:.4f}, {modal_weights_audio[i,1]:.4f}]")
    # print(f"- Hierarchical weights: [{hier_weights_parameter[i,0]:.4f}, {hier_weights_parameter[i,1]:.4f}]")
    print(f"  Video hier weights (v → v), (a → v): [{hier_weights_video[i,0]:.4f}, {hier_weights_video[i,1]:.4f}]")
    print(f"  Audio hier weights (v → a), (a  → a): [{hier_weights_audio[i,0]:.4f}, {hier_weights_audio[i,1]:.4f}]")
    print()

print("Analysis completed successfully!")