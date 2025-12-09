"""
inference.py
------------------------------------
Loads the trained model and performs prediction on test clips.
Outputs:
 • Predicted Count
 • Predicted Density Map (32×32)
 • Predicted Motion (2×16×16)
"""

import torch
import numpy as np
from model_architecture import CrowdPredictor
from dataset_preprocessing import load_frames, make_clips


# --------------------------------------------------------
# Load Model
# --------------------------------------------------------
def load_trained_model(model_path, wifi_dim=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrowdPredictor(wifi_dim=wifi_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# --------------------------------------------------------
# Inference Function
# --------------------------------------------------------
def predict_on_test(model, device, test_frames_folder):
    frames = load_frames(test_frames_folder)
    clips = make_clips(frames, clip_len=10)

    wifi_fake = np.random.randint(5, 50, size=(len(clips), 10)).astype("float32")

    clips_tensor = torch.tensor(clips).unsqueeze(1).to(device)
    wifi_tensor = torch.tensor(wifi_fake).to(device)

    with torch.no_grad():
        pred_count, pred_density, pred_motion = model(clips_tensor, wifi_tensor)

    return (
        pred_count.cpu().numpy(),
        pred_density.cpu().numpy(),
        pred_motion.cpu().numpy(),
    )
