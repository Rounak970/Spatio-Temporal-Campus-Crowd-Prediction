"""
train.py
------------------------------------
Training script for the Spatio-Temporal Crowd Prediction Model.
Handles:
 • Multi-task loss (Count + Density + Motion)
 • Validation loop
 • Saving best model
"""

import torch
import torch.nn as nn
from model_architecture import CrowdPredictor


# --------------------------------------------------------
# Loss Functions
# --------------------------------------------------------
count_loss_fn = nn.MSELoss()
density_loss_fn = nn.MSELoss()
motion_loss_fn = nn.MSELoss()


# --------------------------------------------------------
# One Training Epoch
# --------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0

    for video, wifi, count, density, motion in loader:
        video, wifi = video.to(device), wifi.to(device)
        count, density, motion = count.to(device), density.to(device), motion.to(device)

        optimizer.zero_grad()

        pred_count, pred_density, pred_motion = model(video, wifi)

        loss = (
            count_loss_fn(pred_count, count.unsqueeze(1)) +
            density_loss_fn(pred_density, density) +
            motion_loss_fn(pred_motion, motion)
        )

        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


# --------------------------------------------------------
# Validation Epoch
# --------------------------------------------------------
def validate(model, loader, device):
    model.eval()
    total = 0

    with torch.no_grad():
        for video, wifi, count, density, motion in loader:
            video, wifi = video.to(device), wifi.to(device)
            count, density, motion = count.to(device), density.to(device), motion.to(device)

            pred_count, pred_density, pred_motion = model(video, wifi)

            loss = (
                count_loss_fn(pred_count, count.unsqueeze(1)) +
                density_loss_fn(pred_density, density) +
                motion_loss_fn(pred_motion, motion)
            )

            total += loss.item()

    return total / len(loader)


# --------------------------------------------------------
# Full Training
# --------------------------------------------------------
def train_model(train_loader, val_loader, wifi_dim=50, epochs=10, save_path="best_3dcnn_convlstm_model.pth"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    model = CrowdPredictor(wifi_dim=wifi_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        val = validate(model, val_loader, device)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {tr:.4f}")
        print(f"  Val Loss  : {val:.4f}")

        if val < best_val_loss:
            best_val_loss = val
            torch.save(model.state_dict(), save_path)
            print("  ✓ Best model saved!")

    return model
