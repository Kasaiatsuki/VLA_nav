import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model_omnivla_edge import OmniVLA_edge
from vla_nav_dataset_local import OmniVLAEdgeDataset

def train():
    # --- Configuration ---
    DATA_ROOT = "/home/kasai/ros2_ws/src/VLA_nav/data_sample" # Change to your actual dataset path
    MODEL_PATH = "/home/kasai/ros2_ws/src/VLA_nav/omnivla/inference/omnivla-edge.pth" # Path to base weights
    SAVE_PATH = "omnivla-edge-finetuned.pth"
    
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Model Initialization ---
    model = OmniVLA_edge(
        context_size=5, 
        len_traj_pred=8, 
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=1024
    )
    
    # Load weights if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: {MODEL_PATH} not found. Starting from scratch or random weights.")
    
    model.to(DEVICE)
    model.train()
    
    # --- Goal & Mask Preparation (Language Mode 7) ---
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    prompt = "move forward avoiding walls" # Adjust to your training goal
    text_token = clip.tokenize([prompt]).to(DEVICE)
    with torch.no_grad():
        feat_text = clip_model.encode_text(text_token).float()
    
    # Dummy inputs as used in vla_nav_node.py
    goal_pose = torch.zeros((1, 4)).to(DEVICE)
    map_images = torch.zeros((1, 9, 192, 192)).to(DEVICE)
    goal_img = torch.zeros((1, 3, 192, 192)).to(DEVICE)
    modality_id = torch.tensor([7]).to(DEVICE) # Language mode ID
    
    # --- Dataset & Dataloader ---
    dataset = OmniVLAEdgeDataset(DATA_ROOT, context_size=5, len_traj_pred=8)
    if len(dataset) == 0:
        print("Error: No samples found in dataset. Check DATA_ROOT.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss() 
    
    # --- Training Loop ---
    print(f"Starting fine-tuning on {DEVICE}...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for obs_stack, action_label in pbar:
            obs_stack = obs_stack.to(DEVICE)
            action_label = action_label.to(DEVICE)
            
            # current_img is the latest frame in the stack (last 3 channels)
            current_img = obs_stack[:, -3:, :, :]
            
            # Forward pass
            # Returns: dist_pred, waypoints, _, _, _
            batch_size = obs_stack.shape[0]
            # Expand dummies for batch
            b_goal_pose = goal_pose.expand(batch_size, -1)
            b_map_images = map_images.expand(batch_size, -1, -1, -1)
            b_goal_img = goal_img.expand(batch_size, -1, -1, -1)
            b_modality_id = modality_id.expand(batch_size)
            b_feat_text = feat_text.expand(batch_size, -1)
            
            outputs = model(
                obs_stack, 
                b_goal_pose, 
                b_map_images, 
                b_goal_img, 
                b_modality_id, 
                b_feat_text, 
                current_img
            )
            
            # waypoints is the second return value
            waypoints = outputs[1]
            
            # Action prediction loss
            loss = criterion(waypoints, action_label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")
        
        # Save every epoch
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved checkpoint to {SAVE_PATH}")

if __name__ == "__main__":
    train()
