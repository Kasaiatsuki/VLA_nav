import os
import pickle
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as PILImage

class OmniVLAEdgeDataset(Dataset):
    """
    Local dataset for OmniVLA-edge fine-tuning.
    Expects data collected in the format:
    traj_X/
        images/
            00000.jpg, 00001.jpg, ...
        traj_data.pkl (contains 'position' and 'yaw' lists)
    """
    def __init__(self, data_root, context_size=5, len_traj_pred=8, image_size=(192, 192)):
        self.data_root = data_root
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        traj_dirs = [d for d in os.listdir(self.data_root) if d.startswith('traj_')]
        for traj_dir in traj_dirs:
            traj_path = os.path.join(self.data_root, traj_dir)
            pkl_path = os.path.join(traj_path, 'traj_data.pkl')
            img_dir = os.path.join(traj_path, 'images')
            
            if not os.path.exists(pkl_path) or not os.path.exists(img_dir):
                continue
                
            with open(pkl_path, 'rb') as f:
                traj_data = pickle.load(f)
            
            positions = traj_data['position']
            yaws = traj_data['yaw']
            num_frames = len(positions)
            
            # We need at least context_size+1 frames to form a sequence, 
            # and len_traj_pred frames in the future for waypoints.
            for i in range(self.context_size, num_frames - self.len_traj_pred):
                self.samples.append({
                    'img_dir': img_dir,
                    'index': i,
                    'positions': positions,
                    'yaws': yaws
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_dir = sample['img_dir']
        curr_idx = sample['index']
        
        # 1. Load context images (history + current)
        context_images = []
        for i in range(curr_idx - self.context_size, curr_idx + 1):
            img_path = os.path.join(img_dir, f"{i:05d}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img_tensor = self.transform(PILImage.fromarray(img))
            context_images.append(img_tensor)
        
        # Stack images along channel dimension as expected by model_omnivla_edge.py
        # Current implementation in model expects (B, 3 * (context_size + 1), H, W)
        obs_stack = torch.cat(context_images, dim=0)
        
        # 2. Extract future trajectory (8 steps) and convert to relative coordinates
        future_pos = np.array(sample['positions'][curr_idx : curr_idx + self.len_traj_pred])
        future_yaw = np.array(sample['yaws'][curr_idx : curr_idx + self.len_traj_pred])
        
        # Reference frame is the current post (curr_idx)
        ref_pos = np.array(sample['positions'][curr_idx])
        ref_yaw = sample['yaws'][curr_idx]
        
        # Transformation to robot-centric coordinates
        rel_waypoints = []
        for i in range(self.len_traj_pred):
            pos = future_pos[i]
            yaw = future_yaw[i]
            
            # Translation
            dx = pos[0] - ref_pos[0]
            dy = pos[1] - ref_pos[1]
            
            # Rotation (Rotate by -ref_yaw)
            rot_x = dx * np.cos(-ref_yaw) - dy * np.sin(-ref_yaw)
            rot_y = dx * np.sin(-ref_yaw) + dy * np.cos(-ref_yaw)
            
            rel_yaw = yaw - ref_yaw
            
            # The model expects [dx, dy, cos(rel_yaw), sin(rel_yaw)]
            rel_waypoints.append([rot_x, rot_y, np.cos(rel_yaw), np.sin(rel_yaw)])
            
        action_label = torch.tensor(rel_waypoints, dtype=torch.float32)
        
        return obs_stack, action_label
