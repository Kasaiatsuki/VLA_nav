import os
import pickle
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])

def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    """
    Convert absolute positions to local coordinates relative to curr_pos and curr_yaw.
    positions: (N, 2)
    curr_pos: (2,)
    curr_yaw: float (radians)
    """
    rotmat = yaw_rotmat(curr_yaw)
    rot2 = rotmat[:2, :2]
    # Rotate to local frame
    return (positions - curr_pos) @ rot2

class Local_Dataset(Dataset):
    """
    Custom Dataset for local VLA data collected with create_data_vla.py.
    """
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_folder: str,
        horizon: int = 8,
        predict_stop_token: bool = True,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.data_folder = Path(data_folder)
        self.horizon = horizon
        self.predict_stop_token = predict_stop_token
        self.image_size = image_size
        
        # Tokenizer and Transforms
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.image_transform = image_transform

        # Build Index
        self.samples = []
        traj_names_file = self.data_folder / "traj_names.txt"
        if not traj_names_file.exists():
            raise FileNotFoundError(f"traj_names.txt not found in {data_folder}")
            
        with open(traj_names_file, "r") as f:
            traj_names = [line.strip() for line in f if line.strip()]
            
        self.trajectory_data = {}
        for traj_name in traj_names:
            traj_path = self.data_folder / traj_name
            pkl_path = traj_path / "traj_data.pkl"
            if not pkl_path.exists():
                print(f"Warning: {pkl_path} not found. Skipping.")
                continue
                
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.trajectory_data[traj_name] = data
                
            num_frames = len(data["position"])
            # We need at least horizon steps in the future
            for i in range(num_frames - self.horizon):
                self.samples.append((traj_name, i))
                
        print(f"Loaded Local_Dataset with {len(self.samples)} samples from {len(self.trajectory_data)} trajectories.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, traj_name: str, frame_idx: int) -> Image.Image:
        # Try .jpg then .png
        img_path = self.data_folder / traj_name / "images" / f"{frame_idx:05d}.jpg"
        if not img_path.exists():
             img_path = self.data_folder / traj_name / "images" / f"{frame_idx:05d}.png"
        
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, i: int) -> Dict[str, Any]:
        traj_name, curr_idx = self.samples[i]
        
        # 1. Load Image
        image = self._load_image(traj_name, curr_idx)
        pixel_values = self.image_transform(image)
        
        # 2. Compute Actions (Future Waypoints)
        data = self.trajectory_data[traj_name]
        positions = data["position"] # (N, 2)
        yaws = data["yaw"] # (N, 1) or (N,)
        
        curr_pos = positions[curr_idx]
        curr_yaw = yaws[curr_idx]
        if isinstance(curr_yaw, np.ndarray): curr_yaw = curr_yaw.item()
        
        # Get future waypoints [curr_idx + 1, ..., curr_idx + horizon]
        future_poses = positions[curr_idx + 1 : curr_idx + self.horizon + 1]
        future_yaws = yaws[curr_idx + 1 : curr_idx + self.horizon + 1]
        
        # Convert to local relative coordinates
        waypoints = to_local_coords(future_poses, curr_pos, curr_yaw)
        
        # Action format for OpenVLA usually includes (dx, dy, cos_dyaw, sin_dyaw)
        # or similar. Based on OmniVLA's robot_pos_model, it uses:
        # (z_traj, -x_traj, cos(-yaw_traj), sin(-yaw_traj)) / metric_waypoint_spacing
        # But here let's simplify to (dx, dy, cos_yaw, sin_yaw) relative to start frame
        
        actions = []
        for j in range(len(waypoints)):
            rel_yaw = future_yaws[j] - curr_yaw
            if isinstance(rel_yaw, np.ndarray): rel_yaw = rel_yaw.item()
            
            # (dx, dy, cos_yaw_diff, sin_yaw_diff)
            # Using metric scaling if needed, usually 1.0 if not specified
            actions.append([
                waypoints[j, 0], 
                waypoints[j, 1], 
                np.cos(rel_yaw), 
                np.sin(rel_yaw)
            ])
        
        actions = torch.tensor(actions, dtype=torch.float32) # (horizon, 4)
        
        # 3. Tokenize Action for Prompt
        # In OpenVLA, we might want to predict the action tokens too.
        # But OmniVLA-edge mostly uses the action_head (L1 Regression).
        # We still need the prompt tokens.
        
        action_chunk_string = "".join(self.action_tokenizer(actions))
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": "follow the path"}, # Default prompt
            {"from": "gpt", "value": action_chunk_string},
        ]
        
        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   
        
        # Mask labels to only take loss on action tokens
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=actions,
            dataset_name="local",
            modality_id=6, # "image only" mode in OmniVLA
            action_select_mask=torch.tensor(1.0), # Use raw action
            img_PIL=image,
            lan_prompt="follow the path"
        )
