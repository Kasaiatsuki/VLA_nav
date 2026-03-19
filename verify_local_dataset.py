import os
import sys
import torch
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.getcwd())

from transformers import AutoProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.local_dataset import Local_Dataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

def verify():
    vla_path = "openvla/openvla-7b"
    # Note: Using a dummy path for now or a real one if exists
    data_dir = "/home/orne/kasai_ws/src/VLA_nav/data_sample" 
    
    # Create a dummy data sample if not exists for testing
    os.makedirs(f"{data_dir}/traj_0/images", exist_ok=True)
    with open(f"{data_dir}/traj_names.txt", "w") as f:
        f.write("traj_0\n")
    
    import numpy as np
    import pickle
    from PIL import Image
    
    # Create 10 dummy frames
    pos = np.zeros((20, 2), dtype=np.float32)
    yaw = np.zeros((20, 1), dtype=np.float32)
    for i in range(20):
        pos[i] = [i * 0.1, 0.0]
        yaw[i] = 0.0
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        img.save(f"{data_dir}/traj_0/images/{i:05d}.jpg")
        
    with open(f"{data_dir}/traj_0/traj_data.pkl", "wb") as f:
        pickle.dump({"position": pos, "yaw": yaw}, f)

    print("--- Initializing Processor ---")
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    print("--- Initializing Local_Dataset ---")
    dataset = Local_Dataset(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        data_folder=data_dir,
        horizon=8
    )
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    
    print("Sample keys:", sample.keys())
    print("Pixel values shape:", sample['pixel_values'].shape)
    print("Actions shape:", sample['actions'].shape)
    print("Action values (first 3):\n", sample['actions'][:3])
    print("Input IDs shape:", sample['input_ids'].shape)
    print("Prompt:", processor.tokenizer.decode(sample['input_ids']))
    
    print("\n--- Success! ---")

if __name__ == "__main__":
    verify()
