import os
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

def create_dummy_data(base_path: str):
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    traj_name = "dummy_traj_0"
    traj_dir = base_dir / traj_name
    img_dir = traj_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create dummy images (20 frames to allow horizon=8)
    for i in range(20):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(img_dir / f"{i:05d}.jpg")
    
    # 2. Create dummy traj_data.pkl
    # Required keys: 'position', 'yaw'
    x = np.linspace(0, 5, 20)
    y = np.zeros(20)
    data = {
        'position': np.stack([x, y], axis=1), # (N, 2)
        'yaw': np.zeros(20)
    }
    with open(traj_dir / "traj_data.pkl", "wb") as f:
        pickle.dump(data, f)
        
    # 3. Create traj_names.txt
    with open(base_dir / "traj_names.txt", "w") as f:
        f.write(traj_name + "\n")

    print(f"Dummy dataset created at {base_path}")

if __name__ == "__main__":
    create_dummy_data("./dummy_dataset")
