import os
import sys
import io
import matplotlib.pyplot as plt

# ROS
#from sensor_msgs.msg import Image

# pytorch
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF

import clip
import numpy as np
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional

#model architecture
from model_omnivla_edge import OmniVLA_edge


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """事前に学習されたモデル(チェックポイント)から重みと設定を読み込み、推論可能な状態にする"""
    model_type = config["model_type"]
    
    # 1. configファイルの設定値をもとに空のモデル（ネットワーク構造）を組み立てる
    if config["model_type"] == "omnivla-edge":
        model = OmniVLA_edge(        
            context_size=config["context_size"],  # 過去の画像を何フレーム考慮するか
            len_traj_pred=config["len_traj_pred"],  # 予測する軌跡（アクション）の長さ
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],  # 観測データ(画像)を処理するエンコーダーの種類
            obs_encoding_size=config["obs_encoding_size"],  # エンコード後の特徴量サイズ
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],  # Attention層のヘッド数
            mha_num_attention_layers=config["mha_num_attention_layers"],  # Attention層の階層数
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )  
        # 言語指示（テキスト）を処理するためのCLIPモデルとその前処理ツールをロードする
        text_encoder, preprocess = clip.load(config["clip_type"], device=device)    
        # 推論の安定性と互換性のためにfloat32精度に変換する
        text_encoder.to(torch.float32)    
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # 2. 保存されている学習済みの重みデータ（チェックポイント）をメモリ(指定デバイス)に読み込む
    checkpoint = torch.load(model_path, map_location=device)
    
    # 3. 読み込んだ重みデータを、先ほど組み立てた空のモデルに流し込む
    if model_type == "omnivla-edge":
        state_dict = checkpoint
        # strict=Trueでモデル構造と重みの形が完全一致するか厳密にチェックしながらロードする
        model.load_state_dict(state_dict, strict=True)
    else:
        loaded_model = checkpoint["model"]
        # 複数GPU(DDP)で学習したモデルは "module." という名前空間が付くため、
        # エラー処理(try-except)を用いて、単一GPUと複数GPUどちらの重みファイルにも対応できるようにする
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)
    
    # 4. 組み立てたモデル全体を指定されたデバイス(GPU等)に転送する
    model.to(device)
    
    # モデル本体、言語理解用のテキストエンコーダー、前処理関数の3点セットを返す
    return model, text_encoder, preprocess

def transform_images_PIL_mask(pil_imgs: List[PILImage.Image], mask) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            #transforms.ToTensor(),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        transf_img = transform_type(TF.to_tensor(pil_img*mask)/255.0) #/255.0
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def transform_images_PIL(pil_imgs: List[PILImage.Image]) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        transf_img = transform_type(pil_img.copy())
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def transform_images_map(pil_imgs: List[PILImage.Image]) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    image_size_small = (96, 96)
    
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        pil_img = pil_img.resize(image_size_small) 
        transf_img = transform_type(pil_img)          
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)
