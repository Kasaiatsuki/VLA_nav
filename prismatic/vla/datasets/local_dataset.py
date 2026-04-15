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

# Prismatic の共通設定やツール
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform

def yaw_rotmat(yaw: float) -> np.ndarray:
    """
    指定されたヨー角（yaw）から 3x3 の回転行列を作成します。
    """
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])

def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    """
    絶対座標（世界座標）での位置を、現在のロボットの位置と向きを基準とした
    「ローカル座標（ロボットから見た相対的な位置）」に変換します。
    
    positions: (N, 2) 変換したい未来の座標リスト
    curr_pos: (2,) 現在のロボットの X, Y 座標
    curr_yaw: 現在のロボットが向いている方位（ラジアン）
    """
    rotmat = yaw_rotmat(curr_yaw)
    rot2 = rotmat[:2, :2]
    # 現在地を原点（0,0）にずらした後、現在の向きに合わせて回転させる
    return (positions - curr_pos) @ rot2

class Local_Dataset(Dataset):
    """
    ZEDカメラで収集した画像と軌跡データ (traj_data.pkl) を、
    VLAの学習形式に変換するためのカスタムデータセットクラスです。
    """
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_folder: str,
        horizon: int = 8,                # 未来の何ステップ分を学習するか
        predict_stop_token: bool = True, # 停止トークンを学習に含めるか
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.data_folder = Path(data_folder).expanduser().resolve()
        self.horizon = horizon
        self.predict_stop_token = predict_stop_token
        self.image_size = image_size
        
        # トークナイザーと画像処理
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.image_transform = image_transform

        # 1. 保存されたフォルダをスキャンしてインデックスを作成
        self.samples = []
        traj_names_file = self.data_folder / "traj_names.txt"
        if not traj_names_file.exists():
            raise FileNotFoundError(f"traj_names.txt not found in {data_folder}")
            
        with open(traj_names_file, "r") as f:
            # 各行に書かれた軌跡フォルダ名を読み込む
            traj_names = [line.strip() for line in f if line.strip()]
            
        self.trajectory_data = {}
        for traj_name in traj_names:
            traj_path = self.data_folder / traj_name
            pkl_path = traj_path / "traj_data.pkl"
            if not pkl_path.exists():
                print(f"Warning: {pkl_path} not found. Skipping.")
                continue
            
            # 軌跡データ（絶対座標の時系列）をメモリにロード
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                self.trajectory_data[traj_name] = data
            
            num_frames = len(data["position"])
            # 未来の horizon ステップ分確保できるフレームだけを学習対象にする
            for i in range(num_frames - self.horizon):
                self.samples.append((traj_name, i))
                
        print(f"Loaded Local_Dataset with {len(self.samples)} samples from {len(self.trajectory_data)} trajectories.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, traj_name: str, frame_idx: int) -> Image.Image:
        # フォルダから画像（.jpg または .png）を探して開く
        img_path = self.data_folder / traj_name / "images" / f"{frame_idx:05d}.jpg"
        if not img_path.exists():
             img_path = self.data_folder / traj_name / "images" / f"{frame_idx:05d}.png"
        
        img = Image.open(img_path).convert("RGB")
        return img

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """
        学習ループから呼ばれるメインのメソッド。
        1つの訓練データ（画像 ＋ 命令文 ＋ 正解アクション）を作成して返します。
        """
        traj_name, curr_idx = self.samples[i]
        
        # 1. 画像のロードとリサイズ
        image = self._load_image(traj_name, curr_idx)
        pixel_values = self.image_transform(image)
        
        # 2. 正解アクション（未来の軌跡）の計算
        data = self.trajectory_data[traj_name]
        positions = data["position"] # 時系列の絶対X,Y座標
        yaws = data["yaw"]           # 時系列の向き（ヨー角）
        
        curr_pos = positions[curr_idx]
        curr_yaw = yaws[curr_idx]
        if isinstance(curr_yaw, np.ndarray): curr_yaw = curr_yaw.item()
        
        # 「現在」から数えて未来の poses を horizon 個分取り出す
        future_poses = positions[curr_idx + 1 : curr_idx + self.horizon + 1]
        future_yaws = yaws[curr_idx + 1 : curr_idx + self.horizon + 1]
        
        # 【重要】絶対座標をロボット中心の相対座標に変換
        # これにより、ロボットは「自分が今どこにいても」相対的にどっちへ進めばいいかを学べます
        waypoints = to_local_coords(future_poses, curr_pos, curr_yaw)
        
        # アクションの組み立て: [dx, dy, cos(yaw), sin(yaw)]
        actions = []
        for j in range(len(waypoints)):
            rel_yaw = future_yaws[j] - curr_yaw
            if isinstance(rel_yaw, np.ndarray): rel_yaw = rel_yaw.item()
            
            actions.append([
                waypoints[j, 0],   # 前方向への移動量
                waypoints[j, 1],   # 横方向への移動量
                np.cos(rel_yaw),   # 未来の向き（cos）
                np.sin(rel_yaw)    # 未来の向き（sin）
            ])
        
        actions = torch.tensor(actions, dtype=torch.float32) # (horizon, 4)
        
        # 3. 動作内容に基づいて言語指示を自動生成
        # 未来の軌跡の累積ヨー角変化からどちらに曲がるかを判断する
        yaw_diffs = []
        for j in range(len(actions)):
            rel_yaw = float(future_yaws[j]) - float(curr_yaw.item() if isinstance(curr_yaw, np.ndarray) else curr_yaw)
            # -pi〜pi に正規化
            rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi
            yaw_diffs.append(rel_yaw)
        
        total_yaw_change = float(np.sum(yaw_diffs))
        yaw_threshold = 0.15  # ラジアン（約8.6度）以上の変化があれば「曲がっている」と判定
        
        if total_yaw_change > yaw_threshold:
            lang_prompt = "turn left"
        elif total_yaw_change < -yaw_threshold:
            lang_prompt = "turn right"
        else:
            lang_prompt = "go straight"
        
        # 命令文（プロンプト）の作成
        # ロボットへの指示と、それに対する「返答（アクション）」という会話形式で構成
        action_chunk_string = "".join(self.action_tokenizer(actions))
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": lang_prompt},        # 動作に応じた命令
            {"from": "gpt", "value": action_chunk_string},  # 正解アクション
        ]
        
        # OpenAI/Llamaなどの指定されたモデルの書式に合わせて命令文を整形
        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # 言葉を ID（input_ids）に変換
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   
        
        # マスクの作成：指示文の部分では学習せず、アクション（GPTの返答）の部分だけで誤差を計算するようにする
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # 最終的にモデルに渡す辞書
        return dict(
            pixel_values=pixel_values,         # 入力画像
            input_ids=input_ids,               # プロンプトID
            labels=labels,                     # 正解データ（誤差計算用）
            actions=actions,                   # 生アクション（可視化やデバッグ用）
            dataset_name="local",
            modality_id=6,                     # モダリティ設定 (画像＋ナビゲーション)
            action_select_mask=torch.tensor(1.0),
            img_PIL=image,
            lan_prompt=lang_prompt
        )
