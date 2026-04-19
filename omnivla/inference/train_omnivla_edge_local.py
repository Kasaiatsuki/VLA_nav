import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model_omnivla_edge import OmniVLA_edge
from vla_nav_dataset_local import OmniVLAEdgeDataset

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import draccus
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EdgeTrainingConfig:
    """
    学習のハイパーパラメータ（設定値）を管理するデータクラスです。
    YAMLファイル（例: config_nav/finetune_edge_local.yaml）から値を読み込みます。
    """
    vla_path: str = "/home/orne/kasai_ws/src/VLA_nav/omnivla/inference/omnivla-edge/omnivla-edge.pth" # 学習元となる重みファイルのパス
    data_dir: str = "/home/orne/kasai_ws/src/VLA_nav/data/20260408_191058_combined_dataset_t_2"       # 学習用データセットのディレクトリ
    output_dir: str = "checkpoints_edge"                                                                 # 学習後の重みを保存するディレクトリ
    run_id: str = "edge-vla-finetune-v1"                                                                 # 今回の学習の識別コード
    
    batch_size: int = 4            # バッチサイズ（一度にGPUに送るデータ数）
    epochs: int = 10               # エポック数（データセット全体を何周学習するか）
    learning_rate: float = 1e-4    # 学習率（一回の学習で重みをどれくらい大きく更新するか）
    warmup_steps: int = 50         # 指定したステップ数までは学習率を徐々に上げ、その後ゆっくり下げる
    prompt: str = "move forward avoiding walls" # モデルに与える前提となる言語指示

@draccus.wrap()
def train(cfg: EdgeTrainingConfig):
    # --- Configuration ---
    DATA_ROOT = cfg.data_dir
    MODEL_PATH = cfg.vla_path 
    
    save_dir = Path(cfg.output_dir) / cfg.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = str(save_dir / "omnivla-edge-finetuned.pth")
    
    BATCH_SIZE = cfg.batch_size
    EPOCHS = cfg.epochs
    LR = cfg.learning_rate
    prompt_text = cfg.prompt
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CuDNNのシステムライブラリエラーを回避するため、CuDNNを無効化（PyTorchネイティブの処理にフォールバック）
    import torch.backends.cudnn as cudnn
    cudnn.enabled = False
    
    # --- Model Initialization (モデルの初期化) ---
    # EfficientNetをバックボーン（画像特徴抽出）とする軽量なモデル(OmniVLA_edge)を定義します
    model = OmniVLA_edge(
        context_size=5, 
        len_traj_pred=8, 
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=1024
    )
    
    # Load weights if available (既存モデルの読み込み分岐)
    # これが「ファインチューニングか」「一からのスクラッチ学習か」を決める部分です。
    if os.path.exists(MODEL_PATH):
        # 指定したパスに重みファイルがあれば、それを読み込んで続きから学習させます (ファインチューニング)
        print(f"Loading weights from {MODEL_PATH} (既存の重みを読み込んでファインチューニングを開始します)")
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        # 重みが見つからない場合は、ランダムな初期値から全く新しく学習します (スクラッチ学習・一から学習)
        print(f"Warning: {MODEL_PATH} not found. Starting from scratch or random weights. (エラー: 重みファイルが見つからないため、一から学習を開始します)")
    
    model.to(DEVICE)
    model.train()
    
    # --- Goal & Mask Preparation (目標と言語指示の準備) ---
    # 通常のOmniVLAは多様な指示を受け取りますが、本スクリプトでは固定の文字列(prompt_text)として学習します。
    # CLIPモデルを使って文字列をベクトル（特徴量）に変換します。
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    text_token = clip.tokenize([prompt_text]).to(DEVICE)
    with torch.no_grad():
        feat_text = clip_model.encode_text(text_token).float()
    
    # Dummy inputs as used in vla_nav_node.py
    goal_pose = torch.zeros((1, 4)).to(DEVICE)
    map_images = torch.zeros((1, 9, 192, 192)).to(DEVICE)
    goal_img = torch.zeros((1, 3, 192, 192)).to(DEVICE)
    modality_id = torch.tensor([7]).to(DEVICE) # Language mode ID
    
    # --- Dataset & Dataloader (データ準備) ---
    # 指定されたディレクトリ(DATA_ROOT)から、画像と目標軌道のセットを読み込めるようにします。
    dataset = OmniVLAEdgeDataset(DATA_ROOT, context_size=5, len_traj_pred=8)
    if len(dataset) == 0:
        print("Error: No samples found in dataset. Check DATA_ROOT.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,         # 画像読み込みを並列化して高速化
        pin_memory=True        # GPUへのデータ転送を高速化
    )
    
    # --- Optimizer & Loss ---
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss() 
    
    # --- Training Loop (学習ループ) ---
    # 実行時にどのような設定で学習が行われるかを表示します
    print("="*60)
    print(f"▶ 使用する学習データ: {DATA_ROOT}")
    print(f"▶ 目標言語指示      : '{prompt_text}'")
    print(f"▶ 学習設定          : バッチサイズ={BATCH_SIZE}, エポック数={EPOCHS}, 学習率={LR}")
    print(f"▶ モデル保存先      : {SAVE_PATH}")

    # TensorBoard の初期設定
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_log_dir = save_dir / "tensorboard" / current_time
    writer = SummaryWriter(log_dir=str(tb_log_dir), flush_secs=60)
    print(f"▶ TensorBoardログ   : {tb_log_dir}")
    print("="*60)
    print(f"※ 確認用コマンド: tensorboard --logdir {save_dir.resolve() / 'tensorboard'} --bind_all")
    print("="*60)
    
    print(f"Starting fine-tuning on {DEVICE}... ({DEVICE} デバイスを用いて学習を開始します)")
    
    # 学習率スケジューラの初期化 (コサインカーブで滑らかに徐々に減衰させる)
    from transformers import get_cosine_schedule_with_warmup
    total_optimization_steps = len(dataloader) * EPOCHS
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optimization_steps
    )
    
    global_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for obs_stack, action_label in pbar:
            # データをGPU等の実行環境に転送
            obs_stack = obs_stack.to(DEVICE)       # 入力画像（履歴スタック）
            action_label = action_label.to(DEVICE) # 正解の行動（予測させたい未来の軌道）
            
            # current_img is the latest frame in the stack (last 3 channels)
            # 履歴画像スタックの中から、一番最新のRGB画像フレームを抽出
            current_img = obs_stack[:, -3:, :, :]
            
            # Forward pass (順伝播：モデルに現在の画像や指示を入力し、予測結果を出力させる)
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
            
            # waypoints is the first return value (推論された予測軌道)
            # outputs = (action_pred, dist_pred, no_goal_mask)
            waypoints = outputs[0]
            
            # 正解データ(action_label)の次元が [B, 8, 4, 1] など余分な次元がある場合に合わせて揃える
            if len(action_label.shape) == 4 and action_label.shape[-1] == 1:
                action_label = action_label.squeeze(-1)
            
            # Action prediction loss (損失計算: モデルの予測軌道と、正解の軌道のズレを計算)
            loss = criterion(waypoints, action_label)
            
            # Backward pass (逆伝播: ズレから重みの修正量を計算し、実際にモデルを更新する)
            optimizer.zero_grad()  # 過去の勾配をリセット
            loss.backward()        # 誤差逆伝播
            optimizer.step()       # 重みの更新
            lr_scheduler.step()    # スケジューラの方針に従って学習率を徐々に下げる
            
            total_loss += loss.item()
            global_step += 1
            
            # 最初のステップ(1) または 5ステップごとにTensorBoardに損失と学習率を記録
            if global_step == 1 or global_step % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)
                writer.flush()  # 書き込みを即座に反映させる
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")
        
        # Save every epoch (一回り学習が終わるごとにモデルの重みを保存する)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved checkpoint to {SAVE_PATH}")

    # 学習終了時にTensorBoardの書き込みを完了させる
    writer.close()

if __name__ == "__main__":
    train()
