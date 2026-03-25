import os
import sys
import torch
import draccus
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# 分散学習やメモリ効率化のためのライブラリ
import torch.distributed as dist
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model  # LoRA (低ランク適応) 用
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard ログ用
from transformers import AutoProcessor, AdamW, get_cosine_schedule_with_warmup

# プロジェクトのルートディレクトリをパスに追加して自作モジュールを読み込めるようにする
sys.path.append(os.getcwd())

# Prismatic (OmniVLAの基盤) 関連のコンポーネント
from prismatic.models.load import load_vla
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.local_dataset import Local_Dataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

@dataclass
class LocalTrainingConfig:
    """
    学習のハイパーパラメータ（設定値）を管理するクラスです。
    実行時にコマンドライン引数で上書きすることも可能です。
    """
    # モデルとデータのパス設定
    vla_path: str = "openvla/openvla-7b"         # ベースとなるVLAモデルの名前またはパス
    data_dir: str = "path/to/your/dataset"      # ZEDカメラで収集したデータの場所
    run_id: str = "local-vla-finetune"          # 今回の実験に付ける名前
    
    # 学習の基本パラメータ
    batch_size: int = 2                         # 1回にGPUに送るデータの数（メモリ不足なら 1 に下げる）
    grad_accumulation_steps: int = 4            # 勾配を溜めてから更新する回数（実質的なバッチサイズを増やす）
    learning_rate: float = 2e-5                 # 学習率（重みをどれくらい大きく動かすか）
    max_steps: int = 1000                       # 最大何ステップ学習を回すか
    save_steps: int = 250                       # 何ステップごとにモデルを保存するか
    
    # LoRA (低ランク適応) の設定：巨大なモデルの「差分」だけを賢く学習する手法
    use_lora: bool = True
    lora_rank: int = 32                         # LoRA行列のランク（大きいほど表現力が増すがメモリも使う）
    lora_alpha: int = 16                        # スケーリング係数
    lora_dropout: float = 0.05                  # 過学習を防ぐためのドロップアウト率
    
    # 最適化とスケジュール
    warmup_steps: int = 50                      # 序盤に学習率を徐々に上げるステップ数
    weight_decay: float = 0.0                   # 正則化の強さ
    
    # ハードウェア設定
    mixed_precision: str = "fp16"               # メモリ節約のための演算精度（bf16 または fp16）
    device: str = "cuda"                        # 使用デバイス (cuda/cpu)
    use_4bit: bool = True                       # 4-bit 量子化を使用するか (8GB VRAM 等の低メモリ環境で必須)
    num_workers: int = 8                        # データロードのワーカー数
    
    # 出力先
    output_dir: str = "checkpoints"             # 学習後のモデルを保存するフォルダ
    resume_from_checkpoint: Optional[str] = None  # 追加: チェックポイントから再開する場合のフォルダパス

@draccus.wrap()
def train(cfg: LocalTrainingConfig):
    # 1. Accelerator の初期化 (複数GPUや精度の管理を自動化してくれる便利なツール)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
    )
    device = accelerator.device  # 実行デバイス (CUDAなど)

    # TensorBoard のライター初期化（実行ごとにユニークなフォルダを作成）
    from datetime import datetime
    writer = None
    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        tb_log_dir = Path(cfg.output_dir) / cfg.run_id / "tensorboard" / current_time
        writer = SummaryWriter(log_dir=str(tb_log_dir), flush_secs=10)
        print(f"TensorBoard logs: {tb_log_dir}")
        print(f"  → 全体の確認コマンド: tensorboard --logdir {Path(cfg.output_dir) / cfg.run_id / 'tensorboard'}")
    
    # 2. VLAモデルとプロセッサのロード
    # load_vla は指定されたパスから Hugging Face のモデルをダウンロード/読み込みます
    # 4-bit 量子化を有効にすることで、8GB 程度の GPU メモリでもロード可能にします
    model, processor = load_vla(cfg.vla_path, load_for_training=True, use_4bit=cfg.use_4bit)
    
    # 勾配チェックポインティングを有効にしてメモリを節約 (8GB VRAM では必須)
    model.llm_backbone.llm.gradient_checkpointing_enable()
    model.llm_backbone.llm.enable_input_require_grads()
    
    # Vision Backbone は重みを動かさないので eval モードにし、fp16 に固定
    model.vision_backbone.eval()
    if cfg.mixed_precision == "fp16":
        model.vision_backbone.to(torch.float16)
        model.projector.to(torch.float16) # Projector も小規模だが念のため
    
    # LoRA を適用：モデル全体の約1%以下のパラメータだけを訓練対象にします
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            # MLP レイヤー (gate/up/down_proj) は 8GB VRAM では逆伝播時に OOM になるため除外
            # アテンション層だけに LoRA を適用することでメモリを大幅削減
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 3. データセットとデータローダーの作成
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    # 自作した Local_Dataset を使用してZEDデータを読み込む
    dataset = Local_Dataset(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        data_folder=cfg.data_dir,
    )
    
    # Collator：複数のデータを1つのバッチにまとめる際、長さの足りない部分を埋める(Padding)処理
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,                  # データの順番をシャッフルする
        collate_fn=collator,
        num_workers=0,
    )

    # 4. オプティマイザ (最適化手法) と スケジューラ (学習率の変化) の設定
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # コサインカーブを描いて学習率を減衰させるスケジューラ
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    # 5. Accelerator に登録
    # 注意: 4-bit 量子化モデルは既に GPU に乗っているため、accelerator.prepare での自動移動をスキップして
    # 重複したメモリ確保を避けます。オプティマイザとローダーのみ登録します。
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    optimizer, train_loader, lr_scheduler = accelerator.prepare(
        optimizer, train_loader, lr_scheduler
    )
    # モデルは手動でデバイスに合わせる（既に CUDA にあるはずですが念のため）
    model = model.to(device)

    # 5.5 チェックポイントからの再開
    step = 0
    if cfg.resume_from_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"Loading checkpoint from {cfg.resume_from_checkpoint}...")
        
        # モデルの重みが保存されている場合は読み込む (PEFT/LoRA)
        checkpoint_path = Path(cfg.resume_from_checkpoint)
        if (checkpoint_path / "adapter_model.bin").exists() or (checkpoint_path / "adapter_model.safetensors").exists():
            if accelerator.is_main_process:
                print(f"  -> Loading adapter weights...")
            # model.load_adapter は PEFT モデルに新しいアダプターをロードします
            model.load_adapter(cfg.resume_from_checkpoint)
        
        # オプティマイザやスケジューラの状態を復元
        accelerator.load_state(cfg.resume_from_checkpoint)
        
        # ステップ数をフォルダ名から抽出 (例: "checkpoint-1900" -> 1900)
        try:
            step = int(checkpoint_path.name.split("-")[-1])
            if accelerator.is_main_process:
                print(f"  -> Resuming from step {step}")
        except (ValueError, IndexError):
            if accelerator.is_main_process:
                print(f"  -> Could not parse step number from {checkpoint_path.name}, starting from 0")
            step = 0

    # 6. 学習ループ
    model.train()
    print(f"Starting training for {cfg.max_steps} steps...")
    
    # 学習開始前に残留メモリを解放
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    while step < cfg.max_steps:
        for batch in train_loader:
            if step >= cfg.max_steps: break
            
            # 勾配アキュムレーション（メモリ節約テクニック）のスコープ
            with accelerator.accumulate(model):
                # pixel_values が dict (DinoSigLIP) の場合は fp16 にキャスト
                pv = batch["pixel_values"]
                if isinstance(pv, dict):
                    pv = {k: v.to(device=device, dtype=torch.float16) for k, v in pv.items()}
                else:
                    pv = pv.to(device=device, dtype=torch.float16)
                # input_ids と labels はデバイスに移す
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                # 順伝播：モデルに画像と言葉を投げ、出力と誤差(Loss)を計算
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pv,
                    labels=labels,
                )
                loss = outputs.loss
                
                # 逆伝播：どれくらい間違っていたかを元に勾配を計算
                accelerator.backward(loss)
                optimizer.step()       # 重みの更新
                lr_scheduler.step()    # 学習率の更新
                optimizer.zero_grad()  # 蓄積された勾配をリセット
                
                # ここから重要：実際に重みの更新が行われた際のみ実行
                if accelerator.sync_gradients:
                    step += 1
                    
                    # 進捗の表示（メインプロセスのみ）
                    if accelerator.is_main_process:
                        if step % 10 == 0:
                            loss_val = loss.item()
                            lr_val = lr_scheduler.get_last_lr()[0]
                            print(f"Step {step}: Loss = {loss_val:.4f}")
                            # TensorBoard にロスと学習率を記録
                            writer.add_scalar("train/loss", loss_val, step)
                            writer.add_scalar("train/learning_rate", lr_val, step)
                            
                        # 定期的なモデルの保存
                        if step > 0 and step % cfg.save_steps == 0:
                            save_path = Path(cfg.output_dir) / cfg.run_id / f"checkpoint-{step}"
                            accelerator.save_state(save_path)
                            
                            # 追加: 推論用の重みファイル (adapter_model.bin) も保存
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(save_path)
                            print(f"Saved checkpoint and model weights to {save_path}")


            pass  # step increment moved inside sync_gradients

    if accelerator.is_main_process:
        writer.close()
    print("Training complete!")

if __name__ == "__main__":
    train()
