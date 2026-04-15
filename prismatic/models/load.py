"""
load.py

推論や学習のために学習済み VLM/VLA モデルをロードするためのエントリポイントです。
ローカルディスクまたは HuggingFace Hub からのロードをサポートしています。
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import HfFileSystem, hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlas import OpenVLA
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Overwatch の初期化：ロギング（情報の出力）を管理します
overwatch = initialize_overwatch(__name__)


# === HuggingFace Hub のリポジトリ設定 ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"
VLA_HF_HUB_REPO = "openvla/openvla-dev"


def available_models() -> List[str]:
    """利用可能なモデル ID の一覧を返します"""
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    """利用可能なモデルの別名一覧を返します"""
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    """特定のモデルの説明を取得します"""
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))
    return description


# === 通常の VLM (Vision-Language Model) をロードする関数 ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> PrismaticVLM:
    """ローカルまたは HF Hub から学習済み PrismaticVLM をロードします。"""
    
    # 1. ローカルパスか HF Hub かを判別してチェックポイントを特定
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # 2. 設定ファイル (config.json) の読み込み
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # 3. Vision Backbone（画像の目）をロード
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # 4. LLM Backbone（言語の脳）をロード
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
        use_4bit=use_4bit,
    )

    # 5. すべてを統合して VLM インスタンスを作成
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )

    return vlm


# === VLA (Vision-Language-Action) モデルをロードする関数：今回の学習で使用 ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    step_to_load: Optional[int] = None,
    model_type: str = "pretrained",
    use_4bit: bool = False,
) -> OpenVLA:
    """ローカルまたは HF Hub から学習済み OpenVLA をロードします。"""

    # 1. ローカルのチェックポイントファイル (.pt) が指定された場合
    if os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")
        assert (checkpoint_pt.suffix == ".pt") and (checkpoint_pt.parent.name == "checkpoints")
        run_dir = checkpoint_pt.parents[1]
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"

    # 2. 指定（"openvla-7b" など）に基づいて HF Hub からダウンロードする場合
    else:
        # もし model_id_or_path に "/" が含まれている場合は、フルリポジトリIDとみなす
        if "/" in str(model_id_or_path):
            hf_repo_id = str(model_id_or_path)
            hf_path = hf_repo_id
        else:
            hf_repo_id = VLA_HF_HUB_REPO
            hf_path = str(Path(VLA_HF_HUB_REPO) / model_type / model_id_or_path)

        overwatch.info(f"Checking HF for `{hf_path}`")
        if not (tmpfs := HfFileSystem()).exists(hf_path):
            raise ValueError(f"Couldn't find valid HF Hub Path `{hf_path = }`")

        # リポジトリ構造に応じて設定ファイルとチェックポイントを取得
        with overwatch.local_zero_first():
            # ケースA: 公式の openvla/openvla-7b などの構造
            if hf_repo_id == hf_path:
                # config.json はルートにあるはず
                config_json = hf_hub_download(repo_id=hf_repo_id, filename="config.json", cache_dir=cache_dir)
                # dataset_statistics.json は公式リポジトリにはない場合があるが、config.json 内に含まれている
                # 一旦ダミーとして扱うか、config.json から抽出する
                dataset_statistics_json = None 
                # チェックポイントは .safetensors (AutoModel がハンドルする)
                checkpoint_pt = hf_repo_id # HF リポジトリIDをそのまま渡す
            
            # ケースB: 著者の prismatic 開発用構造 (openvla-dev)
            else:
                target_ckpt = Path(tmpfs.glob(f"{hf_path}/checkpoints/step-{step_to_load:06d if step_to_load else ''}*.pt")[-1]).name
                relpath = Path(model_type) / model_id_or_path
                config_json = hf_hub_download(repo_id=hf_repo_id, filename=f"{(relpath / 'config.json')!s}", cache_dir=cache_dir)
                dataset_statistics_json = hf_hub_download(repo_id=hf_repo_id, filename=f"{(relpath / 'dataset_statistics.json')!s}", cache_dir=cache_dir)
                checkpoint_pt = hf_hub_download(repo_id=hf_repo_id, filename=f"{(relpath / 'checkpoints' / target_ckpt)!s}", cache_dir=cache_dir)

    # 3. VLA 設定と統計情報のロード
    with open(config_json, "r") as f:
        cfg_data = json.load(f)
    
    # 公式のリポジトリ (openvla/openvla-7b) の場合、構造が違うので変換
    if "norm_stats" in cfg_data:
        from types import SimpleNamespace
        norm_stats = cfg_data["norm_stats"]
        vla_cfg = {
            "base_vlm": cfg_data.get("model_type", "openvla"),
        }
        # ModelConfig が期待するキーをセット (SimpleNamespace で代用)
        model_cfg = SimpleNamespace()
        model_cfg.model_id = cfg_data.get("model_id", "openvla-7b")
        model_cfg.vision_backbone_id = cfg_data["vision_backbone_id"]
        model_cfg.llm_backbone_id = cfg_data["llm_backbone_id"]
        model_cfg.arch_specifier = cfg_data["arch_specifier"]
        model_cfg.image_resize_strategy = cfg_data["image_resize_strategy"]
        model_cfg.llm_max_length = cfg_data.get("llm_max_length", 2048)
    else:
        vla_cfg = cfg_data["vla"]
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # 4. Vision Backbone のロード
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # 5. LLM Backbone のロード
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
        use_4bit=use_4bit,
    )

    # 6. アクション用トークナイザーの作成（数値をトークンに変換するパーツ）
    action_tokenizer = ActionTokenizer(tokenizer)

    # 7. すべてを組み合わせて OpenVLA インスタンスを作成
    vla = OpenVLA(
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=True,
        arch_specifier=model_cfg.arch_specifier,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
    )

    # 8. 重みのロード (最後の仕上げ)
    # ケースA: 公式のリポジトリの場合 (バックボーン初期化時に既にロードされているはず)
    if "/" in str(checkpoint_pt) and not os.path.isfile(checkpoint_pt):
        overwatch.info(f"Using weights from official HF repo: {checkpoint_pt}")
        # 公式リポジトリの場合、llm_backbone_id が正しければバックボーンが重みをロード済みです
    
    # ケースB: 著者の .pt チェックポイントの場合 (明示的にロード)
    elif os.path.isfile(checkpoint_pt):
        overwatch.info(f"Loading weights from local checkpoint: {checkpoint_pt}")
        state_dict = torch.load(checkpoint_pt, map_location="cpu")["model"]
        vla.projector.load_state_dict(state_dict["projector"])
        vla.llm_backbone.load_state_dict(state_dict["llm_backbone"])
        if "vision_backbone" in state_dict:
             vla.vision_backbone.load_state_dict(state_dict["vision_backbone"])
    
    # プロセッサ（Tokenizer と ImageProcessor のラップ）を作成して返す
    from types import SimpleNamespace
    class VLAProcessor:
        def __init__(self, tokenizer, image_transform):
            self.tokenizer = tokenizer
            self.image_processor = SimpleNamespace(apply_transform=image_transform)

    return vla, VLAProcessor(tokenizer, image_transform)
