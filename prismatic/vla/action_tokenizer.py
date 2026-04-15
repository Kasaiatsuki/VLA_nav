"""
action_tokenizer.py

ロボットの連続的な動きを離散化し、言語モデルのトークンに変換するためのクラスです。
"""

from typing import List, Union
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

class ActionTokenizer:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        bins: int = 256, 
        min_action: float = -1.0, 
        max_action: float = 1.0
    ) -> None:
        """
        連続的なロボットのアクションを N 個のビンに分割し、語彙の末尾にあるトークンにマッピングします。

        :param tokenizer: ベースとなる LLM/VLM のトークナイザー。
        :param bins: 各連続値に対する分割数（ビン数）。
        :param min_action: アクションの最小値（これ以下の値はクリップされます）。
        :param max_action: アクションの最大値（これ以上の値はクリップされます）。
        """
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # 1. 均等なビン（仕切り）を作成し、各ビンの中心値を計算します
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # 2. アクション用トークンの開始インデックスを設定します
        #    通常、語彙（vocab）の最後の方にある、あまり使われないトークンを再利用します。
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def encode_action(self, action: np.ndarray) -> np.ndarray:
        """
        連続値のアクションを、対応するトークン ID（整数）に変換します。
        """
        # 値を最小・最大範囲内に収める
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        
        # どのビンに属するかを判定 (1 〜 self.n_bins の値が返ります)
        discretized_action = np.digitize(action, self.bins)
        
        # ビン番号をトークナイザーの語彙インデックス（後ろからのオフセット）に変換
        return (self.tokenizer.vocab_size - discretized_action).astype(np.int64)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        トークン ID（整数）を、元の連続値のアクション（数値）に逆変換します。
        """
        # ID から逆算して何番目のビンかを取得
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        
        # インデックス範囲を [0, bin_centers.shape[0]-1] に収める
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        # ビンの中心値を返す
        return self.bin_centers[discretized_actions]

    def __call__(self, actions: Union[torch.Tensor, np.ndarray]) -> str:
        """
        PyTorch テンソルまたは NumPy 配列を受け取り、トークン化された文字列を返します。
        データセット作成時の `action_chunk_string = "".join(self.action_tokenizer(actions))` で使われます。
        """
        # 入力がテンソルの場合は NumPy に変換
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
            
        # 数値を ID へ変換
        token_ids = self.encode_action(actions)
        
        # ID の羅列を、一つの文字列としてデコードして返します
        # flatten() を使うことで、多次元配列（軌跡）を一本のトークン列にします
        return self.tokenizer.decode(token_ids.flatten())

    @property
    def vocab_size(self) -> int:
        return self.n_bins
