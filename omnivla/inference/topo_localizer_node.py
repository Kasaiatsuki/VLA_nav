#!/usr/bin/env python3
"""
topo_localizer_node.py  (ハイブリッドモード版)

役割:
  PlaceNet + ベイズフィルタでトポロジカルマップ上の自己位置を推定し、
  OmniVLA-edge に対して「言語指示」と「次ノードのビジュアルゴール画像」の
  両方を同時に配信するハイブリッドナビゲーションノード。

モデル内部の動作:
  - 言語   : lan_prompt → CLIP → FiLM → ResNet → goal_encoding_lan → Transformer
  - ビジュアル : goal_image → EfficientNet-B0(6ch) → goal_encoding_img → Transformer
  - modality_id=5 を使用することで両方のトークンが Transformer に入力される
    (all_masks[5] = goal_mask_4 : map チャンネルのみマスク)

Subscribes:
  /image_raw          (sensor_msgs/Image) - 単眼カメラ画像
  /autonomous         (std_msgs/Bool)     - 自律走行フラグ

Publishes:
  /lan_prompt         (std_msgs/String)   - VLA への言語指示
  /goal_image         (sensor_msgs/Image) - VLA への次ノードの参照画像
  /autonomous         (std_msgs/Bool)     - stop ノード到達時に False を送信
  /topo_nav/current_node (std_msgs/Int32) - 推定ノード ID（デバッグ用）
  /topo_nav/action    (std_msgs/String)   - 推定ノードのアクション（デバッグ用）
"""

import os
import sys
import math
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, Int32
from cv_bridge import CvBridge
import cv2
import torch
from PIL import Image as PILImage

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ============================================================
# アクション → 言語指示 変換テーブル
# ============================================================
ACTION_TO_PROMPT: dict[str, str] = {
    "straight": "go straight",
    "left":     "turn left",
    "right":    "turn right",
}


class TopoLocalizerNode(Node):
    """PlaceNet + ベイズフィルタによるトポロジカル自己位置推定＋ハイブリッド配信ノード。"""

    def __init__(self) -> None:
        super().__init__('topo_localizer_node')

        # -------------------------------------------------------
        # パラメータ宣言
        # -------------------------------------------------------
        self.declare_parameter(
            'map_path',
            '/home/kasai/ros2_ws/install/imitation_nav/share/imitation_nav'
            '/config/topo_map/topomap.yaml')
        self.declare_parameter(
            'model_path',
            '/home/kasai/ros2_ws/install/imitation_nav/share/imitation_nav'
            '/weights/placenet/placenet.pt')
        self.declare_parameter(
            'image_dir',
            '/home/kasai/ros2_ws/install/imitation_nav/share/imitation_nav'
            '/config/topo_map/images/')
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('window_lower', -2)
        self.declare_parameter('window_upper', 3)
        self.declare_parameter('use_observation_based_init', True)
        self.declare_parameter('delta', 5.0)
        # ゴール画像を何ノード先にするか（1=直近の次ノード）
        self.declare_parameter('goal_node_offset', 1)

        map_path       = self.get_parameter('map_path').value
        model_path     = self.get_parameter('model_path').value
        self.image_dir = self.get_parameter('image_dir').value
        camera_topic   = self.get_parameter('camera_topic').value
        self.window_lower     = self.get_parameter('window_lower').value
        self.window_upper     = self.get_parameter('window_upper').value
        self.use_obs_init     = self.get_parameter('use_observation_based_init').value
        self.delta            = float(self.get_parameter('delta').value)
        self.goal_node_offset = int(self.get_parameter('goal_node_offset').value)

        # -------------------------------------------------------
        # 内部状態
        # -------------------------------------------------------
        self.bridge           = CvBridge()
        self.device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_initialized   = False
        self.is_autonomous    = False
        self.belief:    list[float] = []
        self.transition: list[float] = []
        self.lambda1:   float = 1.0
        self.stopped_node_ids: set[int] = set()

        # 前回配信したノード ID（変化した時だけゴール画像を再配信するためのキャッシュ）
        self._prev_node_id: int = -1
        # ゴール画像のキャッシュ（ノード ID → RGB numpy array）
        self._goal_img_cache: dict[int, np.ndarray] = {}

        # -------------------------------------------------------
        # マップ読み込み
        # -------------------------------------------------------
        self.map_nodes: list[dict] = []
        self._load_map(map_path)

        # -------------------------------------------------------
        # PlaceNet モデル読み込み
        # -------------------------------------------------------
        if os.path.exists(model_path):
            self.get_logger().info(f'Loading PlaceNet from {model_path} ...')
            self.placenet = torch.jit.load(model_path)
            self.placenet.to(self.device)
            self.placenet.eval()
            self.get_logger().info(f'PlaceNet loaded on {self.device}.')
        else:
            self.get_logger().error(f'PlaceNet not found: {model_path}')
            self.placenet = None

        # -------------------------------------------------------
        # 遷移モデルの初期化
        # -------------------------------------------------------
        self._setup_transition()

        # -------------------------------------------------------
        # Publisher / Subscriber
        # -------------------------------------------------------
        self.prompt_pub     = self.create_publisher(String, '/lan_prompt',              10)
        self.goal_img_pub   = self.create_publisher(Image,  '/goal_image',              10)
        self.autonomous_pub = self.create_publisher(Bool,   '/autonomous',               10)
        self.node_id_pub    = self.create_publisher(Int32,  '/topo_nav/current_node',   10)
        self.action_pub     = self.create_publisher(String, '/topo_nav/action',          10)

        self.create_subscription(Image, camera_topic,  self._image_callback,      10)
        self.create_subscription(Bool,  '/autonomous', self._autonomous_callback,  10)

        self.get_logger().info(
            f'[TopoLocalizer] Ready. nodes={len(self.map_nodes)}, '
            f'goal_offset={self.goal_node_offset}, device={self.device}')

    # ===========================================================
    # マップ読み込み
    # ===========================================================

    def _load_map(self, map_path: str) -> None:
        """YAML のトポロジカルマップを読み込む。
        各ノードの id, image, action（edges[0].action）, edge_target（edges[0].target）を保持。
        """
        self.get_logger().info(f'Loading map: {map_path}')
        with open(map_path, 'r') as f:
            root = yaml.safe_load(f)

        for node in root['nodes']:
            action      = 'straight'
            edge_target = None  # 次に進むべきノードID（エッジのターゲット）

            if node.get('edges') and len(node['edges']) > 0:
                action      = node['edges'][0].get('action', 'straight')
                edge_target = node['edges'][0].get('target', None)

            self.map_nodes.append({
                'id':          node['id'],
                'image':       node.get('image', ''),
                'action':      action,
                'edge_target': edge_target,            # 次ノードID（Noneの場合は終端）
                'feature':     np.array(node['feature'], dtype=np.float32),
            })

        # id → インデックスの逆引き辞書（高速アクセス用）
        self._id_to_idx: dict[int, int] = {n['id']: i for i, n in enumerate(self.map_nodes)}
        self.get_logger().info(f'Loaded {len(self.map_nodes)} nodes.')

    # ===========================================================
    # 遷移モデル
    # ===========================================================

    def _setup_transition(self) -> None:
        """ガウス分布を用いた遷移確率ベクトルを生成する。"""
        sigma   = 1.0
        offsets = range(self.window_lower, self.window_upper + 1)
        weights = [math.exp(-0.5 * (o / sigma) ** 2) for o in offsets]
        total   = sum(weights)
        self.transition = [w / total for w in weights]

    # ===========================================================
    # 特徴量抽出
    # ===========================================================

    def _extract_feature(self, cv_bgr: np.ndarray) -> np.ndarray:
        """BGR 画像（np.ndarray）から PlaceNet で特徴量を抽出する。"""
        rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb).resize((85, 85))
        arr = np.array(pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr  = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.placenet.forward([tensor]).squeeze(0)
        return feat.cpu().numpy()

    # ===========================================================
    # ベイズフィルタ
    # ===========================================================

    def _cos_distance(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """正規化特徴量間のコサイン距離（0〜2）。"""
        dot = float(np.dot(f1, f2))
        return math.sqrt(max(0.0, 2.0 - 2.0 * dot))

    def _initialize_belief(self, query_feat: np.ndarray) -> None:
        """最初のフレームで信念分布を初期化し lambda1 を決定する。"""
        n     = len(self.map_nodes)
        dists = [self._cos_distance(query_feat, node['feature']) for node in self.map_nodes]

        sorted_d = sorted(dists)
        q025   = sorted_d[int(0.025 * n)]
        q975   = sorted_d[int(0.975 * n)]
        drange = q975 - q025
        self.lambda1 = 1.0 if drange < 1e-6 else math.log(self.delta) / drange

        self.belief  = [0.0] * n
        uniform_prob = 1.0 / 5

        if self.use_obs_init:
            best_idx = int(np.argmin(dists))
            start    = max(0, best_idx - 2)
            end      = min(n, start + 5)
            for i in range(start, end):
                self.belief[i] = uniform_prob
            self.get_logger().info(
                f'[Init] obs-based, center={self.map_nodes[best_idx]["id"]}, '
                f'lambda1={self.lambda1:.3f}')
        else:
            for i in range(min(5, n)):
                self.belief[i] = uniform_prob
            self.get_logger().info(
                f'[Init] ID0-centered, lambda1={self.lambda1:.3f}')

        self.is_initialized = True

    def _obs_likelihood(self, query_feat: np.ndarray) -> list[float]:
        """各ノードの観測尤度を計算する。"""
        return [
            math.exp(max(-50.0, -self.lambda1 * self._cos_distance(query_feat, n['feature'])))
            for n in self.map_nodes
        ]

    def _apply_transition(self) -> list[float]:
        """遷移モデルを畳み込んで予測分布を求める（循環境界）。"""
        n         = len(self.belief)
        predicted = [0.0] * n
        for i in range(n):
            for j, t in enumerate(self.transition):
                src = (i - self.window_lower - j) % n
                predicted[i] += self.belief[src] * t
        return predicted

    def _update_belief(self, predicted: list[float], likelihoods: list[float]) -> None:
        """観測尤度で信念を更新・正規化する。ウィンドウ外の確率はゼロにする。"""
        n        = len(predicted)
        best_idx = int(np.argmax(predicted))

        for i in range(n):
            offset = i - best_idx
            if offset < self.window_lower or offset > self.window_upper:
                self.belief[i] = 0.0
            else:
                self.belief[i] = predicted[i] * likelihoods[i]

        total = sum(self.belief)
        if total > 1e-6:
            self.belief = [b / total for b in self.belief]
        else:
            self.get_logger().warn('[Belief] Normalization failed, keeping previous belief.')

    def _infer_node_id(self, cv_bgr: np.ndarray) -> int:
        """ベイズフィルタを1ステップ進め、最尤ノードIDを返す。"""
        query_feat = self._extract_feature(cv_bgr)

        if not self.is_initialized:
            self._initialize_belief(query_feat)
            return self.map_nodes[int(np.argmax(self.belief))]['id']

        likelihoods = self._obs_likelihood(query_feat)
        predicted   = self._apply_transition()
        self._update_belief(predicted, likelihoods)
        return self.map_nodes[int(np.argmax(self.belief))]['id']

    # ===========================================================
    # ノード情報アクセス
    # ===========================================================

    def _get_node(self, node_id: int) -> dict | None:
        """ノードIDから辞書を返す。"""
        idx = self._id_to_idx.get(node_id)
        return self.map_nodes[idx] if idx is not None else None

    def _get_goal_node_id(self, current_node_id: int) -> int | None:
        """goal_node_offset 先のノードIDを edge_target チェーンで求める。"""
        target_id = current_node_id
        for _ in range(self.goal_node_offset):
            node = self._get_node(target_id)
            if node is None or node['edge_target'] is None:
                return None  # マップ終端
            target_id = node['edge_target']
        return target_id

    def _load_goal_image_rgb(self, node_id: int) -> np.ndarray | None:
        """ノード ID の参照画像を RGB で返す（キャッシュあり）。"""
        if node_id in self._goal_img_cache:
            return self._goal_img_cache[node_id]

        node = self._get_node(node_id)
        if node is None or not node['image']:
            return None

        img_path = os.path.join(self.image_dir, node['image'])
        bgr = cv2.imread(img_path)
        if bgr is None:
            self.get_logger().error(f'Cannot load image: {img_path}')
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._goal_img_cache[node_id] = rgb
        return rgb

    # ===========================================================
    # 配信ヘルパー
    # ===========================================================

    def _publish_goal_image(self, node_id: int) -> bool:
        """指定ノードの参照画像を /goal_image に配信する。成功したら True を返す。"""
        rgb = self._load_goal_image_rgb(node_id)
        if rgb is None:
            return False

        try:
            msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.goal_img_pub.publish(msg)
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to publish goal_image: {e}')
            return False

    def _is_near_stopped(self, node_id: int) -> bool:
        """過去に停止したノードの ±10 以内か判定する。"""
        return any(abs(node_id - sid) <= 10 for sid in self.stopped_node_ids)

    # ===========================================================
    # ROS 2 コールバック
    # ===========================================================

    def _autonomous_callback(self, msg: Bool) -> None:
        """走行フラグの更新。True に変わった時は初期化をリセットして再スタートする。"""
        prev = self.is_autonomous
        self.is_autonomous = msg.data
        if not prev and msg.data:
            self.is_initialized = False
            self.stopped_node_ids.clear()
            self._prev_node_id   = -1
            self._goal_img_cache.clear()
            self.get_logger().info('[TopoLocalizer] Autonomous ON → re-initialize on next frame.')

    def _image_callback(self, msg: Image) -> None:
        """
        カメラ画像コールバック：
        1. ベイズフィルタで現在ノードを推定
        2. action → lan_prompt を /lan_prompt に配信
        3. offset 先のノード画像を /goal_image に配信（ノードが変わった時のみリロード）
        """
        if not self.is_autonomous or self.placenet is None:
            return

        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # --- 自己位置推定 ---
        node_id = self._infer_node_id(cv_bgr)
        node    = self._get_node(node_id)
        if node is None:
            return

        action = node['action']

        # 既に停止済み地点の近くなら stop を回避
        if action == 'stop' and self._is_near_stopped(node_id):
            self.get_logger().info(
                f'[TopoLocalizer] node={node_id} is near a previous stop. Treating as straight.')
            action = 'straight'

        # --- stop 処理 ---
        if action == 'stop':
            self.get_logger().warn(
                f'[TopoLocalizer] STOP at node {node_id}. Halting navigation.')
            self.stopped_node_ids.add(node_id)
            self.is_autonomous = False
            self.autonomous_pub.publish(Bool(data=False))
            return

        # --- 1. 言語プロンプト配信 ---
        prompt = ACTION_TO_PROMPT.get(action, 'go straight')
        self.prompt_pub.publish(String(data=prompt))

        # --- 2. ビジュアルゴール画像配信（ノードが変わった時のみ再ロード） ---
        goal_node_id = self._get_goal_node_id(node_id)
        if goal_node_id is not None:
            if node_id != self._prev_node_id:
                # ノード遷移発生 → ゴール画像を更新
                success = self._publish_goal_image(goal_node_id)
                if success:
                    self.get_logger().info(
                        f'[TopoLocalizer] node={node_id}, goal_node={goal_node_id}, '
                        f'action="{action}", prompt="{prompt}"')
            else:
                # 同一ノード滞在中 → キャッシュ画像を継続配信
                self._publish_goal_image(goal_node_id)
        else:
            self.get_logger().warn(
                f'[TopoLocalizer] node={node_id} has no further edge (map end?).')

        self._prev_node_id = node_id

        # --- デバッグ用トピック ---
        self.node_id_pub.publish(Int32(data=node_id))
        self.action_pub.publish(String(data=action))


# ============================================================
# エントリポイント
# ============================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TopoLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
