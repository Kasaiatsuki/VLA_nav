#!/usr/bin/env python3
"""
vla_nav_node.py
役割: OmniVLA-edgeモデルを利用して、カメラ画像から角速度(angular.z)と線形速度(linear.x)を推論し、
ROS 2のTwistメッセージとして配信('/cmd_vel')するノードです。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
import sys
import math
import time
from PIL import Image as PILImage
from typing import Optional, List

# 自身のディレクトリ(inference)をパスに追加
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from utils_policy import load_model, transform_images_PIL_mask, transform_images_map
    import clip
except ImportError as e:
    print(f"Failed to import from VLA_nav: {e}")

# e2e_nav_box1がインストールされている前提でimport
try:
    from e2e_nav_box1.zed_capture import ZedCameraWrapper
except ImportError as e:
    print(f"Failed to import ZedCameraWrapper from e2e_nav_box1. Ensure it is built and sourced. Error: {e}")

### ROS 2 ノード: VLA Navigation Node (OmniVLA-edge)
### 役割: ZEDカメラからの画像と自然言語の指示（プロンプト）を受け取り、
###       OmniVLA-edgeモデルを通じてロボットの速度指令(cmd_vel)を生成します。
class VlaNavNode(Node):
    def __init__(self) -> None:
        super().__init__('vla_nav_node')

        # --- パラメータの設定 ---
        # モデルの重みパス、デフォルトの言語プロンプト、実行周期などを宣言します。
        self.declare_parameter('model_path', os.path.join(script_dir, 'omnivla-edge/omnivla-edge.pth'))
        self.declare_parameter('lan_prompt', 'blue trash bin')
        self.declare_parameter('interval_ms', 333) # 周期: 約3Hz (0.33秒ごと)
        self.declare_parameter('is_autonomous', False) # 自動走行モードのフラグ

        self.model_path = self.get_parameter('model_path').value
        self.lan_prompt = self.get_parameter('lan_prompt').value
        interval_ms = self.get_parameter('interval_ms').value
        self.is_autonomous = self.get_parameter('is_autonomous').value

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- ZED カメラの初期化 ---
        # e2e_nav_box1のZedCameraWrapperを利用してカメラをオープンします。
        try:
            self.zed_camera = ZedCameraWrapper(fps=30)
            self.zed_camera.open()
            self.get_logger().info('ZED camera initialized successfully')
        except Exception as e:
            self.get_logger().error(f"Failed to initialize ZED camera: {e}")
            self.zed_camera = None

        # --- OmniVLA-edge モデル用設定 ---
        # モデルの構造を定義するハイパーパラメータです。学習時と同じ設定である必要があります。
        self.model_params = {
            "model_type": "omnivla-edge",
            "len_traj_pred": 8,          # 8ステップ先までの経路を予測
            "learn_angle": True,
            "context_size": 5,           # 過去5枚+最新1枚のコンテキストを利用
            "obs_encoder": "efficientnet-b0", # 視覚バックボーン
            "encoding_size": 256,
            "obs_encoding_size": 1024,
            "goal_encoding_size": 1024,
            "late_fusion": False,
            "mha_num_attention_heads": 4,
            "mha_num_attention_layers": 4,
            "mha_ff_dim_factor": 4,
            "clip_type": "ViT-B/32"      # 言語モデル(CLIP)のタイプ
        }

        # 画像のリサイズサイズ (モデル入力用: 96x96, CLIP用: 224x224)
        self.imgsize = (96, 96)
        self.imgsize_clip = (224, 224)
        
        # 過去の画像を保持するキュー (6枚分)
        self.context_queue = []
        
        # 360度カメラ用ではないため、全面有効なマスクを使用
        self.mask_360_pil_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_360_pil_224 = np.ones((224, 224, 3), dtype=np.float32)

        # --- モデルのロード ---
        if os.path.exists(self.model_path):
            self.get_logger().info(f'Loading model from {self.model_path}...')
            self.model, self.text_encoder, self.clip_preprocess = load_model(
                self.model_path,
                self.model_params,
                self.device
            )
            self.text_encoder.eval()
            self.model.eval()
            self.get_logger().info('Model loaded successfully')
        else:
            self.get_logger().warn(f'Model weights NOT found at {self.model_path}.')
            self.model = None

        # --- Publisher / Subscriber ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Bool, 'autonomous', self._autonomous_callback, 10)
        self.create_subscription(String, 'lan_prompt', self._lan_prompt_callback, 10)

        # 指定周期で推論を実行するタイマー
        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)

    # autonomousトピックの更新を受け取る
    def _autonomous_callback(self, msg: Bool) -> None:
        self.is_autonomous = msg.data
        status = "ENABLED" if self.is_autonomous else "DISABLED"
        self.get_logger().info(f'Autonomous mode: {status}')

    # 外部から新しい言語プロンプト（「青いゴミ箱へ行け」など）を受け取る
    def _lan_prompt_callback(self, msg: String) -> None:
        self.lan_prompt = msg.data
        self.get_logger().info(f'Language prompt updated to: {self.lan_prompt}')

    # メインループ: 定期的に実行される推論処理
    def timer_callback(self) -> None:
        # 自動走行オフ、またはモデル/カメラが未準備なら何もしない
        if not self.is_autonomous or self.model is None or self.zed_camera is None:
            return

        # 1. 画像の取得
        cv_image = self.zed_camera.grab_image()
        if cv_image is None:
            return
        
        # 2. 画像の前処理 (BGR -> RGB, PIL変換, リサイズ)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        
        image_96 = pil_image.resize(self.imgsize)
        image_224 = pil_image.resize(self.imgsize_clip)
        
        # 3. コンテキストキューの更新 (常に最新6枚を保つ)
        self.context_queue.append(image_96)
        if len(self.context_queue) > 6:
            self.context_queue.pop(0)
        
        # キューが足りない場合は最新画像で埋める（起動直後用）
        if len(self.context_queue) < 6:
            while len(self.context_queue) < 6:
                self.context_queue.insert(0, image_96)

        # --- テンソル形式への変換 ---
        # 複数の画像を1つのテンソルに統合
        obs_images = transform_images_PIL_mask(self.context_queue, self.mask_360_pil_96)
        obs_images = torch.split(obs_images.to(self.device), 3, dim=1)
        obs_image_cur = obs_images[-1].to(self.device)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)
        
        # CLIP用の高解像度画像
        cur_large_img = transform_images_PIL_mask(image_224, self.mask_360_pil_224).to(self.device)
        
        # マップ入力（今回はダミーの黒画像を使用）
        satellite_cur = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        satellite_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        current_map_image = transform_images_map(satellite_cur)
        goal_map_image = transform_images_map(satellite_goal)
        map_images = torch.cat((current_map_image.to(self.device), goal_map_image.to(self.device), obs_image_cur), axis=1)
        
        # 言語命令をトークン化
        obj_inst_lan = clip.tokenize(self.lan_prompt, truncate=True).to(self.device)
        
        # ゴールポーズ入力（今回は使用しないためゼロ埋め）
        goal_pose_torch = torch.zeros((1, 4)).to(self.device)
        goal_image_PIL = PILImage.new("RGB", self.imgsize, color=(0, 0, 0))
        goal_image = transform_images_PIL_mask(goal_image_PIL, self.mask_360_pil_96).to(self.device)
        
        # 推論モードのID (言語指示ベースのナビゲーションは [7] 等)
        modality_id_select = torch.tensor([7]).to(self.device)
        
        # 4. モデルの実行 (推論)
        with torch.no_grad():
            feat_text_lan = self.text_encoder.encode_text(obj_inst_lan) # プロンプトを特徴量に変換
            predicted_actions, _, _ = self.model(
                obs_images, 
                goal_pose_torch, 
                map_images, 
                goal_image, 
                modality_id_select, 
                feat_text_lan, 
                cur_large_img
            )
        
        # 5. 予測結果を元に速度指令を算出
        waypoints = predicted_actions.float().cpu().numpy() # 予測された経路座標 (dx, dy)
        
        # 経路の5番目（インデックス4）の地点をターゲットにする
        waypoint_select = 4
        metric_waypoint_spacing = 0.1 # モデルの座標系から現実のメートルへのスケール
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint # 到達目標の座標(dx, dy)と向き(hx, hy)

        EPS = 1e-8
        DT = 1 / 3 # 推論周期(s)
        
        # --- 簡易的なコントローラ (P制御ライクな変換) ---
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_v = 0.0
            angular_v = 1.0 * math.atan2(hy, hx) / DT
        elif np.abs(dx) < EPS:
            linear_v = 0.0
            angular_v = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            # 目標地点に到達するための前進速度と回転速度を計算
            linear_v = dx / DT
            angular_v = math.atan(dy / dx) / DT

        # ロボットの限界に合わせてクランプ (前進最大 0.5m/s, 回転最大 1.0rad/s)
        linear_v = np.clip(linear_v, 0, 0.5)
        angular_v = np.clip(angular_v, -1.0, 1.0)
        
        # 6. Twist情報の配信
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_v)
        twist_msg.angular.z = float(angular_v)
        self.cmd_vel_pub.publish(twist_msg)

# エントリポイント
def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlaNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 終了時にカメラを安全に閉じる
        if node.zed_camera:
            node.zed_camera.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
