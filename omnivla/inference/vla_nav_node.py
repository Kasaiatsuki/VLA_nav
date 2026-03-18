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

# VLA_navの親ディレクトリ(src/VLA_nav)をパスに追加して相対インポートを可能にする
vla_nav_root = os.path.dirname(os.path.dirname(script_dir))
if vla_nav_root not in sys.path:
    sys.path.insert(0, vla_nav_root)

from utils_policy import load_model, transform_images_PIL_mask, transform_images_map
import clip

# ローカルのZedCameraWrapperを使用
from omnivla.zed_capture import ZedCameraWrapper

### ROS 2 ノード: VLA Navigation Node (OmniVLA-edge)
### 役割: ZEDカメラからの画像と自然言語の指示（プロンプト）を受け取り、
###       OmniVLA-edgeモデルを通じてロボットの速度指令(cmd_vel)を生成します。
class VlaNavNode(Node):
    def __init__(self) -> None:
        super().__init__('vla_nav_node')

        # --- パラメータの設定 ---
        self.declare_parameter('model_path', os.path.join(script_dir, 'omnivla-edge/omnivla-edge.pth'))
        self.declare_parameter('lan_prompt', 'go forward')
        self.declare_parameter('interval_ms', 333) # 周期: 約3Hz
        self.declare_parameter('is_autonomous', False)
        self.declare_parameter('linear_vel_max', 0.3)
        self.declare_parameter('angular_vel_max', 0.5)
        self.declare_parameter('waypoint_select', 4) # 0-7
        self.declare_parameter('metric_waypoint_spacing', 0.1)

        self.model_path = self.get_parameter('model_path').value
        self.lan_prompt = self.get_parameter('lan_prompt').value
        interval_ms = self.get_parameter('interval_ms').value
        self.is_autonomous = self.get_parameter('is_autonomous').value
        self.max_v = self.get_parameter('linear_vel_max').value
        self.max_w = self.get_parameter('angular_vel_max').value
        self.waypoint_select = self.get_parameter('waypoint_select').value
        self.metric_waypoint_spacing = self.get_parameter('metric_waypoint_spacing').value

        self.prev_v = 0.0
        self.prev_w = 0.0

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- ZED カメラの初期化 ---
        try:
            self.zed_camera = ZedCameraWrapper(fps=15) # fpsを30から15に変更（omnivla_navに合わせる）
            self.zed_camera.open()
            self.get_logger().info('ZED camera initialized successfully')
        except Exception as e:
            self.get_logger().error(f"Failed to initialize ZED camera: {e}")
            self.zed_camera = None

        # --- OmniVLA-edge モデル用設定 ---
        self.declare_parameter('model_type', 'omnivla-edge')
        self.declare_parameter('len_traj_pred', 8)
        self.declare_parameter('learn_angle', True)
        self.declare_parameter('context_size', 5)
        self.declare_parameter('obs_encoder', 'efficientnet-b0')
        self.declare_parameter('encoding_size', 256)
        self.declare_parameter('obs_encoding_size', 1024)
        self.declare_parameter('goal_encoding_size', 1024)
        self.declare_parameter('late_fusion', False)
        self.declare_parameter('mha_num_attention_heads', 4)
        self.declare_parameter('mha_num_attention_layers', 4)
        self.declare_parameter('mha_ff_dim_factor', 4)
        self.declare_parameter('clip_type', 'ViT-B/32')

        self.model_params = {
            "model_type": self.get_parameter('model_type').value,
            "len_traj_pred": self.get_parameter('len_traj_pred').value,
            "learn_angle": self.get_parameter('learn_angle').value,
            "context_size": self.get_parameter('context_size').value,
            "obs_encoder": self.get_parameter('obs_encoder').value,
            "encoding_size": self.get_parameter('encoding_size').value,
            "obs_encoding_size": self.get_parameter('obs_encoding_size').value,
            "goal_encoding_size": self.get_parameter('goal_encoding_size').value,
            "late_fusion": self.get_parameter('late_fusion').value,
            "mha_num_attention_heads": self.get_parameter('mha_num_attention_heads').value,
            "mha_num_attention_layers": self.get_parameter('mha_num_attention_layers').value,
            "mha_ff_dim_factor": self.get_parameter('mha_ff_dim_factor').value,
            "clip_type": self.get_parameter('clip_type').value
        }

        self.imgsize = (96, 96)
        self.imgsize_clip = (224, 224)
        self.context_queue = []
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

        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)

    def _autonomous_callback(self, msg: Bool) -> None:
        self.is_autonomous = msg.data
        status = "ENABLED" if self.is_autonomous else "DISABLED"
        self.get_logger().info(f'Autonomous mode: {status}')

    def _lan_prompt_callback(self, msg: String) -> None:
        self.lan_prompt = msg.data
        self.get_logger().info(f'Language prompt updated to: {self.lan_prompt}')

    def timer_callback(self) -> None:
        if not self.is_autonomous or self.model is None or self.zed_camera is None:
            return

        # 1. 画像の取得
        cv_image = self.zed_camera.grab_image()
        if cv_image is None:
            return
        
        # 2. 画像の前処理
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        
        image_96 = pil_image.resize(self.imgsize)
        image_224 = pil_image.resize(self.imgsize_clip)
        
        # 3. コンテキストキューの更新
        if len(self.context_queue) == 0:
            self.context_queue = [image_96] * 6
        else:
            self.context_queue.append(image_96)
            if len(self.context_queue) > 6:
                self.context_queue.pop(0)
        
        # --- テンソル形式への変換 ---
        obs_images = transform_images_PIL_mask(self.context_queue, self.mask_360_pil_96).to(self.device)
        obs_images_split = torch.split(obs_images, 3, dim=1)
        obs_image_cur = obs_images_split[-1]
        obs_images_cat = torch.cat(obs_images_split, dim=1)
        
        cur_large_img = transform_images_PIL_mask(image_224, self.mask_360_pil_224).to(self.device)
        
        # ダミー入力（言語指示ナビゲーション用）
        satellite_dummy = torch.zeros((1, 3, 96, 96)).to(self.device)
        map_images = torch.cat((satellite_dummy, satellite_dummy, obs_image_cur), axis=1)
        
        obj_inst_lan = clip.tokenize(self.lan_prompt, truncate=True).to(self.device)
        
        goal_pose_dummy = torch.zeros((1, 4)).to(self.device)
        goal_image_dummy = transform_images_PIL_mask(image_96, self.mask_360_pil_96).to(self.device)
        
        modality_id = torch.tensor([7]).to(self.device)
        
        # 4. 推論実行
        with torch.no_grad():
            feat_text = self.text_encoder.encode_text(obj_inst_lan)
            predicted_actions, _, _ = self.model(
                obs_images_cat, 
                goal_pose_dummy, 
                map_images, 
                goal_image_dummy, 
                modality_id, 
                feat_text, 
                cur_large_img
            )
        
        # 5. 制御 (Waypoints -> Velocity)
        waypoints = predicted_actions[0].float().cpu().numpy()
        v, w = self.compute_velocity(waypoints)
        
        # 6. Twist配信
        twist_msg = Twist()
        twist_msg.linear.x = v
        twist_msg.angular.z = w
        self.cmd_vel_pub.publish(twist_msg)

    def compute_velocity(self, waypoints):
        """予測された軌道に基づいた比例制御 + スムージングによる速度計算"""
        target = waypoints[self.waypoint_select].copy()
        dx, dy, hx, hy = target
        
        # スケーリング
        dx *= self.metric_waypoint_spacing
        dy *= self.metric_waypoint_spacing
        
        # --- 制御ロジックの改善 ---
        # 1.0m 離れていたらマックス (max_v) になるような比例制御 (P-control)
        v_raw = dx * 1.0 
        
        # 横方向の偏差に基づいた角速度計算（3.0倍 -> 2.0倍 にマイルド化）
        if abs(dx) < 1e-8:
            w_raw = np.arctan2(hy, hx) * 2.0
        else:
            w_raw = np.arctan2(dy, dx) * 2.0
            
        # --- スムージングの適用 (Low-pass Filter) ---
        alpha_v = 0 # 50% 以前の値を保持
        alpha_w = 0 # 70% 以前の値を保持 (旋回はより慎重に)
        
        v_smoothed = alpha_v * self.prev_v + (1.0 - alpha_v) * v_raw
        w_smoothed = alpha_w * self.prev_w + (1.0 - alpha_w) * w_raw
        
        self.prev_v = v_smoothed
        self.prev_w = w_smoothed
        
        # 安全のためにクランプ
        v = np.clip(v_smoothed, 0.0, self.max_v)
        w = np.clip(w_smoothed, -self.max_w, self.max_w)
        
        return float(v), float(w)

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
