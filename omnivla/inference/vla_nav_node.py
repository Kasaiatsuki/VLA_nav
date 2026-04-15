#!/usr/bin/env python3
"""
vla_nav_node.py
役割: OmniVLA-edgeモデルを利用して、カメラ画像から角速度(angular.z)と線形速度(linear.x)を推論し、
ROS 2のTwistメッセージとして配信('/cmd_vel')するノードです。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
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

### ROS 2 ノード: VLA Navigation Node (OmniVLA-edge)
### 役割: ROSトピック経由で単眼カメラからの画像と自然言語の指示（プロンプト）を受け取り、
###       OmniVLA-edgeモデルを通じてロボットの速度指令(cmd_vel)を生成します。
class VlaNavNode(Node):
    def __init__(self) -> None:
        super().__init__('vla_nav_node')

        # --- パラメータの設定 ---
        self.declare_parameter('model_path', os.path.join(script_dir, 'omnivla-edge/omnivla-edge.pth'))
        self.declare_parameter('lan_prompt', 'go forward')
        self.declare_parameter('interval_ms', 100) # 周期: 10Hz
        self.declare_parameter('is_autonomous', False)
        self.declare_parameter('linear_vel_max', 0.3)
        self.declare_parameter('angular_vel_max', 1.0)
        self.declare_parameter('waypoint_select', 7) # 0-7
        self.declare_parameter('metric_waypoint_spacing', 0.3)

        self.model_path = self.get_parameter('model_path').value
        self.lan_prompt = self.get_parameter('lan_prompt').value
        interval_ms = self.get_parameter('interval_ms').value
        self.is_autonomous = self.get_parameter('is_autonomous').value
        self.max_v = self.get_parameter('linear_vel_max').value
        self.max_w = self.get_parameter('angular_vel_max').value
        self.waypoint_select = self.get_parameter('waypoint_select').value
        self.metric_waypoint_spacing = self.get_parameter('metric_waypoint_spacing').value

        # --- 描画用パラメータ（カメラの取り付け位置） ---
        self.declare_parameter('cam_offset_x', -0.2)  # base_linkから前方への距離(m)
        self.declare_parameter('cam_offset_z', 0.2)  # base_linkから高さ(m)
        self.declare_parameter('cam_pitch', 0.3)     # 下向きの傾斜角(rad)
        
        self.cam_x = self.get_parameter('cam_offset_x').value
        self.cam_z = self.get_parameter('cam_offset_z').value
        self.cam_pitch = self.get_parameter('cam_pitch').value

        self.prev_v = 0.0
        self.prev_w = 0.0

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- 単眼カメラ用サブスクライバの初期化 ---
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')
        camera_topic = self.get_parameter('camera_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        
        self.latest_cv_image = None
        self.cam_p = None # カメラパラメータ用辞書 (fx, fy, cx, cy)
        
        self.create_subscription(Image, camera_topic, self._camera_image_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, 10)
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')
        self.get_logger().info(f'Subscribed to camera info topic: {camera_info_topic}')

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
        self.context_size = self.get_parameter('context_size').value
        self.context_queue = []
        self.mask_360_pil_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_360_pil_224 = np.ones((224, 224, 3), dtype=np.float32)
        self.prev_v = 0.0
        self.prev_w = 0.0
        self.inference_time_ms = 0.0

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
        self.waypoint_marker_pub = self.create_publisher(Marker, 'waypoints_marker', 10)
        self.debug_img_pub = self.create_publisher(Image, '/vla_nav/debug_image', 10)
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

    def _camera_image_callback(self, msg: Image) -> None:
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self.cam_p is None:
            self.cam_p = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }

    def timer_callback(self) -> None:
        if not self.is_autonomous or self.model is None or self.latest_cv_image is None:
            return

        # 1. 画像の取得
        cv_image = self.latest_cv_image.copy()
        
        # 2. 画像の前処理
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        
        image_96 = pil_image.resize(self.imgsize)
        image_224 = pil_image.resize(self.imgsize_clip)
        
        # 3. コンテキストキューの更新
        if len(self.context_queue) == 0:
            self.context_queue = [image_96] * (self.context_size + 1)
        else:
            self.context_queue.append(image_96)
            if len(self.context_queue) > (self.context_size + 1):
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
        t_start = time.time()
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
        t_end = time.time()
        self.inference_time_ms = (t_end - t_start) * 1000.0
        
        # 5. 制御 (Waypoints -> Velocity)
        waypoints = predicted_actions[0].float().cpu().numpy()
        self.publish_waypoints_marker(waypoints)
        v, w = self.compute_velocity(waypoints)
        
        # 6. 画像へのオーバーレイ描画
        self.draw_waypoints_on_image(cv_image, waypoints)

        # 7. Twist配信
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
        
        # 横方向の偏差に基づいた角速度計算（ゲインを 2.0 -> 4.0 へ強化）
        if abs(dx) < 1e-8:
            w_raw = np.arctan2(hy, hx) * 4.0
        else:
            w_raw = np.arctan2(dy, dx) * 4.0
            
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

    def publish_waypoints_marker(self, waypoints):
        """予測されたウェイポイントを Marker (LINE_STRIP) として配信"""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "predicted_waypoints"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 線の太さと色
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        # 始点としてロボットの位置 (0,0,0) を追加
        p_start = Point()
        p_start.x = 0.0
        p_start.y = 0.0
        p_start.z = 0.0
        marker.points.append(p_start)
        
        for wp in waypoints:
            dx, dy, hx, hy = wp
            p = Point()
            p.x = float(dx * self.metric_waypoint_spacing)
            p.y = float(dy * self.metric_waypoint_spacing)
            p.z = 0.0
            marker.points.append(p)
            
        self.waypoint_marker_pub.publish(marker)

    def draw_waypoints_on_image(self, bgr_img, waypoints):
        """ウェイポイントを画像上に投影して描画し、配信する"""
        if self.cam_p is None:
            return

        # カメラパラメータの取得
        cam_p = self.cam_p
        
        # 描画用コピー
        canvas = bgr_img.copy()
        h, w, _ = canvas.shape # 通常 360, 640
        
        pts_2d = []
        for i, wp in enumerate(waypoints):
            dx, dy, _, _ = wp
            x_base = dx * self.metric_waypoint_spacing
            y_base = dy * self.metric_waypoint_spacing
            
            # --- 座標変換 (base_link -> camera_frame) ---
            # 1. カメラ相対位置 (x:前, y:左, z:上)
            rel_x = x_base - self.cam_x
            rel_y = y_base - 0.0
            rel_z = 0.0 - self.cam_z # 地面(z=0)
            
            # 2. カメラ座標系への回転・軸変換 (X:右, Y:下, Z:前)
            # 未回転時: x_c = -rel_y, y_c = -rel_z, z_c = rel_x
            # Pitch回転適用後 (Downward positive theta):
            # y' = y*cos(theta) - z*sin(theta)
            # z' = y*sin(theta) + z*cos(theta)
            y_c0 = -rel_z
            z_c0 = rel_x
            
            p_cam_x = -rel_y
            p_cam_y = y_c0 * math.cos(self.cam_pitch) - z_c0 * math.sin(self.cam_pitch)
            p_cam_z = y_c0 * math.sin(self.cam_pitch) + z_c0 * math.cos(self.cam_pitch)
            
            # 3. 投影
            if p_cam_z > 0.1:
                u = cam_p['fx'] * (p_cam_x / p_cam_z) + cam_p['cx']
                v = cam_p['fy'] * (p_cam_y / p_cam_z) + cam_p['cy']
                
                # 画像範囲内かどうかにかかわらず追加（あとで線を描くため）
                pts_2d.append((int(u), int(v)))

        # デバッグログ (2秒に一度)
        if len(pts_2d) > 0 and (int(time.time() * 10) % 20 == 0):
            wp_str = ", ".join([f"({u},{v})" for u, v in pts_2d])
            self.get_logger().info(f"DEBUG: Inference: {self.inference_time_ms:.1f}ms, Trajectory 2D: [{wp_str}]")
            self.get_logger().info(f"DEBUG: Selected target (idx {self.waypoint_select}): {pts_2d[min(self.waypoint_select, len(pts_2d)-1)]}")

        # 線と点の描画
        for i in range(len(pts_2d)):
            u, v = pts_2d[i]
            # 描画範囲内なら点を描く
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(canvas, (u, v), 5, (0, 255, 0), -1)
            
            # 線を描く (前の点と結ぶ)
            if i > 0:
                p1 = pts_2d[i-1]
                p2 = pts_2d[i]
                cv2.line(canvas, p1, p2, (0, 255, 255), 2)

        # テキスト情報
        cv2.putText(canvas, f"Prompt: {self.lan_prompt}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Mode: {'AUTO' if self.is_autonomous else 'MANUAL'}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.is_autonomous else (0, 0, 255), 2)

        # パブリッシュ
        debug_msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        self.debug_img_pub.publish(debug_msg)

# エントリポイント
def main(args=None) -> None:
    rclpy.init(args=args)
    node = VlaNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
