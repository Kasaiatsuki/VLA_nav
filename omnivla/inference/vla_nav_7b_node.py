#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import torch
from transformers import GenerationMixin
import numpy as np
import os
import sys
from PIL import Image as PILImage
from typing import Optional

# VLA_nav root
script_dir = os.path.dirname(os.path.abspath(__file__))
vla_nav_root = os.path.dirname(os.path.dirname(script_dir))
if vla_nav_root not in sys.path:
    sys.path.insert(0, vla_nav_root)

from prismatic.models.load import load_vla
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
# from omnivla.zed_capture import ZedCameraWrapper

class VlaNav7BNode(Node):
    def __init__(self) -> None:
        super().__init__('vla_nav_7b_node')

        # --- パラメータの設定 ---
        self.declare_parameter('base_model_id', 'openvla/openvla-7b')
        self.declare_parameter('checkpoint_path', '')
        self.declare_parameter('lan_prompt', 'go straight')
        self.declare_parameter('linear_vel_max', 0.3)
        self.declare_parameter('angular_vel_max', 0.5)
        self.declare_parameter('interval_ms', 1000)
        self.declare_parameter('waypoint_select', 0)
        self.declare_parameter('metric_waypoint_spacing', 1.0)
        self.declare_parameter('v_gain', 1.0)
        self.declare_parameter('w_gain', 2.0)
        self.declare_parameter('camera_topic', '/image_raw')

        # 1. ROS 2 パラメータの取得
        self.base_model_id = self.get_parameter('base_model_id').value.strip()
        self.checkpoint_path = self.get_parameter('checkpoint_path').value.strip()
        self.lan_prompt = self.get_parameter('lan_prompt').value.strip()
        self.max_v = float(self.get_parameter('linear_vel_max').value)
        self.max_w = float(self.get_parameter('angular_vel_max').value)
        interval_ms = int(self.get_parameter('interval_ms').value)
        self.waypoint_select = int(self.get_parameter('waypoint_select').value)
        self.metric_waypoint_spacing = float(self.get_parameter('metric_waypoint_spacing').value)
        self.v_gain = float(self.get_parameter('v_gain').value)
        self.w_gain = float(self.get_parameter('w_gain').value)
        camera_topic = self.get_parameter('camera_topic').value

        # 2. 【フォールバック】直接 sys.argv から取得 (ROS 2 Foxy の python3 直接実行対策)
        if not self.checkpoint_path:
            for arg in sys.argv:
                if 'checkpoint_path:=' in arg:
                    self.checkpoint_path = arg.split(':=')[1].split(' ')[0].strip()
        
        self.get_logger().info(f"Final Checkpoint Path: '{self.checkpoint_path}'")
        
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            self.get_logger().error("FATAL: No valid checkpoint_path provided. Model WILL NOT behave as a VLA.")
            # return # 続行しても無駄だが一旦警告のみ
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bridge = CvBridge()
        
        # --- 単眼カメラの設定 ---
        self.latest_cv_image = None
        self.create_subscription(Image, camera_topic, self._camera_image_callback, qos_profile_sensor_data)
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')

        # --- OmniVLA (7B) モデルのロード ---
        self.get_logger().info(f'Loading VLA base model: {self.base_model_id} (4-bit)...')
        self.model, self.processor = load_vla(self.base_model_id, load_for_training=False, use_4bit=True)
        
        # --- LoRA アダプターのロード ---
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.get_logger().info(f'Loading LoRA adapters from: {self.checkpoint_path}...')
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
                self.get_logger().info(f"LoRA adapters loaded. Active: {self.model.active_adapter}")
            except Exception as e:
                self.get_logger().error(f'Failed to load PEFT adapter: {e}')
        
        # 推論モードの設定
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.get_logger().info(f"ActionTokenizer: begin_idx={self.action_tokenizer.action_token_begin_idx}")
        
        # GPUへ転送 (Backboneなどを float16 に)
        self.model.vision_backbone.to(device=self.device, dtype=torch.float16)
        self.model.projector.to(device=self.device, dtype=torch.float16)
        self.model.eval()

        # --- ROS 相互作用 ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vla_cmd_vel_pub = self.create_publisher(Twist, '/vla_cmd_vel', 10) # 診断用
        
        self.create_subscription(Bool, 'autonomous', self._autonomous_callback, 10)
        self.create_subscription(String, 'lan_prompt', self._lan_prompt_callback, 10)

        # デバッグ用パブリッシャ
        self.debug_img_pub = self.create_publisher(Image, '/vla_nav_7b/debug_image', 10)

        self.is_autonomous = True
        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)
        self.get_logger().info('VLA Nav 7B Node READY.')

    def _autonomous_callback(self, msg: Bool) -> None:
        self.is_autonomous = msg.data
        self.get_logger().info(f'Autonomous mode: {"ENABLED" if self.is_autonomous else "DISABLED"}')

    def _lan_prompt_callback(self, msg: String) -> None:
        self.lan_prompt = msg.data
        self.get_logger().info(f"Updated prompt: '{self.lan_prompt}'")

    def _camera_image_callback(self, msg: Image) -> None:
        if self.latest_cv_image is None:
            self.get_logger().info('Received FIRST image from camera in 7B node.')
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to process camera image: {e}")

    @torch.inference_mode()
    def timer_callback(self) -> None:
        if not self.is_autonomous or self.latest_cv_image is None:
            return

        # 1. 画像取得
        frame = self.latest_cv_image.copy()
        if frame is None:
            return

        # 2. 前処理
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # プロンプトの構築
        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", self.lan_prompt)
        prompt = prompt_builder.get_prompt()

        try:
            # 入力エンコード
            input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat((input_ids, torch.unsqueeze(torch.tensor([29871]).to(self.device).long(), dim=0)), dim=1)
            
            # 画像前処理
            pixel_values = self.processor.image_processor.apply_transform(pil_image)
            if isinstance(pixel_values, dict):
                pixel_values = {k: v[None, ...].to(self.device, dtype=torch.float16) for k, v in pixel_values.items()}
            else:
                pixel_values = pixel_values[None, ...].to(self.device, dtype=torch.float16)

            # --- 推論 ---
            action_dim = 4
            gen_length = 8
            generated_ids = GenerationMixin.generate(
                self.model,
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=gen_length,
                do_sample=False
            )
            
            # 結果の抽出
            new_tokens = generated_ids[0, input_ids.shape[1]:]
            token_list = new_tokens.cpu().numpy()
            generated_text = self.processor.tokenizer.decode(new_tokens)
            self.get_logger().info(f"Generated tokens: {token_list} (Text: [{generated_text}])")

            # アクション妥当性チェック
            valid_action = True
            for t in token_list[:action_dim]:
                if not (31744 <= t <= 31999):
                    valid_action = False
                    break
            
            if not valid_action:
                self.get_logger().warn("Warning: Model Hallucination. Sending STOP.")
                v, w = 0.0, 0.0
            else:
                action_token_ids = token_list[:action_dim]
                normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(action_token_ids)
                waypoints = normalized_actions.reshape(-1, 4)
                
                # パラメータに基づく制御計算
                target_wp = waypoints[self.waypoint_select]
                dx, dy = target_wp[0], target_wp[1]
                v = np.clip(dx * self.v_gain, 0.0, self.max_v)
                w = np.clip(dy * self.w_gain, -self.max_w, self.max_w)
                self.get_logger().info(f"Inferred (wp={self.waypoint_select}): dx={dx:.3f}, dy={dy:.3f} -> v={v:.2f}, w={w:.2f}")

            # 配信
            twist_msg = Twist()
            twist_msg.linear.x = float(v)
            twist_msg.angular.z = float(w)
            self.cmd_vel_pub.publish(twist_msg)
            self.vla_cmd_vel_pub.publish(twist_msg)

            # --- デバッグ表示用 (描画と配信) ---
            if valid_action:
                self.draw_waypoints_on_image(frame, waypoints, v, w)
            
        except Exception as e:
            self.get_logger().error(f"Inference Loop Error: {e}")

    def draw_waypoints_on_image(self, bgr_img, waypoints, v, w):
        """予測されたウェイポイントを画像上に簡易描画し、配信する"""
        canvas = bgr_img.copy()
        h, w_img, _ = canvas.shape
        
        # 描画スケールも調整可能にする
        scale = self.metric_waypoint_spacing * 200.0
        
        # 簡易描画: 画像中央らへんから推論値に基づくベクトルを描画
        start_pt = (int(w_img/2), h - 10)
        # 選択されたウェイポイントを描画
        target_wp = waypoints[self.waypoint_select]
        end_pt = (int(w_img/2 + target_wp[1] * scale), int(h - 10 - target_wp[0] * scale))
        
        cv2.arrowedLine(canvas, start_pt, end_pt, (0, 255, 0), 3)
        cv2.putText(canvas, f"v={v:.2f}, w={w:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Prompt: {self.lan_prompt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        debug_msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        self.debug_img_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VlaNav7BNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
