#!/usr/bin/env python3
"""
capture_goal_images_node.py
役割: /flag や /capture_image トピック（コントローラのボタン入力など）を受信した瞬間に、
ZEDカメラから画像を1枚撮影し、goal_images フォルダに連番（001.jpg, 002.jpg...）で保存するノードです。
"""

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import cv2
import glob

# VLA_nav内の独自モジュールを読み込めるようパスを通す
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
vla_nav_root = os.path.dirname(os.path.dirname(script_dir))
if vla_nav_root not in sys.path:
    sys.path.insert(0, vla_nav_root)

from omnivla.zed_capture import ZedCameraWrapper

class CaptureGoalImagesNode(Node):
    def __init__(self):
        super().__init__('capture_goal_images_node')

        # 保存先ディレクトリ
        self.declare_parameter('goal_images_dir', os.path.join(script_dir, 'goal_images'))
        self.goal_images_dir = self.get_parameter('goal_images_dir').value

        if not os.path.exists(self.goal_images_dir):
            os.makedirs(self.goal_images_dir)

        # ZEDカメラの準備
        self.zed_camera = ZedCameraWrapper(fps=15)
        try:
            self.zed_camera.open()
            self.get_logger().info('ZED camera initialized for manual capture.')
        except RuntimeError as e:
            self.get_logger().error(f"Failed to open ZED camera: {e}")
            raise e

        # 既存ファイルの数を数えて次の連番インデックスを決定
        existing_files = glob.glob(os.path.join(self.goal_images_dir, '*.jpg'))
        self.current_idx = len(existing_files) + 1

        # create_data.py と同じく '/flag' トピック（または /capture_image）でシャッターを切る
        self.create_subscription(Empty, '/flag', self._capture_callback, 10)
        self.create_subscription(Empty, '/capture_image', self._capture_callback, 10)

        self.get_logger().info(f'📸 Ready! コントローラのボタン(/flag)を押すと {self.current_idx:03d}.jpg が撮影されます。')

    def _capture_callback(self, _msg: Empty):
        """トピックを受け取った瞬間に1枚撮影して保存する"""
        image = self.zed_camera.grab_image()
        if image is None:
            self.get_logger().warning('Failed to grab image from ZED camera')
            return

        filename = f"{self.current_idx:03d}.jpg"
        filepath = os.path.join(self.goal_images_dir, filename)

        # 画像を保存 (ZEDはBGRで返ってくるのでそのままimwriteでOK)
        cv2.imwrite(filepath, image)
        self.get_logger().info(f'✅ Captured and saved: {filename}')
        
        self.current_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = CaptureGoalImagesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'zed_camera') and node.zed_camera:
            node.zed_camera.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
