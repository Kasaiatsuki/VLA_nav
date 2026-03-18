#!/usr/bin/env python3
"""
topological_manager_node.py
役割: ディレクトリ内の画像群（ゴール画像）を順番に読み込み、'/goal_image' トピックへ配信するマネージャーノード。
'/next_goal' トピックを受信すると、次の画像へと目標を切り替えます。
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
import cv2
from cv_bridge import CvBridge

class TopologicalManagerNode(Node):
    def __init__(self):
        super().__init__('topological_manager_node')

        # パラメータの宣言: ゴール画像が保存されているフォルダパス
        self.declare_parameter('goal_images_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'goal_images'))
        self.declare_parameter('publish_rate_hz', 1.0)
        
        self.goal_images_dir = self.get_parameter('goal_images_dir').value
        publish_rate = self.get_parameter('publish_rate_hz').value

        self.bridge = CvBridge()
        
        # パブリッシャーとサブスクライバーの設定
        self.goal_pub = self.create_publisher(Image, '/goal_image', 10)
        self.create_subscription(Empty, '/next_goal', self._next_goal_callback, 10)

        self.image_files = []
        self.current_idx = 0
        self.current_cv_image = None

        self._load_image_list()

        # 一定周期で現在のゴール画像をパブリッシュし続ける（途中から接続したノードへ届けるため）
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_callback)

    def _load_image_list(self):
        """指定されたディレクトリから画像リストを読み込み、ソートする"""
        if not os.path.exists(self.goal_images_dir):
            os.makedirs(self.goal_images_dir)
            
        valid_exts = ['.jpg', '.jpeg', '.png']
        files = [f for f in os.listdir(self.goal_images_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not files:
            self.get_logger().warn(f"画像が見つかりません。 {self.goal_images_dir} フォルダに写真を追加してください。")
            return

        # ファイル名でアルファベット順(数値順)にソート (ex: 01.jpg, 02.jpg...)
        self.image_files = sorted(files)
        self.get_logger().info(f"Loaded {len(self.image_files)} goal images.")
        
        self.current_idx = 0
        self._set_current_image()

    def _set_current_image(self):
        """現在のインデックスの画像をディスクから読み込む"""
        if self.current_idx >= len(self.image_files):
            self.get_logger().info("!!! すべての目標画像に到達しました。ナビゲーション完了 !!!")
            self.current_cv_image = None
            return

        img_name = self.image_files[self.current_idx]
        img_path = os.path.join(self.goal_images_dir, img_name)
        
        self.current_cv_image = cv2.imread(img_path)
        if self.current_cv_image is None:
            self.get_logger().error(f"Failed to read image: {img_path}")
        else:
            self.get_logger().info(f"Current Target: [{self.current_idx + 1}/{len(self.image_files)}] -> {img_name}")

    def _next_goal_callback(self, msg: Empty):
        """外部から /next_goal が呼ばれたら、次の目標画像に切り替える"""
        if self.current_idx < len(self.image_files):
            self.current_idx += 1
            self._set_current_image()
            self._timer_callback() # 切り替わったら即座に一回配信する

    def _timer_callback(self):
        """現在設定されているゴール画像を継続的に配信する"""
        if self.current_cv_image is not None:
            try:
                # vla_nav_nodeはRGB形式を期待しているので、BGR->RGBに変換して送信
                rgb_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
                msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
                self.goal_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing goal image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TopologicalManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
