#!/usr/bin/env python3
"""
topological_manager_node.py
役割: 
1. ディレクトリ内の画像（ゴール）を読み込む。
2. 現在のカメラ画像とゴール画像を比較し、一致度を計算する。
3. 一致度が閾値を超えたら、自動的に次のゴール画像へ切り替える。
   ※ 起動直後の誤判定を防ぐため、最初の数秒間は判定をスキップします。
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32
import cv2
from cv_bridge import CvBridge

class TopologicalManagerNode(Node):
    def __init__(self):
        super().__init__('topological_manager_node')

        # --- パラメータの設定 ---
        self.declare_parameter('goal_images_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'goal_images'))
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('similarity_threshold', 15)
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('startup_delay_sec', 3.0)

        self.goal_images_dir = self.get_parameter('goal_images_dir').value
        publish_rate = self.get_parameter('publish_rate_hz').value
        self.threshold = self.get_parameter('similarity_threshold').value
        camera_topic = self.get_parameter('camera_topic').value
        self.startup_delay = self.get_parameter('startup_delay_sec').value

        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 起動時刻を記録
        self.start_time = self.get_clock().now()
        
        # --- 通信の設定 ---
        self.goal_pub = self.create_publisher(Image, '/goal_image', 10)
        self.sim_pub = self.create_publisher(Float32, '/goal_similarity', 10)
        
        self.create_subscription(Empty, '/next_goal', self._next_goal_callback, 10)
        self.create_subscription(Image, camera_topic, self._camera_callback, 1)

        self.image_files = []
        self.current_idx = 0
        self.current_cv_image = None
        self.goal_descriptors = None
        
        self._load_image_list()
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_callback)

    def _load_image_list(self):
        if not os.path.exists(self.goal_images_dir):
            os.makedirs(self.goal_images_dir)
            
        valid_exts = ['.jpg', '.jpeg', '.png']
        files = [f for f in os.listdir(self.goal_images_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not files:
            self.get_logger().warn(f"No goal images found in {self.goal_images_dir}")
            return

        self.image_files = sorted(files)
        self.get_logger().info(f"Loaded {len(self.image_files)} goal images.")
        self.current_idx = 0
        self._set_current_image()

    def _set_current_image(self):
        if self.current_idx >= len(self.image_files):
            self.get_logger().info("🏁 All goals reached! Navigation complete.")
            self.current_cv_image = None
            self.goal_descriptors = None
            return

        img_name = self.image_files[self.current_idx]
        img_path = os.path.join(self.goal_images_dir, img_name)
        
        self.current_cv_image = cv2.imread(img_path)
        if self.current_cv_image is not None:
            _, self.goal_descriptors = self.orb.detectAndCompute(self.current_cv_image, None)
            self.get_logger().info(f"🎯 Target Changed: [{self.current_idx + 1}/{len(self.image_files)}] -> {img_name}")
        else:
            self.get_logger().error(f"Failed to read: {img_path}")

    def _camera_callback(self, msg: Image):
        # 起動直後の猶予期間チェック
        elapsed_sec = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed_sec < self.startup_delay:
            return

        # 判定用画像がない場合はスキップ
        if self.current_cv_image is None or self.goal_descriptors is None:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            _, cur_descriptors = self.orb.detectAndCompute(cv_img, None)

            if cur_descriptors is not None and self.goal_descriptors is not None:
                matches = self.bf.match(self.goal_descriptors, cur_descriptors)
                good_matches = [m for m in matches if m.distance < 45]
                similarity = float(len(good_matches))
                
                self.sim_pub.publish(Float32(data=similarity))

                if similarity >= self.threshold:
                    self.get_logger().info(f"✅ Goal Reached (Matches: {similarity} >= {self.threshold})! Moving to next goal.")
                    self.current_idx += 1
                    self._set_current_image()

        except Exception as e:
            self.get_logger().error(f"Error in similarity calculation: {e}")

    def _next_goal_callback(self, msg: Empty):
        if self.current_idx < len(self.image_files):
            self.current_idx += 1
            self._set_current_image()

    def _timer_callback(self):
        if self.current_cv_image is not None:
            rgb_image = cv2.cvtColor(self.current_cv_image, cv2.COLOR_BGR2RGB)
            msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            self.goal_pub.publish(msg)

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
