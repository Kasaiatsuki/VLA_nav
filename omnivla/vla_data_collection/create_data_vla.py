#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
import pickle
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from nav_msgs.msg import Odometry
from nav_msgs.msg import Odometry

# 注意: このスクリプトは omnivla/vla_data_collection ディレクトリに配置される想定です。
# 同じディレクトリにある zed_capture_vla.py からインポートします。
try:
    from omnivla.vla_data_collection.zed_capture_vla import ZedCameraWrapperVLA
except ImportError:
    from zed_capture_vla import ZedCameraWrapperVLA

# サンプリング間隔(秒)。VLAの場合はもう少し早くても良い(例: 0.1秒)
SAMPLE_INTERVAL = 0.1
# 予測する軌跡の長さ(未来何ステップ分保存するか)
HORIZON = 8

class DataCollectionNodeVLA(Node):
    """
    VLAデータ収集用のROS 2ノード。
    ZEDカメラのPositional Trackingを用いて、自身の軌跡(Odometry)を記録し、
    事後処理で各画像ごとに「未来Nステップ分の相対軌跡」を計算して保存する。
    """
    def __init__(self) -> None:
        super().__init__('data_collection_node_vla')
        
        # 収集したデータ: [(画像, [世界X, 世界Y, 世界Yaw], タイムスタンプ), ...]
        self.raw_data_buffer = []

        # FAST-LIO等からの最新オドメトリを保持
        self.latest_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # データ収集のオン/オフ状態管理
        self.is_paused: bool = True

        self.latest_image = None
        self.latest_image_stamp = 0.0

        self.get_logger().info('Initializing ZED camera...')
        self.zed_camera = ZedCameraWrapperVLA(fps=15)
        
        try:
            self.zed_camera.open()
            self.get_logger().info('ZED camera initialized successfully.')
        except RuntimeError as e:
            self.get_logger().error(str(e))
            raise e

        # カメラ取得用の別スレッドを開始 (ROSの処理をブロックさせないため)
        self.camera_thread_running = True
        self.camera_thread = threading.Thread(target=self._camera_grab_loop, daemon=True)
        self.camera_thread.start()

        # ジョイコントローラ等からのトリガー
        self.create_subscription(Empty, 'flag', self._flag_callback, 10)
        
        # オドメトリのサブスクライブ
        self.create_subscription(Odometry, '/Odometry', self._odom_callback, 10)
        
        # タイマーによるデータ収集
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        self.get_logger().info('Started VLA data collection node. Waiting for /flag (start signal).')

    def _camera_grab_loop(self):
        """ZEDカメラの画像を裏で取得し続ける専用スレッド"""
        while self.camera_thread_running:
            result = self.zed_camera.grab_data()
            if result is not None:
                self.latest_image, self.latest_image_stamp = result
            time.sleep(0.01) # 短いスリープでCPU負荷を下げる

    def _odom_callback(self, msg: Odometry) -> None:
        """FAST-LIO等からのオドメトリを受信し、最新の姿勢を更新する"""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        x = pos.x
        y = pos.y
        # Quaternion から Yaw の計算
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        self.latest_pose = np.array([x, y, yaw], dtype=np.float32)
        if not hasattr(self, '_odom_received'):
            self._odom_received = True
            self.get_logger().info('✅ FAST-LIO Odometry received for the first time!')

    def _flag_callback(self, _msg: Empty) -> None:
        """/flagトピック受信時に録画をトグル(開始/停止)する"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.get_logger().info('⏸️ Data collection PAUSED')
            # もし既にデータが溜まっていたら、ここでセーブするのも手
        else:
            self.get_logger().info('▶️ Data collection RECORDING')

    def timer_callback(self) -> None:
        """定期的に最新の画像とポーズを取得してバッファに詰める"""
        if self.is_paused:
            return

        if self.latest_image is None:
            self.get_logger().warning('Waiting for camera frame...')
            return

        image = self.latest_image.copy()
        timestamp = self.latest_image_stamp
        pose = self.latest_pose.copy()
        
        self.raw_data_buffer.append((image, pose, timestamp))
        
        self.get_logger().info(f'🟡Collected raw data #{len(self.raw_data_buffer)} (Pose: X={pose[0]:.2f}, Y={pose[1]:.2f}, Yaw={pose[2]:.2f})')

    def save_data(self) -> None:
        """
        終了時に、バッファされた生データ(画像+絶対座標)を処理し、VLA用の「画像+未来の相対軌跡」に変換して保存する。
        """
        num_samples = len(self.raw_data_buffer)
        if num_samples < HORIZON + 1:
            self.get_logger().info(f'🔴Not enough data to save. Need at least {HORIZON + 1} samples (got {num_samples}).')
            return

        self.get_logger().info('🔵Processing buffer into trajectories...')
        
        # データは常に VLA_nav/data/ 以下に保存する（インストールディレクトリではなくソースを使う）
        package_root = Path(__file__).resolve()
        # インストール済み or ソース直接どちらでも VLA_nav のルートを見つける
        for parent in package_root.parents:
            if (parent / "vla-scripts").exists():
                package_root = parent
                break
        else:
            # フォールバック: カレントディレクトリ
            package_root = Path.cwd()
        data_base_dir = package_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset_dir = data_base_dir / f'vla_{timestamp}_dataset'
        
        # 構造: datasets/traj_X/ (画像郡とtraj_data.pklを持つ)
        traj_dir = dataset_dir / f'traj_0'
        
        images_dir = traj_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 全時刻のデータを配列化
        positions = []
        yaws = []
        
        for idx in range(num_samples):
            image, pose, t = self.raw_data_buffer[idx]
            
            # 画像の保存
            image_path = images_dir / f'{idx:05d}.jpg'
            cv2.imwrite(str(image_path), image) # VLAではJPEGが多いがPNGでも可
            
            # 軌跡データの蓄積
            positions.append([pose[0], pose[1]]) # [x, y]
            yaws.append([pose[2]]) # [yaw]

        positions_np = np.array(positions, dtype=np.float32)
        yaws_np = np.array(yaws, dtype=np.float32)
        
        traj_data = {
            "position": positions_np,
            "yaw": yaws_np
        }
        
        # VLA（GNM等）の仕様に準じて、全時間ステップの絶対座標をpkl形式で保存する
        pkl_path = traj_dir / 'traj_data.pkl'
        with open(pkl_path, "wb") as f:
            pickle.dump(traj_data, f)
            
        # dataset のルートにパスを羅列した traj_names.txt を作る
        with open(dataset_dir / "traj_names.txt", "w") as f:
             f.write("traj_0\n")

        self.get_logger().info(f'✅Saved Trajectory Data: {num_samples} steps to {dataset_dir}')

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollectionNodeVLA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.save_data()
        node.camera_thread_running = False
        if hasattr(node, 'zed_camera') and node.zed_camera:
             node.zed_camera.close()
        node.camera_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
