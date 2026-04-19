#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
import pickle
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from rclpy.qos import qos_profile_sensor_data

# サンプリング間隔(秒)。10Hz
SAMPLE_INTERVAL = 0.1
# 予測する軌跡の長さ(未来何ステップ分保存するか)
HORIZON = 8

class MonocularDataCollectionNode(Node):
    """
    単眼カメラ(USBカメラ)とMid-360(FAST-LIO)のオドメトリを記録するノード。
    ジョイスティックのボタン(/flag)で記録の開始/停止を行う。
    """
    def __init__(self) -> None:
        super().__init__('monocular_data_collection_node')
        self.bridge = CvBridge()
        
        # 収集データバッファ: [(画像, [x, y, yaw], タイムスタンプ), ...]
        self.raw_data_buffer = []

        # 最新の状態
        self.latest_image = None
        self.latest_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 状態管理
        self.is_recording = False
        self._odom_received = False
        self._img_received = False

        # --- サブスクライバの設定 ---
        # 画像トピック (QoSはsensor_data推奨)
        self.create_subscription(
            Image, 
            '/image_raw', 
            self._image_callback, 
            qos_profile_sensor_data
        )

        # オドメトリトピック (FAST-LIO等は /Odometry)
        self.create_subscription(
            Odometry, 
            '/Odometry', 
            self._odom_callback, 
            qos_profile_sensor_data
        )

        # 録画開始/停止トリガー (ジョイスティック等のボタン)
        self.create_subscription(
            Empty, 
            '/flag', 
            self._flag_callback, 
            10
        )

        # 収集用タイマー
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        
        self.get_logger().info('==========================================')
        self.get_logger().info('Monocular Data Collection Node Started')
        self.get_logger().info('Waiting for /image_raw and /Odometry...')
        self.get_logger().info('Press A button (or publish to /flag) to START/STOP recording')
        self.get_logger().info('==========================================')

    def _image_callback(self, msg: Image) -> None:
        try:
            # VLAのEdgeモデルに合わせてリサイズ(必要なら)
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 推論時と同じ 512x288 にリサイズして保存
            self.latest_image = cv2.resize(cv_img, (512, 288), interpolation=cv2.INTER_LINEAR)
            if not self._img_received:
                self._img_received = True
                self.get_logger().info('✅ Camera image received!')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def _odom_callback(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        
        # Quaternion -> Yaw
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.latest_pose = np.array([pos.x, pos.y, yaw], dtype=np.float32)
        
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info('✅ Mid-360 Odometry received!')

    def _flag_callback(self, _msg: Empty) -> None:
        self.is_recording = not self.is_recording
        if self.is_recording:
            status = "STARTED"
            # 継続して記録する場合、バッファをクリアするかどうかは運用次第
            # ここでは新規収集とみなしてクリアしない(Ctrl+Cまでを1つのTrajとする)
        else:
            status = "PAUSED"
        self.get_logger().info(f'🔴 Recording {status}. Current buffer size: {len(self.raw_data_buffer)}')

    def timer_callback(self) -> None:
        if not self.is_recording:
            return

        if self.latest_image is None or not self._odom_received:
            return

        # 最新の画像と位置をバッファに保存
        self.raw_data_buffer.append((
            self.latest_image.copy(), 
            self.latest_pose.copy(), 
            time.time()
        ))
        
        if len(self.raw_data_buffer) % 10 == 0:
            self.get_logger().info(f'🟡 Collected {len(self.raw_data_buffer)} frames...')

    def save_data(self) -> None:
        num_samples = len(self.raw_data_buffer)
        if num_samples < HORIZON + 1:
            self.get_logger().warn(f'Not enough data to save ({num_samples} frames). Need at least {HORIZON+1}')
            return

        self.get_logger().info(f'Saving {num_samples} frames to dataset...')

        # 保存先の決定 (srcディレクトリのdataフォルダを優先)
        script_path = Path(__file__).resolve()
        package_root = script_path.parents[3] # omnivla/vla_data_collection/node.py -> 3つ上が VLA_nav
        data_dir = package_root / 'data'
        
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        dataset_name = f'vla_mono_{timestamp_str}'
        dataset_path = data_dir / dataset_name / 'traj_0'
        
        images_dir = dataset_path / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        positions = []
        yaws = []

        for i, (img, pose, ts) in enumerate(self.raw_data_buffer):
            # 画像保存
            cv2.imwrite(str(images_dir / f'{i:05d}.jpg'), img)
            # 軌跡保存
            positions.append([pose[0], pose[1]])
            yaws.append([pose[2]])

        # 軌跡データのpkl保存
        traj_data = {
            "position": np.array(positions, dtype=np.float32),
            "yaw": np.array(yaws, dtype=np.float32)
        }
        with open(dataset_path / 'traj_data.pkl', 'wb') as f:
            pickle.dump(traj_data, f)

        # traj_names.txt の作成
        with open(data_dir / dataset_name / 'traj_names.txt', 'w') as f:
            f.write('traj_0\n')

        self.get_logger().info(f'✅ Successfully saved dataset to: {dataset_path.parent}')

def main(args=None):
    rclpy.init(args=args)
    node = MonocularDataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Finishing data collection...')
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
