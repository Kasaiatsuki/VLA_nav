import pyzed.sl as sl
from typing import Optional, Tuple
import numpy as np
import cv2
import time

class ZedCameraWrapperVLA:
    """
    ZEDカメラの初期化と画像取得、および Positional Tracking をカプセル化したラッパークラス。
    VLAが学習するための正確な軌跡情報を得るために、ZED SDKの内蔵Odometryを利用。
    """
    def __init__(self, fps: int = 15, resolution=sl.RESOLUTION.HD720) -> None:
        self.fps = fps
        self.resolution = resolution
        
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = self.resolution
        self.init_params.camera_fps = self.fps
        
        # ROS準拠の右手系Z-Up (X前, Y左, Z上)
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD 
        self.init_params.coordinate_units = sl.UNIT.METER # メートル単位
        
        self.zed_image = sl.Mat()
        self.zed_pose = sl.Pose()
        self.runtime_params = sl.RuntimeParameters()
        
        self.output_size = (512, 288)

    def open(self) -> None:
        """カメラを開き、Positional Trackingを有効化する"""
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
            
        # Positional Tracking の有効化
        tracking_params = sl.PositionalTrackingParameters()
        # SLAMのループクローズは無効化し、純粋なOdometryとする
        tracking_params.enable_area_memory = False 
        
        err = self.camera.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.camera.close()
            raise RuntimeError(f"Failed to enable positional tracking: {err}")
            

    def grab_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        最新のカメラ画像(BGR)と、Positional Trackingによる現在の(X, Y, Yaw)を取得して返す。
        Returns:
            image (np.ndarray): 512x288 BGR image
            pose (np.ndarray): [x, y, yaw] in meters and radians
            timestamp (float): Current timestamp
        """
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # --- 1. 画像の取得とリサイズ ---
            self.camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            image_data = self.zed_image.get_data()
            if image_data is None: 
                return None
            
            bgr_image = image_data[:, :, :3].copy() if image_data.shape[2] == 4 else image_data
            resized_image = cv2.resize(bgr_image, self.output_size, interpolation=cv2.INTER_LINEAR)
            
            # --- 2. 位置と姿勢(Pose)の取得 ---
            state = self.camera.get_position(self.zed_pose, sl.REFERENCE_FRAME.WORLD)
            
            x, y, yaw = 0.0, 0.0, 0.0
            
            if state == sl.POSITIONAL_TRACKING_STATE.OK:
                translation = self.zed_pose.get_translation(sl.Translation()).get()
                rotation = self.zed_pose.get_euler_angles() # [roll, pitch, yaw]
                
                # Z_UP_X_FWD なので Xが前, Yが左。YawはZ軸回り。
                x = translation[0]
                y = translation[1]
                yaw = rotation[2]
            else:
                print(f"[Warning] Positional Tracking State: {state}. Using 0.0 for X,Y,Yaw.")

            pose = np.array([x, y, yaw], dtype=np.float32)
            timestamp = time.time()
            
            return resized_image, pose, timestamp
            
        return None

    def close(self) -> None:
        """カメラとTrackingを適切に閉じる"""
        self.camera.disable_positional_tracking()
        self.camera.close()
