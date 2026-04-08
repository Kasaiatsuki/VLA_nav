import pyzed.sl as sl
from typing import Optional, Tuple
import numpy as np
import cv2
import time

class ZedCameraWrapperVLA:
    """
    ZEDカメラの初期化と画像取得をカプセル化したラッパークラス。
    Positional Trackingは無効化し、純粋に画像のみを取得する。
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
        self.runtime_params = sl.RuntimeParameters()
        
        self.output_size = (512, 288)

    def open(self) -> None:
        """カメラを開く"""
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
            

    def grab_data(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        最新のカメラ画像(BGR)を取得して返す。
        Returns:
            image (np.ndarray): 512x288 BGR image
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
            
            timestamp = time.time()
            return resized_image, timestamp
            
        return None

    def close(self) -> None:
        """カメラを適切に閉じる"""
        self.camera.close()
