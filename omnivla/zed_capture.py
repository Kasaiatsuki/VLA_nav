import pyzed.sl as sl
from typing import Optional
import numpy as np
import cv2

class ZedCameraWrapper:
    """
    ZEDカメラの初期化と画像取得をカプセル化したラッパークラス。
    VLA_nav/omnivla/inference/vla_nav_node.pyで使用されます。
    """
    def __init__(self, fps: int = 15, resolution=sl.RESOLUTION.HD720) -> None:
        self.fps = fps
        self.resolution = resolution
        
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = self.resolution
        self.init_params.camera_fps = self.fps
        
        self.zed_image = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        
        # モデル入力サイズに合わせたリサイズ（必要に応じて調整）
        # 元のomnivla_inference_node.pyでは self.zed.grab_image() の後でPILでリサイズしていたが、
        # ここでOpenCVでリサイズして返す
        self.output_resolution = sl.Resolution(640, 360)

    def open(self) -> None:
        """カメラを開き、初期化する"""
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

    def grab_image(self) -> Optional[np.ndarray]:
        """最新のカメラ画像(左目)をBGR形式(3ch)かつリサイズ済み(640x360)で取得して返す"""
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            image = self.zed_image.get_data()
            if image is None: return None
            
            # 4ch(RGBA) -> 3ch(BGR)
            bgr_image = image[:, :, :3] if image.shape[2] == 4 else image
            
            # 640x360 にリサイズ
            return cv2.resize(bgr_image, (640, 360), interpolation=cv2.INTER_LINEAR)
        return None

    def close(self) -> None:
        """カメラを適切に閉じる"""
        self.camera.close()
