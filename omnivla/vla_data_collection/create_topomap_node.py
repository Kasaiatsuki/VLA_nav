#!/usr/bin/env python3
"""
create_topomap_node.py

役割:
  VLA_nav 用のトポロジカルマップ（topomap.yaml）を対話的に作成するツール。
  ロボットをマニュアル操縦しながらカメラ画像を順番に撮影し、
  各地点のアクションを指定して保存する。
  撮影完了後、PlaceNet で全画像の特徴量を抽出して YAML を書き出す。

操作方法（OpenCV ウィンドウアクティブな状態で）:
  [Space]   現在フレームをノードとして登録（現在設定中のアクションで確定）
  [f]       次のノードのアクションを "straight"（直進）に設定
  [l]       次のノードのアクションを "left"（左折）に設定
  [g]       次のノードのアクションを "right"（右折）に設定
  [p]       次のノードのアクションを "stop"（停止）に設定
  [u]       直前に登録したノードを取り消し（Undo）
  [w/Enter] 特徴量抽出・YAML 保存して終了
  [q/Esc]   保存せず終了

出力:
  <output_dir>/topomap.yaml    : トポロジカルマップ
  <output_dir>/images/         : 参照画像ディレクトリ

Usage:
  ros2 run omnivla create_topomap_node \
      --ros-args \
      -p placenet_path:=/path/to/placenet.pt \
      -p output_dir:=/path/to/map \
      -p camera_topic:=/image_raw
"""

import os
import sys
import shutil
import math
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from PIL import Image as PILImage

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ============================================================
# アクション設定
# ============================================================
ACTIONS = {
    'f': 'straight',
    'l': 'left',
    'g': 'right',
    'p': 'stop',
}

ACTION_COLORS = {
    'straight': (0, 200, 0),
    'left':     (200, 100, 0),
    'right':    (0, 100, 200),
    'stop':     (0, 0, 220),
}


class CreateTopoMapNode(Node):
    """対話的なトポロジカルマップ作成ノード。"""

    def __init__(self) -> None:
        super().__init__('create_topomap_node')

        # -------------------------------------------------------
        # パラメータ
        # -------------------------------------------------------
        self.declare_parameter(
            'placenet_path',
            '/home/kasai/ros2_ws/install/imitation_nav/share/imitation_nav'
            '/weights/placenet/placenet.pt')
        self.declare_parameter(
            'output_dir',
            os.path.join(os.path.expanduser('~'), 'topomap_output'))
        self.declare_parameter('camera_topic', '/image_raw')

        self.placenet_path = self.get_parameter('placenet_path').value
        self.output_dir    = self.get_parameter('output_dir').value
        camera_topic       = self.get_parameter('camera_topic').value

        # -------------------------------------------------------
        # 出力ディレクトリの準備
        # -------------------------------------------------------
        self.images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.get_logger().info(f'Output directory: {self.output_dir}')

        # -------------------------------------------------------
        # 内部状態
        # -------------------------------------------------------
        self.bridge          = CvBridge()
        self.latest_cv_image = None

        # 登録済みノード情報（後で特徴量を入れる）
        # Each entry: {'id': int, 'image': str, 'action': str, 'feature': None}
        self.nodes: list[dict] = []

        # 現在設定中のアクション
        self.current_action = 'straight'

        # 終了フラグ
        self.save_and_quit = False
        self.quit_without_save = False

        # -------------------------------------------------------
        # サブスクライバ
        # -------------------------------------------------------
        self.create_subscription(Image, camera_topic, self._image_callback, 10)
        self.get_logger().info(f'Subscribed to {camera_topic}')

        # -------------------------------------------------------
        # OpenCV ウィンドウの初期化
        # -------------------------------------------------------
        cv2.namedWindow('TopoMap Creator', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('TopoMap Creator', 800, 500)

        # -------------------------------------------------------
        # タイマー（表示更新）
        # -------------------------------------------------------
        self.create_timer(0.05, self._timer_callback)  # 20Hz

        self.get_logger().info('=== Topomap Creator Ready ===')
        self._print_help()

    # ===========================================================
    # ヘルプ表示
    # ===========================================================

    def _print_help(self) -> None:
        self.get_logger().info(
            '\n--- CONTROLS ---\n'
            '  [Space]  : Capture current frame as a node\n'
            '  [f]      : Action = straight\n'
            '  [l]      : Action = left\n'
            '  [g]      : Action = right\n'
            '  [p]      : Action = stop\n'
            '  [u]      : Undo last node\n'
            '  [w/Enter]: Extract features and save topomap.yaml\n'
            '  [q/Esc]  : Quit without saving\n'
        )

    # ===========================================================
    # コールバック
    # ===========================================================

    def _image_callback(self, msg: Image) -> None:
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def _timer_callback(self) -> None:
        """表示更新とキーボード入力処理。"""
        if self.save_and_quit or self.quit_without_save:
            return

        if self.latest_cv_image is None:
            return

        display = self._build_display(self.latest_cv_image)
        cv2.imshow('TopoMap Creator', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            self._capture_node()
        elif key in [ord(k) for k in ACTIONS.keys()]:
            action_key = chr(key)
            self.current_action = ACTIONS[action_key]
            self.get_logger().info(f'Action set to: {self.current_action}')
        elif key == ord('u'):
            self._undo()
        elif key in [ord('w'), 13]:  # 'w' または Enter
            self.save_and_quit = True
        elif key in [ord('q'), 27]:  # 'q' または Esc
            self.quit_without_save = True

    # ===========================================================
    # 表示
    # ===========================================================

    def _build_display(self, frame: np.ndarray) -> np.ndarray:
        """カメラ映像にHUD情報を重ねた表示フレームを作成する。"""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # ====== 左パネル: 現在状態 ======
        # 現在のアクション
        action_color = ACTION_COLORS.get(self.current_action, (200, 200, 200))
        cv2.rectangle(canvas, (0, 0), (280, 40), (30, 30, 30), -1)
        cv2.putText(canvas, f'Action: {self.current_action}',
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)

        # ノード数
        cv2.rectangle(canvas, (0, 40), (280, 75), (20, 20, 20), -1)
        cv2.putText(canvas, f'Nodes: {len(self.nodes)}',
                    (8, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        # ====== 右パネル: キーガイド ======
        guide_x = w - 220
        cv2.rectangle(canvas, (guide_x, 0), (w, 180), (30, 30, 30), -1)
        guides = [
            ('Space', 'Capture'),
            ('r', 'roadside'),
            ('f', 'straight'),
            ('l', 'left'),
            ('g', 'right'),
            ('p', 'stop'),
            ('u', 'Undo'),
            ('w', 'Save & Exit'),
            ('q', 'Quit'),
        ]
        for i, (k, desc) in enumerate(guides):
            cv2.putText(canvas, f'[{k}] {desc}',
                        (guide_x + 6, 18 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ====== 最近登録したノードのサムネイル ======
        thumb_area_x = 0
        thumb_area_y = h - 90
        cv2.rectangle(canvas, (thumb_area_x, thumb_area_y), (w, h), (25, 25, 25), -1)

        MAX_THUMBS = min(len(self.nodes), 8)
        for i, node in enumerate(self.nodes[-MAX_THUMBS:]):
            img_path = os.path.join(self.images_dir, node['image'])
            thumb = cv2.imread(img_path)
            if thumb is not None:
                thumb = cv2.resize(thumb, (80, 60))
                x_off = thumb_area_x + i * 90 + 5
                y_off = thumb_area_y + 15
                canvas[y_off:y_off+60, x_off:x_off+80] = thumb
                color = ACTION_COLORS.get(node['action'], (200, 200, 200))
                cv2.putText(canvas, f"{node['id']}:{node['action'][:3]}",
                            (x_off, y_off + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return canvas

    # ===========================================================
    # ノード操作
    # ===========================================================

    def _capture_node(self) -> None:
        """現在のフレームをノードとして登録する。"""
        if self.latest_cv_image is None:
            return

        node_id  = len(self.nodes)
        img_name = f'img{node_id:05d}.png'
        img_path = os.path.join(self.images_dir, img_name)

        cv2.imwrite(img_path, self.latest_cv_image)

        self.nodes.append({
            'id':      node_id,
            'image':   img_name,
            'action':  self.current_action,
            'feature': None,  # 後で抽出
        })

        self.get_logger().info(
            f'[Node {node_id}] Captured: {img_name}, action={self.current_action}')

    def _undo(self) -> None:
        """直前のノードを取り消す。"""
        if not self.nodes:
            self.get_logger().warn('No nodes to undo.')
            return

        last = self.nodes.pop()
        img_path = os.path.join(self.images_dir, last['image'])
        if os.path.exists(img_path):
            os.remove(img_path)
        self.get_logger().info(f'[Undo] Removed node {last["id"]} ({last["image"]})')

    # ===========================================================
    # 特徴量抽出
    # ===========================================================

    def _extract_all_features(self) -> bool:
        """全ノードの画像に対して PlaceNet で特徴量を抽出する。"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Loading PlaceNet from {self.placenet_path} ...')

        if not os.path.exists(self.placenet_path):
            self.get_logger().error(f'PlaceNet not found: {self.placenet_path}')
            return False

        placenet = torch.jit.load(self.placenet_path)
        placenet.to(device)
        placenet.eval()
        self.get_logger().info(f'PlaceNet loaded on {device}.')

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for node in self.nodes:
            img_path = os.path.join(self.images_dir, node['image'])
            bgr = cv2.imread(img_path)
            if bgr is None:
                self.get_logger().error(f'Cannot read: {img_path}')
                return False

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb).resize((85, 85))
            arr = (np.array(pil, dtype=np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = placenet.forward([tensor]).squeeze(0)

            node['feature'] = feat.cpu().numpy().tolist()
            self.get_logger().info(
                f'  [{node["id"]:4d}/{len(self.nodes)-1}] {node["image"]} ... done')

        return True

    # ===========================================================
    # YAML 書き出し
    # ===========================================================

    def _save_yaml(self) -> None:
        """topomap.yaml を書き出す。"""
        yaml_path = os.path.join(self.output_dir, 'topomap.yaml')

        yaml_nodes = []
        for i, node in enumerate(self.nodes):
            next_id = i + 1 if i + 1 < len(self.nodes) else None
            edges = []
            if next_id is not None:
                edges.append({'target': next_id, 'action': node['action']})
            # 末尾ノードは stop アクションとして記録
            elif len(self.nodes) > 0:
                edges.append({'target': i, 'action': 'stop'})

            yaml_nodes.append({
                'id':      node['id'],
                'image':   node['image'],
                'feature': node['feature'],
                'edges':   edges,
            })

        with open(yaml_path, 'w') as f:
            yaml.dump({'nodes': yaml_nodes}, f,
                      default_flow_style=False, allow_unicode=True)

        self.get_logger().info(
            f'\n=== Saved ===\n'
            f'  YAML  : {yaml_path}\n'
            f'  Images: {self.images_dir}\n'
            f'  Nodes : {len(self.nodes)}\n'
        )
        print(f'\n[DONE] topomap.yaml saved to: {yaml_path}')
        print(f'[DONE] Images saved to: {self.images_dir}')

    # ===========================================================
    # 終了処理（メインループから呼ばれる）
    # ===========================================================

    def finalize(self) -> None:
        """保存処理と終了。spin ループを抜けた後に呼ぶ。"""
        cv2.destroyAllWindows()

        if self.quit_without_save:
            self.get_logger().info('Quit without saving.')
            return

        if not self.nodes:
            self.get_logger().warn('No nodes captured. Nothing to save.')
            return

        self.get_logger().info(
            f'Extracting features for {len(self.nodes)} nodes ...')
        success = self._extract_all_features()
        if not success:
            self.get_logger().error('Feature extraction failed. Map not saved.')
            return

        self._save_yaml()


# ============================================================
# エントリポイント
# ============================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = CreateTopoMapNode()

    try:
        while rclpy.ok() and not node.save_and_quit and not node.quit_without_save:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')

    node.finalize()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
