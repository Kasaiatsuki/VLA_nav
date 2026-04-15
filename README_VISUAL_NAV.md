# Topological Visual Navigation (feature/visual-goal-nav)

このブランチでは、OmniVLA-edge を用いて **「単一の言語指示」ではなく「複数の目標画像（パンくず）を順に辿る」トポロジカル・ナビゲーション（Visual Goal Navigation）機能** を実装しています。

## 概要
通常の推論ノード (`vla_nav_node`) は `modality_id = 7` (言語入力モード) で動作しますが、`/goal_image` トピックから画像を受信すると自動的に `modality_id = 6` (画像ナビモード: ego-image only) に切り替わります。
事前に撮影した目標画像のリストを順番に送信し続けるマネージャープログラムを併用することで、長距離の滑らかな自律走行を実現します。

---

## 使い方・実行手順

### 1. ワークスペースのビルド
追加された各種ノードを ROS2 に認識させるため、ビルドを実行します。
```bash
conda activate omnivla_foxy
cd ~/ros2_ws
colcon build --packages-select omnivla
source install/setup.bash
sed -i '1s|.*|#!/usr/bin/env python3|' ~/kasai_ws/install/omnivla/lib/omnivla/vla_nav_node
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
```

### 2. 目標画像（パンくず）の撮影・保存
ロボットをマニュアル操作で目標とするルートに沿って走らせながら、チェックポイントごとに写真を撮影します。

1. **撮影ノードの起動:**
   ```bash
   ros2 run omnivla capture_goal_images_node
   ```
2. **撮影の実行:**
   コントローラの録画ボタン（`/flag` または `/capture_image` トピックを出すボタン）を押すたびに、ZEDカメラから現在の景色が1枚撮影され、以下のフォルダに `001.jpg, 002.jpg...` と連番で保存されます。
   保存先: `src/VLA_nav/omnivla/inference/goal_images/`

### 3. トポロジカル・ナビゲーションの実行
実際に撮影した画像のリストを辿って、ロボットを自律走行（推論）させます。

1. **推論ノード（メイン）の起動:**
   ```bash
   ros2 launch omnivla vla_nav_node.launch.py
   ```
   （起動直後はデフォルトの言語モードで待機します）

2. **マネージャーノード（目標画像配信）の起動:**
   別のターミナルを開き、マネージャーノードを起動します。
   ```bash
   ros2 run omnivla topological_manager_node
   ```
   自動的に `goal_images/` フォルダ内の最初の画像 (`001.jpg`) が読み込まれ、配信が開始されます。画像データを受信した推論ノードは自動的に「画像ナビモード (ID:6)」に切り替わり、ロボットが指定された画像に向かって進み始めます。

3. **次の目標画像へ切り替える（トリガー）:**
   ロボットが目標の地点に到着したら、手動で以下のトリガーを送ることで「次の画像(`002.jpg` 等)」へ目標を進ませます。
   ```bash
   ros2 topic pub --once /next_goal std_msgs/msg/Empty {}
   ```
   ※ 実用時は、この送信コマンドをジョイスティックの空きボタンに割り当てておくか、画像の一致率で自動進行させる Node を別途噛ませることを推奨します。
