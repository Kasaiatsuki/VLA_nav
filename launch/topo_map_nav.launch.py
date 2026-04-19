#!/usr/bin/env python3
"""
topo_map_nav.launch.py

起動するノード:
  1. vla_nav_node     - OmniVLA-edge による速度指令生成
  2. topo_localizer_node - PlaceNet + ベイズフィルタによる自己位置推定 → プロンプト配信
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share  = get_package_share_directory('omnivla')
    config_file = os.path.join(pkg_share, 'config', 'topo_map_nav_node.yaml')

    vla_nav_node = Node(
        package='omnivla',
        executable='vla_nav_node',
        name='vla_nav_node',
        output='screen',
        parameters=[config_file]
    )

    topo_localizer_node = Node(
        package='omnivla',
        executable='topo_localizer_node',
        name='topo_localizer_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        vla_nav_node,
        topo_localizer_node,
    ])
