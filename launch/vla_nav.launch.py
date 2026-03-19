import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('omnivla')
    config_file = os.path.join(pkg_share, 'config', 'vla_nav_node.yaml')

    vla_nav_node = Node(
        package='omnivla',
        executable='vla_nav_node',
        name='vla_nav_node',
        output='screen',
        parameters=[config_file]
    )

    topological_manager_node = Node(
        package='omnivla',
        executable='topological_manager_node',
        name='topological_manager_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        vla_nav_node,
        topological_manager_node
    ])
