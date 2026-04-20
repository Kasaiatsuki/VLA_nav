from setuptools import setup, find_packages
import os

package_name = 'omnivla'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/vla_nav.launch.py', 'launch/vla_nav_7b.launch.py', 'launch/topo_map_nav.launch.py']),
        (os.path.join('share', package_name, 'config'), ['config/vla_nav_node.yaml', 'config/vla_nav_7b_node.yaml', 'config/topo_map_nav_node.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kasai',
    maintainer_email='kasaiatuski@gmail.com',
    description='OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vla_nav_node = omnivla.inference.vla_nav_node:main',
            'vla_nav_7b_node = omnivla.inference.vla_nav_7b_node:main',
            'create_data_vla = omnivla.vla_data_collection.create_data_vla:main',
            'vla_data_collection_node = omnivla.vla_data_collection.monocular_data_collection_node:main',
            'topological_manager_node = omnivla.inference.topological_manager_node:main',
            'capture_goal_images_node = omnivla.inference.capture_goal_images_node:main',
            'topo_localizer_node = omnivla.inference.topo_localizer_node:main',
            'create_topomap_node = omnivla.vla_data_collection.create_topomap_node:main',
        ],
    },
)
