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
        (os.path.join('share', package_name, 'launch'), ['launch/vla_nav.launch.py']),
        (os.path.join('share', package_name, 'config'), ['config/vla_nav_node.yaml']),
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
        ],
    },
)
