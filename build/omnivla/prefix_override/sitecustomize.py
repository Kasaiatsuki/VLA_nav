import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/kasai/ros2_ws/src/VLA_nav/install/omnivla'
