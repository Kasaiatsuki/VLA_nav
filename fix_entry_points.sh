#!/bin/bash
# colcon build 後にエントリーポイントを pkg_resources なしの形式に差し替えるスクリプト
# Usage: bash fix_entry_points.sh

INSTALL_DIR="$HOME/kasai_ws/install/omnivla/lib/omnivla"

for NODE in vla_nav_node create_data_vla topological_manager_node capture_goal_images_node; do
    MODULE=""
    case $NODE in
        vla_nav_node) MODULE="omnivla.inference.vla_nav_node" ;;
        create_data_vla) MODULE="omnivla.vla_data_collection.create_data_vla" ;;
        topological_manager_node) MODULE="omnivla.inference.topological_manager_node" ;;
        capture_goal_images_node) MODULE="omnivla.inference.capture_goal_images_node" ;;
    esac

    cat > "$INSTALL_DIR/$NODE" << EOF
#!/usr/bin/python3
# Fixed: pkg_resources を使わない直接 import 形式
import re
import sys

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?\$', '', sys.argv[0])
    from $MODULE import main
    sys.exit(main())
EOF
    chmod +x "$INSTALL_DIR/$NODE"
    echo "Fixed: $INSTALL_DIR/$NODE"
done

echo "Done! All entry points have been fixed."
