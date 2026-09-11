"""Unitree G1 在线推理，复用通用 infer 装配。"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.inference import infer_lerobot
from examples.southgrid.unitree_g1.g1_pick_constraints import pin_all_joints


def main():
    sys.argv = [
        sys.argv[0],
        "--agent_name",
        "g1_pick",
        "--level",
        os.environ.get("LEVEL", "default"),
        "--task_config",
        os.path.join(os.path.dirname(__file__), "../../unitree_g1/example.yaml"),
        "--pin_joints",
        *sys.argv[1:],
    ]
    infer_lerobot.main(constraints_hook=pin_all_joints)


if __name__ == "__main__":
    main()
