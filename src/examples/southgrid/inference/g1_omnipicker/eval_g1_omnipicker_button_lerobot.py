"""G1 OmniPicker 按钮任务在线推理，复用通用 infer 装配。"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from examples.inference import infer_lerobot


def main():
    sys.argv = [
        sys.argv[0],
        "--agent_name",
        "g1_omnipicker",
        "--level",
        os.environ.get("LEVEL", "default"),
        "--task_config",
        os.path.join(os.path.dirname(__file__), "../../configs/example.yaml"),
        "--prompt",
        "按红色按钮",
        *sys.argv[1:],
    ]
    infer_lerobot.main()


if __name__ == "__main__":
    main()
