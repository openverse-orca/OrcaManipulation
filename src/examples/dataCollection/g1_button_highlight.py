"""G1 OmniPicker 电柜按钮颜色高亮工具脚本。

加载场景后，找到电柜上的四个按钮 body，把其所属 geom 的 rgba 分别染成：
  button01 → 红   button02 → 绿   Button03 → 蓝   Button04 → 黄
颜色写入 env._mjModel.geom_rgba，在 MuJoCo viewer（截图/仿真窗口）中立即生效。

同时打印每个按钮的精确 body 名、所有 geom ID 及其位置（base 坐标系），
方便确认命名和坐标是否正确。

用法（先 conda activate orcalab_lerobot）：
  cd src/examples/dataCollection
  python -u g1_button_highlight.py --task_config example.yaml
可选：
  --hold 10   # 高亮后保持仿真运行的秒数（默认 10 秒，Ctrl+C 随时退出）
"""
import argparse
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
from yaml import Loader, load

from conf import g1_omnipicker_conf as agent_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage
from scene.scene_manager import SceneManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

# 精确 body 名 -> 高亮颜色 RGBA
# body名 -> (编号标签, 颜色描述)
BUTTON_LABELS = {
    "Group_Static_ElectricalCabinet_button01": ("01", "红"),
    "Group_Static_ElectricalCabinet_button02": ("02", "绿"),
    "Group_Static_ElectricalCabinet_Button04": ("04", "黄"),
    # Button03 当前场景不存在
    # "Group_Static_ElectricalCabinet_Button03": ("03", "蓝"),
}


def highlight_buttons(env, scene_manager) -> None:
    """在每个按钮正上方放置 block 标记 actor，并打印按钮坐标。"""
    base_name = env.body(agent_conf.base_body)

    # block 资产路径（场景里已有该资产）
    BLOCK_ASSET = (
        "assets/e071469a36d3c8aa/southgrid_competetion_2026"
        "/csg_robotcompetition_2026/props/blocks/block"
    )

    print("\n" + "=" * 60, flush=True)
    print("  电柜按钮位置标记", flush=True)
    print("=" * 60, flush=True)

    for btn_body, (label, color_name) in BUTTON_LABELS.items():
        try:
            pos_b = env.query_position_body_B(btn_body, base_name)
            pos_str = f"[{pos_b[0]:.4f}, {pos_b[1]:.4f}, {pos_b[2]:.4f}]"
        except Exception:
            print(f"  [{label}] {btn_body}  → 坐标查询失败", flush=True)
            continue

        print(
            f"  [{label}/{color_name}] {btn_body}\n"
            f"        base坐标={pos_str}",
            flush=True,
        )

        # 将按钮 base 坐标转回 OrcaStudio 世界坐标，放置标记 block
        # base_body 在世界坐标系中的位姿（已知：robot origin ≈ [18.91, -22.84, 0.20]，yaw=90°）
        # 直接用 query_position_body_B 的逆：world = R_base @ pos_b + base_world
        # 此处近似：从 XML 已知 base 世界坐标 + yaw=90°，故 world_x = base_wx + pos_b_y (因旋转90°)
        # 简化：直接用 env 提供的接口查 base body 世界位置
        try:
            import numpy as np
            # base body 在世界坐标（近似，yaw≈90°）
            BASE_WORLD = np.array([18.910844, -22.839414, 0.202662])
            # yaw 90°: world_x += pos_b_y, world_y -= pos_b_x, world_z += pos_b_z
            world_pos = np.array([
                BASE_WORLD[0] + pos_b[1],
                BASE_WORLD[1] - pos_b[0],
                BASE_WORLD[2] + pos_b[2] + 0.06,  # 稍微抬高 6cm 悬在按钮上方
            ])
            marker_name = f"_btn_marker_{label}"
            scene_manager.add_actor(
                marker_name, BLOCK_ASSET,
                world_pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                scale=0.3,
            )
            print(f"        ✓ 标记 actor '{marker_name}' 已放置于 {np.round(world_pos, 3).tolist()}", flush=True)
        except Exception as e:
            print(f"        ⚠ 标记 actor 放置失败: {e}", flush=True)

    print("=" * 60 + "\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 电柜按钮颜色高亮工具")
    parser.add_argument("--task_config", type=str, default="example.yaml")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--hold", type=float, default=10.0, help="高亮后保持运行秒数")
    args = parser.parse_args()

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.show_ui_message(1, "按钮高亮测试 红=01 绿=02 蓝=03 黄=04", "0xffff00", showtime=0)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

    _storage = G1OmniPickerDataStorage(dataset_path="/tmp/_g1_btn_hl")
    _n_motor = (len(agent_conf.gripper_l["actuator_names"])
                + len(agent_conf.gripper_r["actuator_names"]))

    def _obs_cb(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return _storage.obs_callback(env)

    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_cb,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        print("update_scene 失败，退出", flush=True)
        env.close()
        return

    # 在按钮上方放置标记 actor 并打印坐标
    highlight_buttons(env, scene_manager)

    # 列出模型中所有含 "button" 的 body，方便排查命名
    all_btn_bodies = [b for b in env.model.get_body_names() if b and "button" in b.lower()]
    if all_btn_bodies:
        print("  模型中所有含 'button' 的 body：", flush=True)
        for b in sorted(all_btn_bodies):
            print(f"    {b}", flush=True)
    else:
        print("  ⚠ 未找到 button body，请确认场景已加载电柜", flush=True)
    print("", flush=True)

    print(f"  保持仿真运行 {args.hold:.0f} 秒（Ctrl+C 随时退出）...", flush=True)
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.hold:
            action = manager.run_controllers()
            env.step(action)
            time.sleep(float(env.dt))
    except KeyboardInterrupt:
        pass

    env.close()
    print("退出。", flush=True)


if __name__ == "__main__":
    main()
