"""数采启动脚本（包装 data_collection_mp.py）。

在 data_collection_mp.py 的基础上增加 --dataset_path 参数，
通过软链接方式将产出目录重定向到指定路径，不修改原始脚本。

同时在每条 episode 开始时打印 C12C 和机械臂末端的世界坐标。

用法：
  python run_collection.py \
      --pose_file pose_mp_no_barcode.yaml \
      --rand_file rand_no_barcode.yaml \
      --dataset_path /home/dht/OrcaManipulation_v3/dataset \
      --episodes 5
"""

import argparse
import os
import sys
import traceback
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))

# ── 抓取成功判据常量 ──────────────────────────────────────────────────────────
# 抓取成功判据：双侧 pad 接触 C12C 且 C12C 被抬离初始高度超过此阈值（米）。
# 仅单帧接触太弱（夹爪闭合瞬间擦碰会误判），加抬升条件可滤除：真实抓取抬升 ~0.35m，
# 误判仅 ~0.01m，0.05m 阈值区分度极大。
GRASP_LIFT_THRESHOLD = 0.05


def _classify_grip_body(bname: str):
    """左臂夹爪 body 名 → 'pad' / 'front' / None（仅内侧 pad 视为有效抓取面）。"""
    b = bname.lower()
    if "l_left_pad" in b or "l_right_pad" in b:
        return "pad"
    if "l_left_follower" in b or "l_right_follower" in b:
        return "front"
    for k in ("l_left_driver", "l_right_driver", "l_left_coupler", "l_right_coupler",
              "l_left_spring_link", "l_right_spring_link"):
        if k in b:
            return "front"
    return None


def _build_grasp_index(env, mp):
    """预解析 C12C geom 集合 + 左夹爪 geom 分类 + 两侧 pad body id。"""
    mj_model = env.gym._mjModel
    c12c_body_name = mp._resolve_body_name(env, "C12C_3c_c12c")
    c12c_bid = mj_model.body(c12c_body_name).id
    c12c_geoms = set()
    grip_geoms = {}  # gid -> 'pad' / 'front'
    for gid in range(mj_model.ngeom):
        bid = int(np.asarray(mj_model.geom(gid).bodyid).flat[0])
        if bid == c12c_bid:
            c12c_geoms.add(gid)
            continue
        bname = mj_model.body(bid).name or ""
        cls = _classify_grip_body(bname)
        if cls is not None:
            grip_geoms[gid] = cls
    lpad_bid = next((mj_model.body(i).id for i in range(mj_model.nbody)
                     if "l_left_pad" in (mj_model.body(i).name or "")), None)
    rpad_bid = next((mj_model.body(i).id for i in range(mj_model.nbody)
                     if "l_right_pad" in (mj_model.body(i).name or "")), None)
    return {"c12c_geoms": c12c_geoms, "grip_geoms": grip_geoms,
            "lpad_bid": lpad_bid, "rpad_bid": rpad_bid, "c12c_bid": c12c_bid}


def _c12c_world_z(env, gidx):
    """C12C 本体当前世界 z 坐标。"""
    return float(env.gym._mjData.xpos[gidx["c12c_bid"]][2])


def _both_pads_contact_c12c(env, gidx):
    """当前帧是否 l_left_pad 和 l_right_pad 两侧都在接触 C12C。"""
    mj_data = env.gym._mjData
    mj_model = env.gym._mjModel
    c12c_geoms = gidx["c12c_geoms"]
    grip_geoms = gidx["grip_geoms"]
    sides = set()
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 < 0 or g2 < 0:
            continue
        if g1 in c12c_geoms and g2 in grip_geoms and grip_geoms[g2] == "pad":
            gg = g2
        elif g2 in c12c_geoms and g1 in grip_geoms and grip_geoms[g1] == "pad":
            gg = g1
        else:
            continue
        bname = mj_model.body(
            int(np.asarray(mj_model.geom(gg).bodyid).flat[0])
        ).name or ""
        if "l_left_pad" in bname.lower():
            sides.add("left")
        elif "l_right_pad" in bname.lower():
            sides.add("right")
    return "left" in sides and "right" in sides


DEFAULT_LEVEL = "competition_warehouse"
DEFAULT_TASK_CONFIG = "competition_warehouse.yaml"

# C12C 干净场景真实默认 free-joint qpos [x,y,z, qw,qx,qy,qz]（--dump_default 实测）。
# env.reset()/update_scene() 都不复位 C12C，而随机化是在"当前"qpos 上加 delta，
# 不先复位会导致 C12C 逐集累加漂移、漂出可抓取区→偶发抓取失败。故每集先复位到此。
C12C_DEFAULT_QPOS = [15.262639999, -0.300000012, 1.075114727,
                     6.123233995736766e-17, 0.0, 0.0, 1.0]


def _make_grasp_device(mp, env, gidx, **kwargs):
    """构造带抓取成功判据的 ScriptedTrajectoryDevice 子类实例。

    判据：某一帧两侧 pad 同时接触 C12C，且 C12C 被抬离初始高度 > GRASP_LIFT_THRESHOLD，
    才算真抓取（滤除闭合瞬间的擦碰误判）。一旦确认即不可撤销。
    """
    class _Device(mp.ScriptedTrajectoryDevice):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._env = env
            self._gidx = gidx
            self._init_z = None       # C12C 初始高度
            self._peak_lift = 0.0     # 记录峰值抬升用于日志
            self.grasp_confirmed = False

        def update(self):
            super().update()
            z = _c12c_world_z(self._env, self._gidx)
            if self._init_z is None:
                self._init_z = z
            lift = z - self._init_z
            if lift > self._peak_lift:
                self._peak_lift = lift
            if not self.grasp_confirmed and lift > GRASP_LIFT_THRESHOLD \
                    and _both_pads_contact_c12c(self._env, self._gidx):
                self.grasp_confirmed = True
                print(f"[GRASP] ✓ 双侧 pad 夹住 C12C 且抬升 {lift*1000:.0f}mm，抓取确认")

    return _Device(**kwargs)


def _setup_symlink(dataset_path, level, no_record):
    """把脚本内置输出路径用软链接重定向到 dataset_path，返回 (symlink_path, created)。"""
    if not dataset_path or no_record:
        return None, False

    target = os.path.abspath(os.path.expanduser(dataset_path))
    os.makedirs(target, exist_ok=True)
    inner_dir = os.path.join(base_dir, "dataset", "humanoid_industrial_robot_1")
    os.makedirs(inner_dir, exist_ok=True)
    symlink_path = os.path.join(inner_dir, level)

    if os.path.islink(symlink_path):
        os.unlink(symlink_path)
    elif os.path.exists(symlink_path):
        print(f"[WARN] {symlink_path} 已存在且不是软链接，数据将保存到原路径而非 {target}")
        return None, False

    os.symlink(target, symlink_path)
    print(f"[INFO] 数据将保存到: {target}")
    return symlink_path, True


def _print_episode_init(env, agent_conf):
    """打印当前 episode 初始化时 C12C、纸箱、机械臂末端的世界坐标。"""
    import numpy as np
    import data_collection_mp as mp

    np.set_printoptions(precision=4, suppress=True)
    mj_data = env.gym._mjData
    mj_model = env.gym._mjModel

    sep = "=" * 60
    print(f"\n{sep}")
    print("  EPISODE INIT  世界坐标")
    print(sep)

    for label, name_hint in [("C12C    ", "C12C_3c_c12c"),
                              ("纸箱    ", "Cardboardbox_01_cardboardbox_01")]:
        try:
            body_name = mp._resolve_body_name(env, name_hint)
            bid = mj_model.body(body_name).id
            pos = mj_data.xpos[bid].copy()
            print(f"  {label} 世界坐标: {pos}")
        except Exception as e:
            print(f"  {label} 查询失败: {e}")

    for label, site_conf_key in [("左臂末端", "l_arm"), ("右臂末端", "r_arm")]:
        try:
            site_name = agent_conf.__dict__[site_conf_key]["ee_site_name"]
            # 找带前缀的完整 site 名
            for i in range(mj_model.nsite):
                sname = mj_model.site(i).name
                if site_name in sname:
                    pos = mj_data.site_xpos[i].copy()
                    print(f"  {label} 世界坐标: {pos}")
                    break
        except Exception as e:
            print(f"  {label} 查询失败: {e}")

    print(sep + "\n")


def main():
    # 强制 stdout 行缓冲，否则 conda run 会块缓冲我们的 print，导致 [✓]/[✗] 等日志不实时显示
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print(">>> run_collection 启动，正在初始化环境（加载模型约需数秒）...", flush=True)

    parser = argparse.ArgumentParser(
        description="data_collection_mp.py 的包装脚本，支持自定义数据集保存路径。"
    )
    parser.add_argument("--pose_file", type=str, default="pose_mp_no_barcode.yaml")
    parser.add_argument("--rand_file", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="数据集保存根目录，默认使用脚本内置路径")
    parser.add_argument("--level", type=str, default=DEFAULT_LEVEL)
    parser.add_argument("--task_config", type=str, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--no_record", action="store_true", help="不保存数据（仅运行查看效果）")
    args = parser.parse_args()

    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    import data_collection_mp as mp
    from conf import d12_conf as agent_conf

    symlink_path, symlink_created = _setup_symlink(args.dataset_path, args.level, args.no_record)

    # 构造 inner_args（与 data_collection_mp.main 中的 args 结构一致）
    inner_args = argparse.Namespace(
        level=args.level,
        task_config=args.task_config,
        pose_file=args.pose_file,
        rand_file=args.rand_file,
        record_data=not args.no_record,
        episodes=args.episodes,
        steps=None, delta_b=None,
        l_target_b=None, r_target_b=None,
        l_quat_b=None, r_quat_b=None,
        gripper_open=None, gripper_close=None,
        dump_pose=None, resolve_pose_only=False,
    )

    spec = mp.load_pose_spec_from_file(os.path.join(base_dir, args.pose_file)) if args.pose_file else {}
    rand_spec = mp.load_yaml_dict(os.path.join(base_dir, args.rand_file)) if args.rand_file else {}
    steps, g_open, g_close = mp._resolve_trajectory_args(inner_args, spec)

    manager, env, l_arm, r_arm, l_grip, r_grip, task_status = \
        mp.create_manager_and_controllers(inner_args, agent_conf)

    # 构建接触索引（geom/body id 不变，只需建一次）
    gidx = _build_grasp_index(env, mp)

    print(f">>> 初始化完成，开始采集，目标成功集数 = {args.episodes}", flush=True)

    try:
        n_success = 0
        episode_index = 0   # 用于 rand seed，失败时递增让下次随机不同
        target_episodes = args.episodes

        # 管理器在 __init__ 注册了 SIGINT 处理器，Ctrl-C 会把 _shutdown_requested 置 True，
        # 之后每次 run_episode 都会立刻空返回。run() 会复位该标志，但我们绕过了 run()，
        # 故这里先复位一次，并在循环中检查它，确保 Ctrl-C 能真正干净停止。
        manager._shutdown_requested = False

        while n_success < target_episodes:
            if manager._shutdown_requested:
                print("\n[STOP] 收到中断信号，停止采集。")
                break
            attempt_label = f"Episode {n_success + 1}/{target_episodes} (seed={episode_index})"
            print(f"\n=== {attempt_label} ===")

            env.reset()
            if not manager.update_scene():
                print("update_scene failed, exit")
                break

            # C12C 先复位到规范默认，再加随机 delta（消除逐集漂移）
            try:
                c12c_body = mp._resolve_body_name(env, "C12C_3c_c12c")
                c12c_joint = mp._find_free_joint_for_body(env, c12c_body)
                env.set_joint_qpos({c12c_joint: np.array(C12C_DEFAULT_QPOS, dtype=np.float64)})
                env.mj_forward()
            except Exception as e:
                print(f"[WARN] C12C 复位默认位失败: {e}")

            episode_rand_spec = mp.advance_rand_spec_seed(rand_spec, episode_index)
            if episode_rand_spec:
                mp.apply_object_randomization(env, episode_rand_spec)

            # 记录本集 C12C 实际位姿，使 scene_info/task_info 非空（对齐 raw_3，支持回放复原）。
            # C12C 是关卡内置静态物体，不在 scene_manager 的 spawnable actor 列表里，
            # 故 scene_manager.get_scene_info() 恒为空，这里直接查其 free-joint qpos 自行构造。
            episode_scene_info, episode_task_info = {}, {}
            try:
                _c12c_body = mp._resolve_body_name(env, "C12C_3c_c12c")
                _c12c_joint = mp._find_free_joint_for_body(env, _c12c_body)
                _c12c_qpos = env.query_joint_qpos([_c12c_joint])[_c12c_joint]
                _actor_info = {
                    "joint_name": _c12c_joint,
                    "joint_qpos": [float(v) for v in np.asarray(_c12c_qpos).ravel()],
                }
                episode_scene_info = {"C12C": _actor_info}
                episode_task_info = {
                    "target_actor": "C12C",
                    "target_actor_info": _actor_info,
                    "goal_name": "Cardboardbox_01",
                    "goal_site": None,
                }
            except Exception as e:
                print(f"[WARN] 记录 C12C scene_info 失败: {e}")

            _print_episode_init(env, agent_conf)

            resolved_spec = mp.resolve_pose_spec_for_current_scene(env, agent_conf, spec)
            l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = mp.build_trajectory_from_resolved_spec(
                env, agent_conf, inner_args, resolved_spec, g_open, g_close, steps
            )

            # 使用带抓取判据的 device 替换原始 ScriptedTrajectoryDevice
            device = _make_grasp_device(
                mp, env, gidx,
                l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                task_status=task_status,
                l_pos=l_pos, l_quat_xyzw=l_quat,
                r_pos=r_pos, r_quat_xyzw=r_quat,
                l_grip_motor=l_gm, r_grip_motor=r_gm,
            )
            manager.set_device(device)
            manager.run_episode()  # EmptyTask 恒 True，忽略返回值

            # 若在 run_episode 中途收到 Ctrl-C，device 没跑完，本集作废且退出
            if manager._shutdown_requested:
                if not args.no_record and manager.data_storage is not None:
                    manager.data_storage.clear_data()
                    if 'time_step' not in manager.data_storage.data:
                        manager.data_storage.data['time_step'] = []
                print("\n[STOP] 收到中断信号，丢弃当前集并停止采集。")
                break

            # 用物理判据判定真实成功
            grasp_ok = device.grasp_confirmed
            episode_index += 1  # 无论成败 seed 都前进，保证重试时随机化不同

            if grasp_ok:
                n_success += 1
                if not args.no_record and manager.data_storage is not None:
                    # scene_info/task_info 用本集自建的 C12C 信息（manager 用 EmptyTask，
                    # 其 get_task_info()/scene_manager.get_scene_info() 恒为空）。
                    manager.data_storage.save_data(
                        task_info=episode_task_info or manager.task.get_task_info(),
                        scene_info=episode_scene_info or manager.scene_manager.get_scene_info(),
                        task_description=manager.task.get_task_description(),
                        extra_hdf5_data=manager.get_device_record_data(),
                    )
                    print(f"[✓] {attempt_label} 抓取成功，数据已保存 ({n_success}/{target_episodes})")
                else:
                    print(f"[✓] {attempt_label} 抓取成功 ({n_success}/{target_episodes})")
            else:
                if not args.no_record and manager.data_storage is not None:
                    manager.data_storage.clear_data()
                    # clear_data() 把 data={} 清空，D12DataStorage 的 time_step 键也丢了，
                    # 下一集 collection_data 会 KeyError；此处手动补回初始键。
                    if not hasattr(manager.data_storage, 'data') or 'time_step' not in manager.data_storage.data:
                        manager.data_storage.data['time_step'] = []
                peak_mm = getattr(device, "_peak_lift", 0.0) * 1000
                print(f"[✗] {attempt_label} 抓取失败（峰值抬升仅 {peak_mm:.0f}mm，未达 {GRASP_LIFT_THRESHOLD*1000:.0f}mm），数据已丢弃，换 seed 重试")

        print(f"\n全部采集完成: 成功 {n_success}/{target_episodes}")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, stopping.")
    except Exception as e:
        print(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        env.close()
        if symlink_created and symlink_path and os.path.islink(symlink_path):
            os.unlink(symlink_path)
            print(f"[INFO] 已清理软链接: {symlink_path}")


if __name__ == "__main__":
    main()
