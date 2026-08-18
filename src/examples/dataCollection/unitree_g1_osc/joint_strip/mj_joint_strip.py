"""编译期剥离非右臂自由度：删 joint，保留 body / geom / camera 树。

背景：Python 侧的位姿注入（pin_floating_base / JointHoldController）只改
mjData，不改 nq/nv/ngeom，所以 mj_step 仍对全身收费。要真正让下肢、腰、左臂
退出计算，只能在模型编译前把它们的 <joint> 删掉——body 留下就成了焊在父刚体上
的静态件，相机链路不受影响。

nq 变短后 OrcaStudio 的 UpdateLocalEnv 会映射错位，所以本模块同时在
gym.update_local_env 上挂一层 qpos 补全：把剥离模型的 qpos 按关节名写回完整
长度的模板数组再推给 Studio。

安全红线：只处理 agent_name 前缀的关节。场景里的工具/工具箱/按钮各自带
freejoint，删掉它们会让抓取目标失去自由度，任务直接报废。

独立自检：
    python mj_joint_strip.py --self_test
    python mj_joint_strip.py --probe_live --orcagym_addr localhost:50051
"""

from __future__ import annotations

import os
import re
import time

import numpy as np

# 右臂 7 关节 + 右夹爪 8 关节的名字片段；命中即保留
KEEP_RIGHT_ARM = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
KEEP_RIGHT_GRIP = ("gripper_r_inner_joint", "gripper_r_outer_joint")
KEEP_DEFAULT = KEEP_RIGHT_ARM + KEEP_RIGHT_GRIP

_JNT_NQ = {0: 7, 1: 4, 2: 1, 3: 1}
_JNT_NV = {0: 6, 1: 3, 2: 1, 3: 1}
_JNT_KIND = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

_TAG_RE = re.compile(r"<(/?)([A-Za-z_][\w:.-]*)((?:\s+[\w:.-]+\s*=\s*\"[^\"]*\")*)\s*(/?)>")
_ATTR_RE = re.compile(r"([\w:.-]+)\s*=\s*\"([^\"]*)\"")
_JOINT_DEF_TAGS = {"joint", "freejoint"}
_ACTUATOR_TAGS = {
    "general", "motor", "position", "velocity", "intvelocity",
    "damper", "cylinder", "muscle", "adhesion",
}


def _attrs(s: str) -> dict:
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(s or "")}


def apply_named_qpos(mj, md, bake_qpos: dict) -> int:
    """按关节名写入 qpos 并 mj_forward。返回成功写入的关节数。"""
    import mujoco
    if not bake_qpos:
        return 0
    n = 0
    for name, val in bake_qpos.items():
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        adr = int(mj.jnt_qposadr[jid])
        w = _JNT_NQ[int(mj.jnt_type[jid])]
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.size == w:
            md.qpos[adr:adr + w] = arr
        elif w == 1:
            md.qpos[adr] = float(arr[0])
        else:
            continue
        n += 1
    if n:
        mujoco.mj_forward(mj, md)
    return n


def _body_rel_pose(md, bid: int, pid: int):
    """子 body 相对父 body 的 pos/quat（MuJoCo wxyz）。"""
    import mujoco
    dp = np.asarray(md.xpos[bid] - md.xpos[pid], dtype=np.float64)
    qn = np.zeros(4)
    mujoco.mju_negQuat(qn, md.xquat[pid])
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, dp, qn)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, qn, md.xquat[bid])
    nrm = np.linalg.norm(quat)
    if nrm > 0:
        quat = quat / nrm
    return pos, quat


_BODY_OPEN_RE = re.compile(
    r"(<body\b)((?:\s+[\w:.-]+\s*=\s*\"[^\"]*\")*)(\s*/?>)",
)


def bake_dropped_bodies(xml: str, mj, md, drop: set) -> tuple[str, list[str]]:
    """把已施加的 qpos 烘进被删关节所属 body 的 pos/quat，再删关节才不会掉回默认下垂。"""
    import mujoco
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in drop:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        bid = int(mj.jnt_bodyid[jid])
        bname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname:
            continue
        pid = int(mj.body_parentid[bid])
        poses[bname] = _body_rel_pose(md, bid, pid)

    if not poses:
        return xml, []

    def _repl(m):
        a = _attrs(m.group(2))
        bname = a.get("name")
        if bname not in poses:
            return m.group(0)
        pos, quat = poses[bname]
        a["pos"] = f"{pos[0]:.8g} {pos[1]:.8g} {pos[2]:.8g}"
        a["quat"] = f"{quat[0]:.8g} {quat[1]:.8g} {quat[2]:.8g} {quat[3]:.8g}"
        for k in ("euler", "axisangle", "xyaxes", "zaxis"):
            a.pop(k, None)
        parts = [m.group(1)]
        order = ["name", "pos", "quat"] + [k for k in a if k not in ("name", "pos", "quat")]
        for k in order:
            parts.append(f' {k}="{a[k]}"')
        parts.append(m.group(3))
        return "".join(parts)

    return _BODY_OPEN_RE.sub(_repl, xml), sorted(poses)


# ---------------------------------------------------------------------------
# 一、决定删哪些关节
# ---------------------------------------------------------------------------
def plan_strip(joint_names, agent_name: str, keep=KEEP_DEFAULT, keep_base: bool = False):
    """返回 (drop:set, keep:list, foreign:list)。foreign 是非本 agent 的关节，一律不动。"""
    prefix = f"{agent_name}_"
    drop, kept, foreign = set(), [], []
    for n in joint_names:
        if not n.startswith(prefix):
            foreign.append(n)
            continue
        tail = n[len(prefix):]
        if any(k in tail for k in keep):
            kept.append(n)
        elif keep_base and "floating_base_joint" in tail:
            kept.append(n)
        else:
            drop.add(n)
    return drop, kept, foreign


# ---------------------------------------------------------------------------
# 二、改 XML：删 joint 定义 + 所有引用了它的 actuator / equality / sensor / tendon
# ---------------------------------------------------------------------------
def strip_xml(xml: str, drop: set):
    """删除 drop 集合里的 joint 定义及其全部依赖元素。body/geom/camera/site 一概保留。"""
    report = {"joint": [], "actuator": [], "equality": [], "sensor": [],
              "tendon": [], "other_ref": []}
    if not drop:
        return xml, report

    out, pos = [], 0
    for m in _TAG_RE.finditer(xml):
        tag = m.group(2)
        a = _attrs(m.group(3))
        closing, selfclose = m.group(1) == "/", m.group(4) == "/"
        if closing:
            continue

        refs = [a[k] for k in ("joint", "joint1", "joint2") if k in a]
        hit_ref = [r for r in refs if r in drop]
        is_def = tag in _JOINT_DEF_TAGS and not refs
        hit_def = is_def and a.get("name") in drop

        if not (hit_def or hit_ref):
            continue

        if hit_def:
            report["joint"].append(a.get("name", "?"))
        elif tag in _ACTUATOR_TAGS:
            report["actuator"].append(a.get("name") or hit_ref[0])
        elif tag in ("weld", "connect", "flex", "distance") or "joint1" in a:
            report["equality"].append(a.get("name") or "|".join(refs))
        elif tag.startswith("joint") or tag in ("actuatorpos", "actuatorvel",
                                                "actuatorfrc", "jointpos",
                                                "jointvel", "jointactuatorfrc",
                                                "jointlimitpos", "jointlimitvel",
                                                "jointlimitfrc"):
            report["sensor"].append(tag)
        elif tag == "joint":
            report["tendon"].append(hit_ref[0])
        else:
            report["other_ref"].append(f"{tag}:{a.get('name') or hit_ref[0]}")

        # 删除整个元素：自闭合直接切掉；成对标签连内容一起切掉
        if selfclose:
            end = m.end()
        else:
            close = re.search(rf"</\s*{re.escape(tag)}\s*>", xml[m.end():])
            end = m.end() + close.end() if close else m.end()
        out.append(xml[pos:m.start()])
        pos = end

    out.append(xml[pos:])
    return "".join(out), report


# ---------------------------------------------------------------------------
# 三、qpos 补全桥：剥离模型 qpos → 完整长度 qpos
# ---------------------------------------------------------------------------
class QposBridge:
    """把剥离模型的 qpos 按关节名写回完整长度数组，供 OrcaStudio 渲染。"""

    def __init__(self, mj_full, md_full, mj_str):
        import mujoco
        self.nq_full = int(mj_full.nq)
        self.nq_str = int(mj_str.nq)
        self.template = np.asarray(md_full.qpos, dtype=np.float64).copy()
        self.pairs: list[tuple[int, int, int]] = []
        self.missing: list[str] = []
        for j in range(int(mj_str.njnt)):
            name = mujoco.mj_id2name(mj_str, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            jf = mujoco.mj_name2id(mj_full, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jf < 0:
                self.missing.append(name)
                continue
            w = _JNT_NQ[int(mj_str.jnt_type[j])]
            self.pairs.append((int(mj_str.jnt_qposadr[j]), int(mj_full.jnt_qposadr[jf]), w))
        self.covered = sum(w for _, _, w in self.pairs)

    def pad(self, qpos):
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if q.size == self.nq_full:
            return q
        if q.size != self.nq_str:
            return q
        out = self.template.copy()
        for s, f, w in self.pairs:
            out[f:f + w] = q[s:s + w]
        return out


# ---------------------------------------------------------------------------
# 四、安全检查
# ---------------------------------------------------------------------------
def safety_check(mj_full, mj_str, bridge, *, required_cameras=(), agent_name="",
                 keep_joints=(), foreign_joints=()):
    """返回 (ok, lines)。任何一条硬红线不过就拒绝剥离，调用方回退原始 XML。"""
    import mujoco
    L, fatal = [], []

    def _nm(m, obj, i):
        return mujoco.mj_id2name(m, obj, i) or ""

    L.append(f"  规模: nq {mj_full.nq}→{mj_str.nq}  nv {mj_full.nv}→{mj_str.nv}  "
             f"nu {mj_full.nu}→{mj_str.nu}  njnt {mj_full.njnt}→{mj_str.njnt}")
    L.append(f"        nbody {mj_full.nbody}→{mj_str.nbody}  "
             f"ngeom {mj_full.ngeom}→{mj_str.ngeom}  ncam {mj_full.ncam}→{mj_str.ncam}  "
             f"nsite {mj_full.nsite}→{mj_str.nsite}")

    # 红线 1：body / geom / camera / site 一个都不能少
    for label, n_full, n_str in (
        ("body", mj_full.nbody, mj_str.nbody),
        ("geom", mj_full.ngeom, mj_str.ngeom),
        ("camera", mj_full.ncam, mj_str.ncam),
        ("site", mj_full.nsite, mj_str.nsite),
    ):
        if n_str < n_full:
            fatal.append(f"{label} 数量减少 {n_full}→{n_str}（应保持不变）")

    # 红线 2：完整模型里有的相机，剥离后必须还在。
    # OrcaStudio 关卡的相机通常不在 MuJoCo XML 里（ncam=0），由 Studio 侧渲染，
    # 这种情况无需检查——只要 body 树没动，相机挂点就还在。
    cams_f = {_nm(mj_full, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(int(mj_full.ncam))}
    cams_s = {_nm(mj_str, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(int(mj_str.ncam))}
    if not cams_f:
        L.append("  XML 内无 MuJoCo 相机（由 OrcaStudio 渲染），跳过相机检查")
    else:
        for want in required_cameras:
            if not [c for c in cams_f if want in c]:
                continue
            hit = [c for c in cams_s if want in c]
            if hit:
                L.append(f"  相机 '{want}' ✓ ({hit[0]})")
            else:
                fatal.append(f"相机 '{want}' 丢失，采集会拿不到画面")

    # 红线 3：保留的关节及其执行器必须完整
    sj = {_nm(mj_str, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(int(mj_str.njnt))}
    lost = [j for j in keep_joints if j not in sj]
    if lost:
        fatal.append(f"应保留的关节被误删 {len(lost)} 个: {lost[:5]}")
    else:
        L.append(f"  应保留关节 {len(keep_joints)} 个 全部在位 ✓")

    n_act = 0
    for i in range(int(mj_str.nu)):
        trn = int(mj_str.actuator_trnid[i, 0])
        if int(mj_str.actuator_trntype[i]) == mujoco.mjtTrn.mjTRN_JOINT and trn >= 0:
            if _nm(mj_str, mujoco.mjtObj.mjOBJ_JOINT, trn) in keep_joints:
                n_act += 1
    L.append(f"  指向保留关节的执行器 {n_act} 个")
    if n_act == 0:
        fatal.append("剥离后没有任何执行器驱动保留关节，右臂将无法控制")

    # 红线 4：场景物体关节一个都不能动（工具是抓取目标）
    lost_f = [j for j in foreign_joints if j not in sj]
    if lost_f:
        fatal.append(f"场景物体关节被误删 {len(lost_f)} 个: {lost_f[:5]}")
    else:
        L.append(f"  场景物体关节 {len(foreign_joints)} 个 全部在位 ✓")

    # 红线 5：qpos 桥必须覆盖剥离模型全部 qpos
    if bridge.missing:
        fatal.append(f"qpos 桥有 {len(bridge.missing)} 个关节在完整模型里找不到: {bridge.missing[:5]}")
    if bridge.covered != bridge.nq_str:
        fatal.append(f"qpos 桥覆盖 {bridge.covered}/{bridge.nq_str}，映射不完整")
    else:
        L.append(f"  qpos 桥 {bridge.nq_str}→{bridge.nq_full} 覆盖完整 ✓")

    # 往返一致性：随机 qpos 补全后，保留关节位置应逐位相等
    rng = np.random.default_rng(0)
    probe = rng.normal(size=bridge.nq_str)
    padded = bridge.pad(probe)
    if padded.size != bridge.nq_full:
        fatal.append(f"补全后长度 {padded.size} != {bridge.nq_full}")
    else:
        err = max((abs(padded[f:f + w] - probe[s:s + w]).max()
                   for s, f, w in bridge.pairs), default=0.0)
        if err > 1e-12:
            fatal.append(f"补全往返误差 {err:.3e}，映射错位")
        else:
            L.append("  补全往返逐位一致 ✓")

    if fatal:
        L.append("  ✗ 安全检查未通过:")
        L.extend(f"      - {f}" for f in fatal)
    else:
        L.append("  ✓ 全部安全检查通过")
    return (not fatal), L


# ---------------------------------------------------------------------------
# 五、关掉被剥离部件的碰撞（可选；接触是 nefc 的主要来源）
# ---------------------------------------------------------------------------
def kill_stripped_collision(mj, md, agent_name: str, keep=KEEP_DEFAULT):
    """把不属于保留链路的 agent geom 的 contype/conaffinity 清零。"""
    import mujoco
    prefix = f"{agent_name}_"
    keep_tokens = tuple(k.replace("_joint", "").replace("joint", "") for k in keep)
    keep_tokens = tuple(t for t in keep_tokens if t) + (
        "right_shoulder", "right_elbow", "right_wrist", "gripper_r", "arm_r_end",
        "camera", "head",
    )
    n = 0
    for g in range(int(mj.ngeom)):
        bn = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY,
                               int(mj.geom_bodyid[g])) or ""
        if not bn.startswith(prefix):
            continue
        tail = bn[len(prefix):].lower()
        if any(t.lower() in tail for t in keep_tokens):
            continue
        if int(mj.geom_contype[g]) or int(mj.geom_conaffinity[g]):
            mj.geom_contype[g] = 0
            mj.geom_conaffinity[g] = 0
            n += 1
    if n:
        mujoco.mj_forward(mj, md)
    return n


# ---------------------------------------------------------------------------
# 六、接线：包 load_model_xml + update_local_env
# ---------------------------------------------------------------------------
class StripHandle:
    def __init__(self):
        self.bridge: QposBridge | None = None
        self.applied = False
        self.report: dict = {}
        self.patched_path: str | None = None
        self.n_col_off = 0
        self.class_level = False
        self.gym_inst = None
        self._want_col_off = False
        self._orig_load = None
        self._orig_ule = None
        self._gym = None

    def restore(self):
        if self._gym is None:
            return
        if self._orig_load is not None:
            self._gym.load_model_xml = self._orig_load
        if self._orig_ule is not None:
            self._gym.update_local_env = self._orig_ule


def install(env, agent_name: str, *, keep=KEEP_DEFAULT, keep_base: bool = False,
            kill_collision: bool = True, required_cameras=("cam_head", "wrist_r"),
            bake_qpos: dict | None = None,
            dump_dir: str = "/tmp/g1_joint_strip", log=print) -> StripHandle:
    """挂剥离补丁。

    env=None 时打在 OrcaGymLocal 类上，必须在 DataCollectionManager 创建之前调用
    —— XML 由 scene_manager 的 init_env 回调触发加载，时机早于脚本里的 env.reset()，
    实例级补丁可能赶不上。传入 env 则只打在该实例上。

    安全检查不通过时自动回退原始 XML，采集照常进行，只是没有剥离效果。
    """
    import mujoco

    h = StripHandle()
    if env is None:
        from orca_gym.core.orca_gym_local import OrcaGymLocal
        gym = OrcaGymLocal
        h.class_level = True
    else:
        gym = getattr(env, "gym", None) or getattr(
            getattr(env, "unwrapped", env), "gym", None)
        if gym is None:
            log("[STRIP] 拿不到 env.gym，剥离未启用")
            return h
    h._gym = gym
    os.makedirs(dump_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    h._orig_load = gym.load_model_xml

    async def _patched_load_model_xml(_self=None):
        orig_path = (await h._orig_load(_self)) if h.class_level \
            else (await h._orig_load())
        if h.class_level and _self is not None:
            h.gym_inst = _self
        try:
            with open(orig_path, "r") as f:
                xml = f.read()

            mj_full = mujoco.MjModel.from_xml_path(orig_path)
            md_full = mujoco.MjData(mj_full)
            mujoco.mj_forward(mj_full, md_full)

            all_j = [mujoco.mj_id2name(mj_full, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                     for i in range(int(mj_full.njnt))]
            drop, kept, foreign = plan_strip(all_j, agent_name, keep, keep_base)
            if not drop:
                log(f"[STRIP] 没有匹配到可删关节（agent={agent_name}），保持原样")
                return orig_path

            n_bake = apply_named_qpos(mj_full, md_full, bake_qpos or {})
            xml, baked_bodies = bake_dropped_bodies(xml, mj_full, md_full, drop)
            if n_bake or baked_bodies:
                log(f"[STRIP] 已烘入 {n_bake} 个关节角 → {len(baked_bodies)} 个 body "
                    f"（左臂平举等静态姿态）")

            new_xml, rep = strip_xml(xml, drop)
            import pathlib
            orig = pathlib.Path(orig_path)
            patched = str(orig.with_stem(orig.stem + "_jointstrip"))
            with open(patched, "w") as f:
                f.write(new_xml)

            mj_str = mujoco.MjModel.from_xml_path(patched)
            bridge = QposBridge(mj_full, md_full, mj_str)
            ok, lines = safety_check(
                mj_full, mj_str, bridge,
                required_cameras=required_cameras, agent_name=agent_name,
                keep_joints=kept, foreign_joints=foreign,
            )

            head = [
                f"[STRIP] 剥离计划 agent={agent_name}",
                f"  删除关节 {len(rep['joint'])} 个，保留 {len(kept)} 个，"
                f"场景关节 {len(foreign)} 个不动",
                f"  连带删除: actuator={len(rep['actuator'])} "
                f"equality={len(rep['equality'])} sensor={len(rep['sensor'])} "
                f"tendon={len(rep['tendon'])} other={len(rep['other_ref'])}",
            ]
            body = head + lines
            for ln in body:
                log(ln)
            with open(os.path.join(dump_dir, f"strip_report_{ts}.txt"), "w") as f:
                f.write("\n".join(body) + "\n\n删除的关节:\n")
                f.write("\n".join(f"  {j}" for j in sorted(rep["joint"])))
                f.write("\n\n保留的关节:\n")
                f.write("\n".join(f"  {j}" for j in kept))
                f.write("\n\n连带删除的执行器:\n")
                f.write("\n".join(f"  {a}" for a in rep["actuator"]) + "\n")

            if not ok:
                log("[STRIP] ✗ 安全检查未通过 → 回退原始 XML，本次不剥离")
                return orig_path

            h.bridge = bridge
            h.applied = True
            h.report = rep
            h.patched_path = patched
            log(f"[STRIP] ✓ 已启用 → {pathlib.Path(patched).name}")
            return patched
        except Exception as exc:
            log(f"[STRIP] 异常({type(exc).__name__}: {exc}) → 回退原始 XML")
            return orig_path

    gym.load_model_xml = _patched_load_model_xml

    h._orig_ule = getattr(gym, "update_local_env", None)
    if h._orig_ule is not None:
        if h.class_level:
            async def _padded_update_local_env(_self, qpos, time_):
                if h.bridge is not None:
                    qpos = h.bridge.pad(qpos)
                return await h._orig_ule(_self, qpos, time_)
        else:
            async def _padded_update_local_env(qpos, time_):
                if h.bridge is not None:
                    qpos = h.bridge.pad(qpos)
                return await h._orig_ule(qpos, time_)

        gym.update_local_env = _padded_update_local_env

    h._want_col_off = bool(kill_collision)
    return h


def finish_install(env, handle: StripHandle, agent_name: str,
                   keep=KEEP_DEFAULT, log=print) -> None:
    """模型加载后调用（env.reset() 之后），关掉被剥离部件的碰撞。"""
    if not handle.applied or not handle._want_col_off:
        return
    gym = handle.gym_inst
    if gym is None:
        gym = getattr(env, "gym", None) or getattr(
            getattr(env, "unwrapped", env), "gym", None)
    mj = getattr(gym, "_mjModel", None)
    md = getattr(gym, "_mjData", None)
    if mj is None or md is None:
        log("[STRIP] 拿不到 _mjModel，跳过碰撞关闭")
        return
    n = kill_stripped_collision(mj, md, agent_name, keep)
    handle.n_col_off = n
    handle._want_col_off = False
    log(f"[STRIP] 已关闭 {n} 个被剥离部件 geom 的碰撞（contype/conaffinity=0）")


# ---------------------------------------------------------------------------
# 自检 / 实机探查
# ---------------------------------------------------------------------------
_TOY = """<mujoco><worldbody>
  <body name="ag_pelvis" pos="0 0 1">
    <freejoint name="ag_floating_base_joint"/>
    <geom name="g_pelvis" size=".1"/>
    <body name="ag_left_knee" pos="0 .1 -.3">
      <joint name="ag_left_knee_joint" axis="0 1 0"/><geom name="g_lk" size=".04"/>
    </body>
    <body name="ag_torso_link_rev_1_0" pos="0 0 .2">
      <joint name="ag_waist_yaw_joint" axis="0 0 1"/><geom name="g_to" size=".08"/>
      <body name="ag_head_camera1" pos="0 0 .3">
        <geom name="g_cam" size=".001"/><camera name="cam_head"/>
      </body>
      <body name="ag_right_shoulder" pos="0 -.2 .1">
        <joint name="ag_right_shoulder_pitch_joint" axis="0 1 0"/>
        <geom name="g_rs" size=".04"/>
        <body name="ag_right_elbow" pos="0 0 -.25">
          <joint name="ag_right_elbow_joint" axis="0 1 0"/><geom name="g_re" size=".03"/>
          <body name="ag_arm_r_end_link" pos="0 0 -.2">
            <geom name="g_wr" size=".02"/><camera name="wrist_r"/>
            <site name="ee_center_site_r"/>
          </body>
        </body>
      </body>
      <body name="ag_left_shoulder" pos="0 .2 .1">
        <joint name="ag_left_shoulder_pitch_joint" axis="0 1 0"/>
        <geom name="g_ls" size=".04"/>
      </body>
    </body>
  </body>
  <body name="Group_Interactive_Screwdriver_task_screwdriver">
    <joint name="Group_Interactive_Screwdriver_task_screwdriver_joint" type="free"/>
    <geom name="g_sd" size=".02"/>
  </body>
</worldbody>
<actuator>
  <position name="ag_left_knee_joint_pctrl" joint="ag_left_knee_joint" kp="100"/>
  <position name="ag_waist_yaw_joint_pctrl" joint="ag_waist_yaw_joint" kp="100"/>
  <general  name="ag_right_shoulder_pitch_joint_pctrl" joint="ag_right_shoulder_pitch_joint" gainprm="100"/>
  <general  name="ag_right_elbow_joint_pctrl" joint="ag_right_elbow_joint" gainprm="100"/>
  <position name="ag_left_shoulder_pitch_joint_pctrl" joint="ag_left_shoulder_pitch_joint" kp="100"/>
</actuator>
<equality>
  <joint name="eq_left" joint1="ag_left_knee_joint" joint2="ag_left_shoulder_pitch_joint"/>
  <weld name="ag_pelvis_weld" body1="ag_pelvis" body2="world"/>
</equality>
<sensor>
  <jointpos name="s_lk" joint="ag_left_knee_joint"/>
  <jointpos name="s_re" joint="ag_right_elbow_joint"/>
</sensor>
</mujoco>"""


def _self_test() -> int:
    import mujoco
    print("=" * 78)
    print("玩具模型自检：删非右臂 joint，验证 body/相机/场景关节/qpos 桥")
    print("=" * 78)
    mj_full = mujoco.MjModel.from_xml_string(_TOY)
    md_full = mujoco.MjData(mj_full)
    mujoco.mj_forward(mj_full, md_full)
    all_j = [mujoco.mj_id2name(mj_full, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(int(mj_full.njnt))]
    drop, kept, foreign = plan_strip(all_j, "ag")
    print(f"\n删除 {len(drop)}: {sorted(drop)}")
    print(f"保留 {len(kept)}: {kept}")
    print(f"场景 {len(foreign)}: {foreign}")

    new_xml, rep = strip_xml(_TOY, drop)
    print(f"\n连带删除: actuator={rep['actuator']}")
    print(f"          equality={rep['equality']} sensor={rep['sensor']}")

    mj_str = mujoco.MjModel.from_xml_string(new_xml)
    bridge = QposBridge(mj_full, md_full, mj_str)
    ok, lines = safety_check(mj_full, mj_str, bridge,
                             required_cameras=("cam_head", "wrist_r"),
                             agent_name="ag", keep_joints=kept,
                             foreign_joints=foreign)
    print()
    for ln in lines:
        print(ln)

    md_str = mujoco.MjData(mj_str)
    for _ in range(50):
        mujoco.mj_step(mj_str, md_str)
    print(f"\n剥离模型步进 50 步正常，qpos={np.round(md_str.qpos, 3)}")
    n = kill_stripped_collision(mj_str, md_str, "ag")
    print(f"关闭碰撞的 geom 数 = {n}")
    print("\n" + ("✓ 自检通过" if ok else "✗ 自检未通过"))
    return 0 if ok else 1


def _probe_live(addr: str, agent_name: str | None) -> int:
    import asyncio
    import grpc
    from orca_gym.protos import mjc_message_pb2, mjc_message_pb2_grpc

    async def go():
        ch = grpc.aio.insecure_channel(addr, options=[
            ("grpc.max_send_message_length", 2 ** 31 - 1),
            ("grpc.max_receive_message_length", 2 ** 31 - 1)])
        stub = mjc_message_pb2_grpc.GrpcServiceStub(ch)
        info = await stub.QueryModelInfo(mjc_message_pb2.QueryModelInfoRequest())
        names = list((await stub.QueryJointNames(
            mjc_message_pb2.QueryJointNamesRequest())).JointNames)
        await ch.close()
        print(f"实机: nq={info.nq} nv={info.nv} nu={info.nu} njnt={info.njnt} "
              f"nbody={info.nbody} ngeom={info.ngeom}")
        ag = agent_name
        if ag is None:
            cands = [n for n in names if "floating_base_joint" in n]
            ag = cands[0].replace("_floating_base_joint", "") if cands else ""
            print(f"自动识别 agent_name = {ag}")
        drop, kept, foreign = plan_strip(names, ag)
        dq = 0
        for n in sorted(drop):
            dq += 7 if "floating_base" in n else 1
        print(f"\n将删除 {len(drop)} 个关节（约 -{dq} qpos）：")
        for n in sorted(drop):
            print("   ", n)
        print(f"\n将保留 {len(kept)} 个：")
        for n in kept:
            print("   ", n)
        print(f"\n场景关节 {len(foreign)} 个保持不动：")
        for n in foreign:
            print("   ", n)
        print(f"\n预估 nq {info.nq} → {info.nq - dq}，"
              f"nv {info.nv} → {info.nv - (dq - 1 if any('floating_base' in n for n in drop) else dq)}")
        return 0

    return asyncio.run(go())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="非右臂自由度剥离工具")
    p.add_argument("--self_test", action="store_true", help="玩具模型离线自检")
    p.add_argument("--probe_live", action="store_true", help="连实机读关节表并预演")
    p.add_argument("--orcagym_addr", default="localhost:50051")
    p.add_argument("--agent_name", default=None)
    args = p.parse_args()
    if args.probe_live:
        raise SystemExit(_probe_live(args.orcagym_addr, args.agent_name))
    raise SystemExit(_self_test())
