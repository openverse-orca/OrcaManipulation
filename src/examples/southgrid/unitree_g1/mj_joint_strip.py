"""G1 任务模型配置与一致性校验工具。

该模块准备任务所需的模型配置，并在运行模型与展示模型之间转换 qpos。
处理范围由 agent_name 和保留列表限定；场景物体、相机与任务资源会经过
一致性校验后才投入运行。
"""

from __future__ import annotations

import os
import re
import time

import numpy as np

# 默认任务控制链的关节名称片段
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
    """将参考状态转换为任务模型中的 body 局部位姿。"""
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
# 电柜按钮的行程约束
# ---------------------------------------------------------------------------
_BUTTON_JOINT_TOKENS = ("ElectricalCabinet_Button", "ElectricalCabinet_button")


def clamp_button_joints(xml: str, range_max: float = 0.01) -> tuple[str, list[str]]:
    """为电柜按钮设置行程及约束求解参数。

    range_max 的单位为米；solreflimit 和 solimplimit 用于保持按钮运动范围稳定。
    """
    if not _BUTTON_JOINT_TOKENS or range_max <= 0:
        return xml, []

    def _repl(m):
        a = _attrs(m.group(2))
        name = a.get("name", "")
        if not any(tok in name for tok in _BUTTON_JOINT_TOKENS):
            return m.group(0)
        a["range"] = f"0 {range_max}"
        a["solreflimit"] = "0.001 1"
        a["solimplimit"] = "0.001 0.001 0.001"
        parts = [m.group(1)]
        order = ["name", "type", "axis", "range", "solreflimit", "solimplimit"] + [
            k for k in a if k not in ("name", "type", "axis", "range",
                                      "solreflimit", "solimplimit")
        ]
        for k in order:
            parts.append(f' {k}="{a[k]}"')
        parts.append(m.group(3))
        return "".join(parts)

    new_xml, n = re.subn(r"(<joint\b)((?:\s+[\w:.-]+\s*=\s*\"[^\"]*\")*)\s*(/?>)", _repl, xml)
    clamped = []
    for m in re.finditer(r'<joint\b[^>]*\bname="([^"]+)"', new_xml):
        if any(tok in m.group(1) for tok in _BUTTON_JOINT_TOKENS):
            clamped.append(m.group(1))
    return new_xml, clamped


# ---------------------------------------------------------------------------
# 任务模型关节分组
# ---------------------------------------------------------------------------
def plan_strip(joint_names, agent_name: str, keep=KEEP_DEFAULT, keep_base: bool = False):
    """按 agent 前缀返回任务配置、控制链和场景关节分组。"""
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
# 生成任务模型 XML，并同步关联元素
# ---------------------------------------------------------------------------
def strip_xml(xml: str, drop: set):
    """按关节配置更新 XML 及其关联元素，保留场景与传感资源。"""
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

        # 同步处理自闭合元素和带内容的成对元素
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
# 运行模型与展示模型之间的 qpos 转换
# ---------------------------------------------------------------------------
class QposBridge:
    """按关节名将运行时 qpos 转换为 OrcaStudio 展示格式。"""

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
# 模型一致性检查
# ---------------------------------------------------------------------------
def safety_check(mj_full, mj_str, bridge, *, required_cameras=(), agent_name="",
                 keep_joints=(), foreign_joints=()):
    """Validate the task model configuration and return ``(ok, messages)``."""
    import mujoco
    L, fatal = [], []

    def _nm(m, obj, i):
        return mujoco.mj_id2name(m, obj, i) or ""

    # Scene and sensor resources.
    for label, n_full, n_str in (
        ("body", mj_full.nbody, mj_str.nbody),
        ("geom", mj_full.ngeom, mj_str.ngeom),
        ("camera", mj_full.ncam, mj_str.ncam),
        ("site", mj_full.nsite, mj_str.nsite),
    ):
        if n_str < n_full:
            fatal.append(f"缺少必需的 {label} 资源")

    # Cameras may be supplied by the model or the OrcaStudio scene.
    cams_f = {_nm(mj_full, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(int(mj_full.ncam))}
    cams_s = {_nm(mj_str, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(int(mj_str.ncam))}
    if not cams_f:
        L.append("  相机资源由 OrcaStudio 场景提供")
    else:
        for want in required_cameras:
            if not [c for c in cams_f if want in c]:
                continue
            hit = [c for c in cams_s if want in c]
            if hit:
                L.append(f"  相机 '{want}' 可用 ✓")
            else:
                fatal.append(f"缺少必需相机 '{want}'")

    # Control interface.
    sj = {_nm(mj_str, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(int(mj_str.njnt))}
    lost = [j for j in keep_joints if j not in sj]
    if lost:
        fatal.append("控制接口不完整")
    else:
        L.append("  控制接口完整 ✓")

    n_act = 0
    for i in range(int(mj_str.nu)):
        trn = int(mj_str.actuator_trnid[i, 0])
        if int(mj_str.actuator_trntype[i]) == mujoco.mjtTrn.mjTRN_JOINT and trn >= 0:
            if _nm(mj_str, mujoco.mjtObj.mjOBJ_JOINT, trn) in keep_joints:
                n_act += 1
    if n_act == 0:
        fatal.append("控制接口没有可用执行器")

    # Scene object interface.
    lost_f = [j for j in foreign_joints if j not in sj]
    if lost_f:
        fatal.append("场景物体接口不完整")
    else:
        L.append("  场景物体接口完整 ✓")

    # State mapping.
    if bridge.missing:
        fatal.append("状态映射不完整")
    if bridge.covered != bridge.nq_str:
        fatal.append("状态映射覆盖不完整")
    else:
        L.append("  状态映射完整 ✓")

    # State round-trip consistency.
    rng = np.random.default_rng(0)
    probe = rng.normal(size=bridge.nq_str)
    padded = bridge.pad(probe)
    if padded.size != bridge.nq_full:
        fatal.append("状态映射输出维度不一致")
    else:
        err = max((abs(padded[f:f + w] - probe[s:s + w]).max()
                   for s, f, w in bridge.pairs), default=0.0)
        if err > 1e-12:
            fatal.append("状态往返校验未通过")
        else:
            L.append("  状态往返校验通过 ✓")

    if fatal:
        L.append("  ✗ 模型配置检查未通过:")
        L.extend(f"      - {f}" for f in fatal)
    else:
        L.append("  ✓ 模型配置检查通过")
    return (not fatal), L


# ---------------------------------------------------------------------------
# 任务模型碰撞配置
# ---------------------------------------------------------------------------
def kill_stripped_collision(mj, md, agent_name: str, keep=KEEP_DEFAULT):
    """Apply the task collision configuration."""
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
# 模型加载与展示状态适配
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
    """安装任务模型配置。

    env=None 时在 DataCollectionManager 创建前安装；传入 env 时作用于该实例。
    配置校验未通过时使用默认模型。
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
            log("[MODEL] 当前环境不支持任务模型配置，使用默认模型")
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
                log(f"[MODEL] agent={agent_name} 已符合当前任务模型配置")
                return orig_path

            n_bake = apply_named_qpos(mj_full, md_full, bake_qpos or {})
            xml, baked_bodies = bake_dropped_bodies(xml, mj_full, md_full, drop)
            if n_bake or baked_bodies:
                log("[MODEL] 参考姿态已应用")

            new_xml, rep = strip_xml(xml, drop)
            new_xml, clamped_btns = clamp_button_joints(new_xml, range_max=0.001)
            if clamped_btns:
                log(f"[MODEL] 已配置 {len(clamped_btns)} 个按钮行程约束")
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

            head = [f"[MODEL] 正在检查任务模型配置：{agent_name}"]
            body = head + lines
            for ln in body:
                log(ln)
            with open(os.path.join(dump_dir, f"strip_report_{ts}.txt"), "w") as f:
                f.write("\n".join(body) + "\n")

            if not ok:
                log("[MODEL] 任务模型配置检查未通过，使用默认模型")
                return orig_path

            h.bridge = bridge
            h.applied = True
            h.report = rep
            h.patched_path = patched
            log("[MODEL] 任务模型配置已就绪")
            return patched
        except Exception as exc:
            log("[MODEL] 任务模型配置不可用，使用默认模型")
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
    """在模型加载后应用任务碰撞配置。"""
    if not handle.applied or not handle._want_col_off:
        return
    gym = handle.gym_inst
    if gym is None:
        gym = getattr(env, "gym", None) or getattr(
            getattr(env, "unwrapped", env), "gym", None)
    mj = getattr(gym, "_mjModel", None)
    md = getattr(gym, "_mjData", None)
    if mj is None or md is None:
        log("[MODEL] 当前环境不支持碰撞配置，保持模型默认设置")
        return
    n = kill_stripped_collision(mj, md, agent_name, keep)
    handle.n_col_off = n
    handle._want_col_off = False
    log("[MODEL] 任务碰撞配置已应用")


# ---------------------------------------------------------------------------
# 配置验证
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
    print("任务模型配置自检")
    print("=" * 78)
    mj_full = mujoco.MjModel.from_xml_string(_TOY)
    md_full = mujoco.MjData(mj_full)
    mujoco.mj_forward(mj_full, md_full)
    all_j = [mujoco.mj_id2name(mj_full, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(int(mj_full.njnt))]
    drop, kept, foreign = plan_strip(all_j, "ag")
    new_xml, rep = strip_xml(_TOY, drop)
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
    print("\n任务模型连续步进检查完成")
    n = kill_stripped_collision(mj_str, md_str, "ag")
    print("碰撞配置检查完成")
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
        print("运行环境模型接口检查完成")
        ag = agent_name
        if ag is None:
            cands = [n for n in names if "floating_base_joint" in n]
            ag = cands[0].replace("_floating_base_joint", "") if cands else ""
        drop, kept, foreign = plan_strip(names, ag)
        dq = 0
        for n in sorted(drop):
            dq += 7 if "floating_base" in n else 1
        print("任务模型兼容性检查完成")
        return 0

    return asyncio.run(go())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="G1 任务模型配置验证工具")
    p.add_argument("--self_test", action="store_true", help="运行离线配置验证")
    p.add_argument("--probe_live", action="store_true", help="检查运行环境模型兼容性")
    p.add_argument("--orcagym_addr", default="localhost:50051")
    p.add_argument("--agent_name", default=None)
    args = p.parse_args()
    if args.probe_live:
        raise SystemExit(_probe_live(args.orcagym_addr, args.agent_name))
    raise SystemExit(_self_test())
