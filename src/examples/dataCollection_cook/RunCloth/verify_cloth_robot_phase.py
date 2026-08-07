#!/usr/bin/env python3
"""ClothRobot 分阶段核验脚本（P0/P1 可用 fixture MJCF；P2 需 mj_forward + 可选 XPBD smoke）。"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[5]
CLOTH_3D = REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cloth_robot_p0_minimal.xml"
ORCA_MANIP = REPO_ROOT / "OrcaManipulation" / "src"
XPBD_BIN = REPO_ROOT / "XPBD" / "build" / "dual_gripper_cross_mjc"
DEFAULT_CONFIG = CLOTH_3D / "cloth_sim_config.test20260508_openloong.json"


def _setup_paths() -> None:
    for p in (str(CLOTH_3D), str(ORCA_MANIP)):
        if p not in sys.path:
            sys.path.insert(0, p)


def verify_p0(model: mujoco.MjModel) -> tuple[bool, list[str]]:
    """P0：MJCF 含 XPBD_CLOTHSHEET / XPBD_TRACK 标记。"""
    msgs: list[str] = []
    ok = True
    cloth_sites = 0
    track_geoms = 0
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "XPBD_CLOTHSHEET" in sname:
            cloth_sites += 1
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "XPBD_TRACK" in gname:
            track_geoms += 1
    msgs.append(f"clothsheet sites: {cloth_sites}")
    msgs.append(f"track geoms: {track_geoms}")
    if cloth_sites < 1:
        ok = False
        msgs.append("FAIL: expected >=1 XPBD_CLOTHSHEET site")
    if track_geoms < 1:
        ok = False
        msgs.append("FAIL: expected >=1 XPBD_TRACK geom")
    if ok:
        msgs.append("PASS: P0 MJCF markers")
    return ok, msgs


def _load_body_map_orcagym():
    path = ORCA_MANIP / "envs" / "cloth" / "body_map_orcagym.py"
    spec = importlib.util.spec_from_file_location("body_map_orcagym", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _deep_merge_config(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if key == "extends":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_config(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_cloth_config(config_path: Path) -> dict:
    """加载 cloth_sim JSON；``extends`` 相对 ``cloth_3d`` 目录递归深合并。"""
    path = config_path.expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    extends = raw.get("extends")
    if not extends:
        return raw
    base_path = (CLOTH_3D / str(extends)).resolve()
    base = load_cloth_config(base_path)
    return _deep_merge_config(base, raw)


def build_xpbd_session_config(base_cfg: dict, adapted_cfg: dict) -> dict:
    """
    构造供 ``MJC_PBD_CONFIG`` 使用的会话 JSON。

    ``xpbd.cloth_discover_only=true`` 时仅将 cloth 发现段交给 XPBD，刚体 map 恢复基配置短链。
    """
    out = copy.deepcopy(adapted_cfg)
    xpbd_blk = out.setdefault("xpbd", {})
    if bool(xpbd_blk.get("cloth_discover_only", True)):
        if "rigid_body_map" in base_cfg:
            out["rigid_body_map"] = copy.deepcopy(base_cfg["rigid_body_map"])
        if "orcalink_rigid_body_map" in base_cfg:
            out["orcalink_rigid_body_map"] = copy.deepcopy(base_cfg["orcalink_rigid_body_map"])
        elif "orcagym_rigid_body_map" in base_cfg:
            out["orcalink_rigid_body_map"] = copy.deepcopy(base_cfg["orcagym_rigid_body_map"])
    return out


def write_xpbd_runtime_session_config(
    config: dict,
    *,
    session_timestamp: str,
    source_config_path: Path | None = None,
    source_mjcf_path: Path | None = None,
) -> Path:
    """将运行时 effective config 写入 ``cloth_3d/cloth_sim_session_{ts}.json``。"""
    session_path = (CLOTH_3D / f"cloth_sim_session_{session_timestamp}.json").resolve()
    payload = copy.deepcopy(config)
    meta: dict[str, str] = {
        "session_timestamp": session_timestamp,
        "source_config": str(source_config_path.resolve()) if source_config_path else "",
    }
    if source_mjcf_path is not None and source_mjcf_path.is_file():
        meta["source_mjcf"] = str(source_mjcf_path.resolve())
    payload["_cloth_robot_session_meta"] = meta
    session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return session_path


def find_latest_studio_mjcf() -> Path | None:
    """
    返回 Studio Play 或 OrcaGym 缓存目录中最新 MJCF（排除 ``out.xml``）。

    优先扫描 ``~/Orca/OrcaStudio/.../tmp``，其次 ``~/.orcagym/tmp``。
    """
    candidates: list[Path] = []
    for d in (
        Path.home() / "Orca/OrcaStudio/{5CA138B2-AC6E-4F4B-90D4-E545B54EB207}/tmp",
        Path.home() / ".orcagym/tmp",
    ):
        if d.is_dir():
            candidates.extend(p for p in d.glob("*.xml") if p.name != "out.xml")
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_p2_session_from_mjcf(
    mjcf_path: Path,
    config_path: Path,
    *,
    session_timestamp: str | None = None,
    level: str | None = None,
) -> tuple[dict, dict, Path]:
    """
    从 Studio MJCF 扫描布片位姿（含 Entity3 旋转）并写出 XPBD session JSON。

    流程：``mj_forward`` → ``adapt_config_for_orcagym``（含 ``enrich_cloth_discovery_pose``）
    → ``build_xpbd_session_config`` → ``cloth_sim_session_*.json``。
    Studio 中旋转布料后须重新 Play 生成 MJCF，再调用本函数刷新 session。

    返回 ``(xpbd_session, adapted_config, session_path)``。
    """
    _setup_paths()
    body_map_orcagym = _load_body_map_orcagym()
    base_cfg = load_cloth_config(config_path)
    if level:
        from envs.cloth.paths import apply_runtime_orcagym_level

        base_cfg = apply_runtime_orcagym_level(base_cfg, level)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    from modules.identify_xpbd_bodies import merge_body_discovery  # noqa: WPS433
    from modules.identify_xpbd_cloth import (  # noqa: WPS433
        enrich_cloth_discovery_pose,
        identify_xpbd_cloth,
        merge_cloth_discovery,
    )

    cloths = enrich_cloth_discovery_pose(model, data, identify_xpbd_cloth(model))
    base_cfg = merge_cloth_discovery(base_cfg, cloths)
    base_cfg = merge_body_discovery(base_cfg, model, data)
    adapted = body_map_orcagym.adapt_config_for_orcagym(model, base_cfg, data=data)
    xpbd_session = build_xpbd_session_config(base_cfg, adapted)
    xpbd_session.setdefault("mujoco", {})["model_path"] = str(mjcf_path.resolve())
    try:
        from envs.cloth.mjcf_tele_layout import (  # noqa: WPS433
            apply_tele_layout_to_session,
            scan_tele_layout_from_model,
        )

        tele = scan_tele_layout_from_model(model)
        xpbd_session = apply_tele_layout_to_session(xpbd_session, tele)
    except (ImportError, RuntimeError, KeyError, ValueError) as exc:
        import logging

        logging.getLogger(__name__).warning("tele_layout scan skipped: %s", exc)
    ts = session_timestamp or datetime.now().strftime("view_%Y%m%d_%H%M%S")
    session_path = write_xpbd_runtime_session_config(
        xpbd_session,
        session_timestamp=ts,
        source_config_path=config_path,
        source_mjcf_path=mjcf_path,
    )
    return xpbd_session, adapted, session_path


def _is_vec3(val: object) -> bool:
    return isinstance(val, (list, tuple)) and len(val) == 3 and all(isinstance(x, (int, float)) for x in val)


def _is_quat_wxyz(val: object) -> bool:
    return isinstance(val, (list, tuple)) and len(val) == 4 and all(isinstance(x, (int, float)) for x in val)


def _rows_logical_equals_mjc(rows: list[dict]) -> tuple[bool, list[str]]:
    """校验每行 ``logical_name == mjc_body_name``。"""
    bad: list[str] = []
    for row in rows:
        mjc = str(row.get("mjc_body_name", ""))
        ln = str(row.get("logical_name", ""))
        if mjc and ln != mjc:
            bad.append(f"{mjc} -> logical_name={ln!r}")
    return (len(bad) == 0, bad)


def verify_p2(
    mjcf_path: Path,
    config_path: Path | None,
) -> tuple[bool, list[str], Path | None]:
    """
    P2.1+P2.2：adapt → build_xpbd_session_config → 写 session JSON；
    校验 cloth.discovered 位姿/VTK，且 cloth_discover_only 时不向 XPBD 传递扫描的 N 刚体。
    """
    msgs: list[str] = []
    ok = True
    session_path: Path | None = None
    adapted: dict = {}

    if config_path and config_path.is_file():
        cfg_label = str(config_path)
    else:
        cfg_label = "<inline minimal>"
    msgs.append(f"config: {cfg_label}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config_path and config_path.is_file():
        xpbd_session, adapted, session_path = build_p2_session_from_mjcf(
            mjcf_path,
            config_path,
            session_timestamp=f"verify_p2_{ts}",
        )
    else:
        base_cfg = {
            "xpbd_auto_discover": {"bodies": True, "cloth": True},
            "orcagym": {"rigid_body_map_key": "orcagym_rigid_body_map"},
            "orcagym_rigid_body_map": [],
            "cloth": {"mesh": "shirt_v4.vtk"},
            "xpbd": {"cloth_discover_only": True},
        }
        body_map_orcagym = _load_body_map_orcagym()
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        adapted = body_map_orcagym.adapt_config_for_orcagym(model, base_cfg, data=data)
        xpbd_session = build_xpbd_session_config(base_cfg, adapted)
        session_path = write_xpbd_runtime_session_config(
            xpbd_session,
            session_timestamp=f"verify_p2_{ts}",
            source_mjcf_path=mjcf_path,
        )
    msgs.append(f"session: {session_path}")

    cloth = xpbd_session.get("cloth") or {}
    adapted_publish = adapted.get("orcalink_rigid_body_map") or []
    session_publish = xpbd_session.get("orcalink_rigid_body_map") or []

    msgs.append(f"adapted publish bodies: {len(adapted_publish)}")
    msgs.append(f"session publish bodies: {len(session_publish)}")

    if not cloth.get("discovered"):
        ok = False
        msgs.append("FAIL: cloth.discovered != true")
    else:
        msgs.append("cloth.discovered: true")

    mesh = str(cloth.get("mesh") or "")
    msgs.append(f"cloth.mesh: {mesh}")
    if not mesh.endswith(".vtk"):
        ok = False
        msgs.append("FAIL: cloth.mesh missing or not .vtk")

    center_yup = cloth.get("center_yup")
    quat_yup = cloth.get("quat_wxyz_yup")
    if not _is_vec3(center_yup):
        ok = False
        msgs.append("FAIL: cloth.center_yup invalid (need 3 floats)")
    else:
        msgs.append(
            f"cloth.center_yup: ({center_yup[0]:.4f}, {center_yup[1]:.4f}, {center_yup[2]:.4f})"
        )
    if not _is_quat_wxyz(quat_yup):
        ok = False
        msgs.append("FAIL: cloth.quat_wxyz_yup invalid (need 4 floats)")
    else:
        msgs.append(
            f"cloth.quat_wxyz_yup: ({quat_yup[0]:.4f}, {quat_yup[1]:.4f}, "
            f"{quat_yup[2]:.4f}, {quat_yup[3]:.4f})"
        )

    discover_only = bool((xpbd_session.get("xpbd") or {}).get("cloth_discover_only", True))
    msgs.append(f"xpbd.cloth_discover_only: {discover_only}")

    session_rigid = xpbd_session.get("rigid_body_map") or []
    adapted_rigid = adapted.get("rigid_body_map") or []
    msgs.append(f"adapted rigid_body_map: {len(adapted_rigid)}")
    msgs.append(f"session rigid_body_map: {len(session_rigid)}")

    ln_ok, ln_bad = _rows_logical_equals_mjc(adapted_rigid)
    if not ln_ok:
        ok = False
        msgs.append("FAIL: adapted rigid_body_map logical_name != mjc_body_name")
        for b in ln_bad[:6]:
            msgs.append(f"  {b}")
    else:
        msgs.append("PASS: logical_name == mjc_body_name (adapted)")

    ln_sess_ok, ln_sess_bad = _rows_logical_equals_mjc(session_rigid)
    if not ln_sess_ok:
        ok = False
        msgs.append("FAIL: session rigid_body_map logical_name != mjc_body_name")
        for b in ln_sess_bad[:6]:
            msgs.append(f"  {b}")

    if discover_only:
        if len(adapted_publish) >= 4 and len(session_publish) >= len(adapted_publish):
            ok = False
            msgs.append(
                "FAIL: session still carries full scanned body map "
                f"({len(session_publish)} >= {len(adapted_publish)})"
            )
        elif len(adapted_publish) >= 4:
            msgs.append(
                f"PASS: N-body scan ({len(adapted_publish)}) not forwarded to XPBD session "
                f"({len(session_publish)} publish rows)"
            )
        if ok:
            msgs.append("PASS: P2.1+P2.2 session cloth discovery + discover_only gate")
    else:
        if len(session_rigid) < len(adapted_rigid):
            ok = False
            msgs.append(
                f"FAIL: session rigid count {len(session_rigid)} < adapted {len(adapted_rigid)}"
            )
        elif len(session_rigid) >= 4:
            msgs.append(
                f"PASS: P2.3 session carries {len(session_rigid)} rigid bodies "
                f"({len(session_publish)} OrcaLink publish)"
            )
        if len(session_publish) != len(adapted_publish):
            ok = False
            msgs.append(
                f"FAIL: publish mismatch session={len(session_publish)} "
                f"adapted={len(adapted_publish)}"
            )
        if ok:
            msgs.append("PASS: P2.3 cloth discovery + full rigid_body_map forward")
    return ok, msgs, session_path


def _capture_text(val: object | None) -> str:
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def verify_p2_xpbd_smoke(
    session_path: Path, *, timeout_sec: float = 8.0, discover_only: bool = True
) -> tuple[bool, list[str]]:
    """P2 C 端：MJC_PBD_CONFIG 指向 session；discover_only 时查布姿，否则再查 scene 刚体数。"""
    msgs: list[str] = []
    if not XPBD_BIN.is_file():
        msgs.append(f"SKIP: XPBD binary not found: {XPBD_BIN}")
        msgs.append("  build: cd XPBD && ./build.sh --release dual_gripper_cross_mjc")
        return True, msgs

    env = os.environ.copy()
    env["MJC_PBD_CONFIG"] = str(session_path.resolve())
    if not discover_only:
        env["MJC_PBD_DISABLE_BASE_PHYS"] = "1"
    try:
        proc = subprocess.run(
            [str(XPBD_BIN)],
            cwd=str(REPO_ROOT / "XPBD"),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = _capture_text(proc.stdout) + _capture_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        out = _capture_text(exc.stdout) + _capture_text(exc.stderr)
        msgs.append(f"XPBD smoke: terminated after {timeout_sec}s (expected for headless demo)")

    smoke_ok = True
    if re.search(r"cloth discovered pose", out, re.IGNORECASE):
        msgs.append("PASS: XPBD log contains 'cloth discovered pose'")
    else:
        msgs.append("FAIL: XPBD log missing 'cloth discovered pose'")
        smoke_ok = False

    if not discover_only:
        m = re.search(r"scene loaded bodies=(\d+)", out)
        if m and int(m.group(1)) >= 4:
            msgs.append(f"PASS: XPBD scene loaded bodies={m.group(1)}")
        else:
            msgs.append("FAIL: XPBD log missing 'scene loaded bodies=' (>=4)")
            smoke_ok = False

    if smoke_ok:
        return True, msgs

    tail = "\n".join(out.strip().splitlines()[-12:])
    if tail:
        msgs.append("--- XPBD log tail ---")
        msgs.extend(f"  {line}" for line in tail.splitlines())
    return False, msgs


def verify_p1(model: mujoco.MjModel, config_path: Path | None) -> tuple[bool, list[str]]:
    """P1：identify_* 扫描 + scan-first config merge。"""
    from modules.identify_xpbd_bodies import (  # noqa: WPS433
        bodies_to_rigid_body_map,
        identify_xpbd_bodies,
    )
    from modules.identify_xpbd_cloth import identify_xpbd_cloth, merge_cloth_discovery  # noqa: WPS433
    body_map_orcagym = _load_body_map_orcagym()
    adapt_config_for_orcagym = body_map_orcagym.adapt_config_for_orcagym

    msgs: list[str] = []
    ok = True

    bodies = identify_xpbd_bodies(model)
    cloths = identify_xpbd_cloth(model)
    msgs.append(f"discovered bodies: {len(bodies)}")
    msgs.append(f"discovered cloths: {len(cloths)}")
    if len(bodies) < 4:
        ok = False
        msgs.append(f"FAIL: expected >=4 track bodies, got {len(bodies)}")
    if len(cloths) != 1:
        ok = False
        msgs.append(f"FAIL: expected 1 cloth, got {len(cloths)}")

    base_cfg: dict = {
        "xpbd_auto_discover": {"bodies": True, "cloth": True},
        "orcagym": {"rigid_body_map_key": "orcagym_rigid_body_map"},
        "orcagym_rigid_body_map": [],
        "cloth": {"mesh": "shirt_v4.vtk"},
    }
    if config_path and config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        base_cfg.update({k: v for k, v in loaded.items() if k != "extends"})

    merged = merge_cloth_discovery(base_cfg, cloths)
    adapted = adapt_config_for_orcagym(model, merged)
    publish = adapted.get("orcalink_rigid_body_map") or adapted.get("rigid_body_map") or []
    msgs.append(f"orcalink publish bodies: {len(publish)}")
    if len(publish) < 4:
        ok = False
        msgs.append("FAIL: adapt_config produced too few publish bodies")

    pose_remap = (adapted.get("orcagym") or {}).get("pose_remap") or {}
    if pose_remap.get("enabled"):
        ok = False
        msgs.append("FAIL: pose_remap still enabled after scan-first merge")

    cloth_mesh = (adapted.get("cloth") or {}).get("mesh")
    msgs.append(f"cloth.mesh: {cloth_mesh}")
    if cloth_mesh != "shirt_v4.vtk":
        ok = False
        msgs.append("FAIL: cloth mesh not from discovery")

    manual = bodies_to_rigid_body_map(bodies)
    msgs.append(f"manual map rows: {len(manual)}")
    if ok:
        msgs.append("PASS: P1 identify + merge")
    return ok, msgs


def main() -> int:
    _setup_paths()
    parser = argparse.ArgumentParser(description="ClothRobot phase verification")
    parser.add_argument("--phase", choices=("p0", "p1", "p2", "all"), default="all")
    parser.add_argument("--mjcf", type=Path, default=FIXTURE, help="MJCF path (default: P0 fixture)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"cloth_sim config JSON (P2 Studio 推荐 {DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--xpbd-smoke",
        action="store_true",
        help="P2 only: run dual_gripper_cross_mjc with MJC_PBD_CONFIG=session",
    )
    args = parser.parse_args()

    mjcf = args.mjcf.expanduser().resolve()
    if not mjcf.is_file():
        print(f"MJCF not found: {mjcf}", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    all_ok = True
    print(f"MJCF: {mjcf}")
    print("=" * 60)

    if args.phase in ("p0", "all"):
        ok, msgs = verify_p0(model)
        print("[P0]")
        for m in msgs:
            print(" ", m)
        all_ok = all_ok and ok
        print()

    if args.phase in ("p1", "all"):
        ok, msgs = verify_p1(model, args.config)
        print("[P1]")
        for m in msgs:
            print(" ", m)
        all_ok = all_ok and ok
        print()

    if args.phase in ("p2", "all"):
        cfg = args.config
        if cfg is None and args.phase == "p2":
            cfg = DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else None
        ok, msgs, session_path = verify_p2(mjcf, cfg)
        print("[P2]")
        for m in msgs:
            print(" ", m)
        all_ok = all_ok and ok
        if args.xpbd_smoke and session_path is not None:
            print()
            print("[P2 XPBD smoke]")
            discover_only = bool(
                json.loads(session_path.read_text(encoding="utf-8"))
                .get("xpbd", {})
                .get("cloth_discover_only", True)
            )
            smoke_ok, smoke_msgs = verify_p2_xpbd_smoke(
                session_path, discover_only=discover_only
            )
            for m in smoke_msgs:
                print(" ", m)
            skipped = any(m.startswith("SKIP:") for m in smoke_msgs)
            if not skipped:
                all_ok = all_ok and smoke_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
