#!/usr/bin/env python3
"""
从 cloth_sim_config + MJCF 导出 XPBD 离线场景 JSON（与 body_map.py / OrcaLink 合同同序）。

输出默认：XPBD/MjcPBD_orcalink/debug_log/xpbd_scene_from_mjcf.json

用法:
  python scripts/export_xpbd_scene_from_mjcf.py
  python scripts/export_xpbd_scene_from_mjcf.py --config cloth_sim_config.phase1_slide.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# ROOT = src/envs/softbody；parents[3] = OrcaApr24 仓库根（含 XPBD/）。
REPO = ROOT.parents[3]
DEFAULT_CONFIG = ROOT / "cloth_sim_config.phase1_slide.json"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export xpbd_scene_from_mjcf.json from MuJoCo MJCF")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="cloth_sim_config JSON")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出路径（默认 debug.debug_log_dir/xpbd_scene_from_mjcf.json）",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"FAIL: config not found: {config_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    from domain.anchor_tetrahedron import anchor_local_positions, anchor_site_names  # noqa: E402
    from domain.body_map import load_body_map  # noqa: E402
    from domain.mjc_coords import (  # noqa: E402
        orca_quat_to_yup_link_orientation,
        orca_vec_to_yup,
    )

    def half_extents_mjc_to_yup(hx: float, hy: float, hz: float) -> list[float]:
        """盒体半长：仅轴置换 (x,y,z)_mjc -> (x,z,y)_yup，取绝对值（无镜像）。"""
        yx, yy, yz = orca_vec_to_yup(hx, hy, hz)
        return [abs(yx), abs(yy), abs(yz)]

    def body_track_body_only(cfg: dict) -> bool:
        bt = cfg.get("body_track") or {}
        return bool(bt.get("enabled")) and not bool(bt.get("use_anchor_sites", False))

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent
    meta = cfg.get("_cloth_robot_session_meta") or {}
    src_mjcf = meta.get("source_mjcf")
    mjcf_rel = cfg.get("mujoco", {}).get("model_path", "")
    if src_mjcf and Path(src_mjcf).is_file():
        mjcf_path = Path(src_mjcf).resolve()
    elif mjcf_rel:
        mjcf_path = (config_dir / mjcf_rel).resolve()
    else:
        print("FAIL: missing mujoco.model_path or _cloth_robot_session_meta.source_mjcf", file=sys.stderr)
        return 1

    dbg = cfg.get("debug", {})
    if args.out is not None:
        out_path = args.out.resolve()
    else:
        dbg_dir = Path(dbg.get("debug_log_dir", REPO / "XPBD" / "MjcPBD_orcalink" / "debug_log"))
        if not dbg_dir.is_absolute():
            dbg_dir = (config_dir / dbg_dir).resolve()
        out_path = dbg_dir / "xpbd_scene_from_mjcf.json"

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    entries_by_mjc = {e.mjc_body_name: e for e in load_body_map(model, cfg)}
    map_rows = cfg.get("rigid_body_map", [])
    if not map_rows:
        print("FAIL: rigid_body_map is empty", file=sys.stderr)
        return 1

    all_entries = [entries_by_mjc[row["mjc_body_name"]] for row in map_rows]
    body_only = body_track_body_only(cfg)
    if body_only:
        for row in map_rows:
            mjc_name = row["mjc_body_name"]
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjc_name) < 0:
                print(f"FAIL: body missing {mjc_name}", file=sys.stderr)
                return 1
    else:
        from domain.body_map import validate_body_map  # noqa: E402

        val_errs = validate_body_map(model, all_entries)
        if val_errs:
            print("FAIL: body_map validation:")
            for e in val_errs:
                print(" ", e)
            return 1

    bodies_out: list[dict] = []
    for body_index, row in enumerate(map_rows):
        mjc_name = row["mjc_body_name"]
        entry = entries_by_mjc[mjc_name]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mjc_name)
        follow = row.get("follow_mode", entry.follow_mode)
        mj_mass = float(model.body_mass[bid])
        mass_kg = 0.0 if follow == "kinematic" else mj_mass

        hx, hy, hz = entry.box_half_extents
        half_yup = half_extents_mjc_to_yup(hx, hy, hz)

        xpos = data.xpos[bid].astype(np.float64)
        xquat = data.xquat[bid].astype(np.float64)
        center_yup = orca_vec_to_yup(float(xpos[0]), float(xpos[1]), float(xpos[2]))
        quat_yup = orca_quat_to_yup_link_orientation(
            float(xquat[0]), float(xquat[1]), float(xquat[2]), float(xquat[3])
        )

        anchors_out: list[dict] = []
        anchor_sites = list(entry.anchor_sites)
        if body_only and len(anchor_sites) < 4:
            anchor_sites = anchor_site_names(mjc_name)
        for ai, sname in enumerate(anchor_sites):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                lp = model.site_pos[sid].astype(np.float64)
                local_mjc = [float(lp[0]), float(lp[1]), float(lp[2])]
            elif body_only:
                _, verts = anchor_local_positions(hx, hy, hz)
                vx, vy, vz = verts[ai]
                local_mjc = [float(vx), float(vy), float(vz)]
            else:
                print(f"FAIL: site missing {sname}", file=sys.stderr)
                return 1
            local_yup = list(orca_vec_to_yup(local_mjc[0], local_mjc[1], local_mjc[2]))
            anchors_out.append(
                {
                    "index": ai,
                    "site_name": sname,
                    "local_pos_mjc": local_mjc,
                    "local_pos_yup": local_yup,
                }
            )

        if len(anchors_out) != 4:
            print(f"FAIL: {mjc_name} expected 4 anchors, got {len(anchors_out)}", file=sys.stderr)
            return 1

        bodies_out.append(
            {
                "body_index": body_index,
                "logical_name": mjc_name,
                "mjc_body_name": mjc_name,
                "follow_mode": follow,
                "mass_kg": mass_kg,
                "mass_mjcf_kg": mj_mass,
                "box_half_extents_mjc": [hx, hy, hz],
                "box_half_extents_yup": list(half_yup),
                "center_mjc": [float(xpos[0]), float(xpos[1]), float(xpos[2])],
                "center_yup": list(center_yup),
                "quat_wxyz_mjc": [
                    float(xquat[0]),
                    float(xquat[1]),
                    float(xquat[2]),
                    float(xquat[3]),
                ],
                "quat_wxyz_yup": list(quat_yup),
                "anchors": anchors_out,
            }
        )

    doc = {
        "schema_version": 1,
        "coord_system_sim": "yup",
        "coord_system_mjcf": "zup",
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_config": str(config_path),
        "source_mjcf": str(mjcf_path),
        "body_count": len(bodies_out),
        "bodies": bodies_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {out_path} ({len(bodies_out)} bodies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
