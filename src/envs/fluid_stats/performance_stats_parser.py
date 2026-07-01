#!/usr/bin/env python3
"""
Parse OrcaSPH performance stats lines.

Supports two formats:
1. Legacy flat format::

    [TIME_STATS] initializeCPUDataEachTimeStep=1.234 initializeGPUDataEachTimeStep=0.567 ...

2. Tree format::

    [TIME_STATS_TREE]
    ├─ timeStepNoGUI                          10.500ms  100.0%
    │  ├─ StepBeforeSimulation                 0.100ms    1.0%
    ...
    [TIME_STATS_TREE_META] numSteps_avg=5.0 batch_count=10 step_count=50

Used by the matplotlib performance-stats viewer; kept free of pyplot and gymnasium.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STAT_MARKER = "[TIME_STATS]"
TREE_BATCH_MARKER = "[TIME_STATS_TREE_BATCH]"
TREE_STEP_MARKER = "[TIME_STATS_TREE_STEP]"
TREE_META_MARKER = "[TIME_STATS_TREE_META]"


@dataclass
class TimingTreeNode:
    name: str
    time_ms: float = 0.0
    pct: float = 0.0
    children: List["TimingTreeNode"] = field(default_factory=list)
    self_time_ms: float = 0.0
    self_pct: float = 0.0


def _compute_depth(prefix: str) -> int:
    idx = -1
    for marker in ("\u251c\u2500", "\u2514\u2500"):
        pos = prefix.rfind(marker)
        if pos >= 0:
            idx = pos
            break
    if idx < 0:
        return 0
    return idx // 3


_TREE_LINE_RE = re.compile(
    r"^([\u2502\u251c\u2514\u2500 |]+)\s+"
    r"(.+?)\s+"
    r"([\d.]+)ms\s+"
    r"([\d.]+)%\s*$"
)

_SELF_LINE_RE = re.compile(
    r"^([\u2502\u251c\u2514\u2500 |]+)\s+"
    r"\[self\]\s+"
    r"([\d.]+)ms\s+"
    r"([\d.]+)%\s*$"
)

_SMALL_ITEMS_RE = re.compile(
    r"^([\u2502\u251c\u2514\u2500 |]+)\s+"
    r"\[(\d+)\s+items[^\]]*\]\s+"
    r"([\d.]+)ms\s+"
    r"([\d.]+)%\s*$"
)


def parse_tree_stats_block(lines: List[str]) -> Optional[TimingTreeNode]:
    if not lines:
        return None

    root: Optional[TimingTreeNode] = None
    stack: List[TimingTreeNode] = []

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        m_self = _SELF_LINE_RE.match(line)
        if m_self:
            prefix = m_self.group(1)
            self_time = float(m_self.group(2))
            self_pct = float(m_self.group(3))
            depth = _compute_depth(prefix)
            target_depth = depth - 1
            if target_depth >= 0 and target_depth < len(stack):
                stack[target_depth].self_time_ms = self_time
                stack[target_depth].self_pct = self_pct
            continue

        m_small = _SMALL_ITEMS_RE.match(line)
        if m_small:
            continue

        m = _TREE_LINE_RE.match(line)
        if not m:
            continue

        prefix = m.group(1)
        name = m.group(2).strip()
        time_ms = float(m.group(3))
        pct = float(m.group(4))

        node = TimingTreeNode(name=name, time_ms=time_ms, pct=pct)
        depth = _compute_depth(prefix)

        if root is None:
            root = node
            stack = [root]
            continue

        while len(stack) > depth:
            stack.pop()

        if stack:
            stack[-1].children.append(node)

        stack.append(node)

    return root


def tree_to_flat(node: TimingTreeNode, out: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if out is None:
        out = {}
    out[node.name] = node.time_ms
    for child in node.children:
        tree_to_flat(child, out)
    return out


def parse_performance_stats_line(line: str) -> Optional[Dict[str, Any]]:
    """Return a dict of typed fields, or None if the line is not a stats line."""
    if STAT_MARKER not in line:
        return None
    i = line.find(STAT_MARKER)
    rest = line[i + len(STAT_MARKER) :].strip()
    if not rest:
        return None
    parts = rest.split()
    raw: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        raw[k] = v
    if not raw:
        return None
    try:
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        return out
    except (KeyError, ValueError):
        return None


@dataclass
class TailState:
    """Incremental read state for a single log file."""

    path: Path
    offset: int = 0


def read_new_performance_records(
    path: Path,
    state: TailState,
) -> List[Dict[str, Any]]:
    """Read newly appended bytes since *state.offset* and return parsed performance records."""
    state.path = path
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size < state.offset:
        state.offset = 0
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(state.offset)
            chunk = f.read()
            state.offset = f.tell()
    except OSError:
        return []
    
    in_stats_block = False
    in_tree_block = None  # None, "batch", or "step"
    tree_lines: List[str] = []
    current_record = {}
    last_dt_value: Optional[float] = None

    for raw_line in chunk.splitlines():
        line = raw_line.strip()

        if "[EachtimeStepInH2D]" in line and "dt_s=" in line:
            try:
                dt_part = line.split("dt_s=")[1]
                dt_str = dt_part.split("x")[0].strip()
                last_dt_value = float(dt_str)
            except:
                pass
            continue

        if TREE_META_MARKER in line:
            if in_tree_block:
                tree_node = parse_tree_stats_block(tree_lines)
                if tree_node:
                    flat = tree_to_flat(tree_node)
                    current_record.update(flat)
                    if in_tree_block == "batch":
                        current_record["_tree_batch"] = tree_node
                    elif in_tree_block == "step":
                        current_record["_tree_step"] = tree_node
                tree_lines = []
                in_tree_block = None
            meta_part = line[line.find(TREE_META_MARKER) + len(TREE_META_MARKER):].strip()
            meta = {}
            for p in meta_part.split():
                if "=" in p:
                    k, v = p.split("=", 1)
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        meta[k] = v
            if current_record is not None:
                current_record.update({"_meta_" + k: v for k, v in meta.items()})
            if current_record:
                if last_dt_value is not None:
                    current_record["_dt_s"] = last_dt_value
                records.append(current_record)
                current_record = {}
            continue

        if TREE_BATCH_MARKER in line:
            if in_stats_block and current_record:
                if last_dt_value is not None:
                    current_record["_dt_s"] = last_dt_value
                records.append(current_record)
            in_stats_block = False
            current_record = {}

            in_tree_block = "batch"
            tree_lines = []
            continue

        if TREE_STEP_MARKER in line:
            if in_tree_block == "batch":
                tree_node = parse_tree_stats_block(tree_lines)
                if tree_node:
                    flat = tree_to_flat(tree_node)
                    current_record.update(flat)
                    current_record["_tree_batch"] = tree_node
                tree_lines = []

            in_tree_block = "step"
            tree_lines = []
            continue

        if in_tree_block:
            stripped = raw_line.rstrip("\n\r")
            if not stripped or (not stripped.startswith("\u2502") and not stripped.startswith("\u251c") and not stripped.startswith("\u2514") and not stripped.startswith(" ")):
                tree_node = parse_tree_stats_block(tree_lines)
                if tree_node:
                    flat = tree_to_flat(tree_node)
                    current_record.update(flat)
                    if in_tree_block == "batch":
                        current_record["_tree_batch"] = tree_node
                    elif in_tree_block == "step":
                        current_record["_tree_step"] = tree_node
                tree_lines = []
                in_tree_block = None
                if current_record:
                    if last_dt_value is not None:
                        current_record["_dt_s"] = last_dt_value
                    records.append(current_record)
                    current_record = {}
                continue
            tree_lines.append(stripped)
            continue
        
        if STAT_MARKER in line:
            if in_stats_block and current_record:
                if last_dt_value is not None:
                    current_record["_dt_s"] = last_dt_value
                records.append(current_record)
            in_stats_block = True
            current_record = {}
            rest = line[line.find(STAT_MARKER) + len(STAT_MARKER):].strip()
            if rest and "=" in rest:
                for p in rest.split():
                    if "=" in p:
                        k, v = p.split("=", 1)
                        try:
                            current_record[k] = float(v)
                        except ValueError:
                            current_record[k] = v
        elif in_stats_block:
            if "=" in line:
                try:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        current_record[k] = float(v)
                    except ValueError:
                        current_record[k] = v
                except:
                    pass
            else:
                if current_record:
                    if last_dt_value is not None:
                        current_record["_dt_s"] = last_dt_value
                    records.append(current_record)
                in_stats_block = False
                current_record = {}
    
    if in_tree_block and tree_lines:
        tree_node = parse_tree_stats_block(tree_lines)
        if tree_node:
            flat = tree_to_flat(tree_node)
            current_record.update(flat)
            if in_tree_block == "batch":
                current_record["_tree_batch"] = tree_node
            elif in_tree_block == "step":
                current_record["_tree_step"] = tree_node
            if last_dt_value is not None:
                current_record["_dt_s"] = last_dt_value
            records.append(current_record)
    
    if in_stats_block and current_record:
        if last_dt_value is not None:
            current_record["_dt_s"] = last_dt_value
        records.append(current_record)
    
    return records


def merge_dedupe(accumulator: List[Dict[str, Any]], new_records: List[Dict[str, Any]]) -> None:
    """Append *new_records* to *accumulator*, dropping duplicates."""
    for r in new_records:
        # Check if this record is already in the accumulator
        duplicate = False
        for existing in accumulator:
            if existing == r:
                duplicate = True
                break
        if not duplicate:
            accumulator.append(r)


def compute_average_times(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute average times for each component."""
    if not records:
        return {}
    
    # Get all unique keys
    keys = set()
    for record in records:
        keys.update(record.keys())
    
    # Compute averages
    averages = {}
    for key in keys:
        values = []
        for record in records:
            if key in record:
                values.append(float(record[key]))
        if values:
            averages[key] = sum(values) / len(values)
    
    return averages


def load_all_performance_records_from_file(path: Path) -> List[Dict[str, Any]]:
    """Parse every performance stats line in *path* from the beginning."""
    if not path.exists():
        return []
    
    state = TailState(path, offset=0)
    return read_new_performance_records(path, state)