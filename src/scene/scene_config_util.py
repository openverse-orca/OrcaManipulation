"""场景 YAML 加载与任务类型解析（数据采集脚本共用）。"""
import os

from yaml import Loader, load

from orca_gym.environment import OrcaGymLocalEnv
from task.abstract_task import AbstractTask, EmptyTask
from task.pick_place_task import PickPlaceTask

# 顶层 type：仅遥操/录制，不做抓取成功判定（参考 conveyor_collect.yaml）
COLLECT_ONLY_TYPES = frozenset({"collect_only", "collection", "manual_record"})


def load_scene_config(base_dir: str, task_config: str | None) -> dict:
    """加载相对 base_dir 的 YAML；task_config 为空时返回空配置（场景由关卡自带）。"""
    if not task_config:
        return {}
    path = os.path.join(base_dir, task_config)
    with open(path, "r", encoding="utf-8") as f:
        return load(f, Loader=Loader) or {}


def should_use_empty_task(config: dict, task_config: str | None) -> bool:
    if not task_config:
        return True
    if config.get("type") in COLLECT_ONLY_TYPES:
        return True
    if not config.get("task"):
        return True
    return False


def create_task(env: OrcaGymLocalEnv, config: dict, task_config: str | None) -> AbstractTask:
    if should_use_empty_task(config, task_config):
        return EmptyTask(env)
    return PickPlaceTask(env)
