"""dataCollection 工具包；避免顶层导入 orca_gym 以便纯数学单测。"""

__all__ = ["KettleTrajectoryDriver"]


def __getattr__(name: str):
    if name == "KettleTrajectoryDriver":
        from .kettle_trajectory_driver import KettleTrajectoryDriver

        return KettleTrajectoryDriver
    raise AttributeError(name)
