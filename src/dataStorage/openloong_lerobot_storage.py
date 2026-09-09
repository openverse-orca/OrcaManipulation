"""OpenLoong / Tiangong2 的 LeRobot storage。"""
from dataStorage.lerobot_storage import LeRobotSimSyncMixin
from dataStorage.openloong_data_storage import OpenLoongDataStorage
from dataStorage.tiangong_data_storage import Tiangong2DataStorage
from policy.openloong_schema import OpenLoongPolicySchema
from policy.tiangong2_schema import Tiangong2PolicySchema


class OpenLoongLeRobotStorage(LeRobotSimSyncMixin, OpenLoongDataStorage):
    def __init__(self, dataset_path: str, **lerobot_kwargs) -> None:
        super().__init__(
            dataset_path=dataset_path, hdf5_path=None, schema=OpenLoongPolicySchema()
        )
        if lerobot_kwargs:
            self.setup_lerobot(**lerobot_kwargs)


class Tiangong2LeRobotStorage(LeRobotSimSyncMixin, Tiangong2DataStorage):
    def __init__(self, dataset_path: str, **lerobot_kwargs) -> None:
        super().__init__(
            dataset_path=dataset_path, hdf5_path=None, schema=Tiangong2PolicySchema()
        )
        if lerobot_kwargs:
            self.setup_lerobot(**lerobot_kwargs)
