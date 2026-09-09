"""G1 OmniPicker / Unitree G1 的 LeRobot storage，状态结构委托给 PolicySchema。"""
from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage
from dataStorage.g1_pick_osc_data_storage import G1PickOscDataStorage
from dataStorage.lerobot_storage import LeRobotSimSyncMixin
from policy.dual_arm_schema import g1_omnipicker_schema, g1_pick_osc_schema


class G1OmniPickerLeRobotStorage(LeRobotSimSyncMixin, G1OmniPickerDataStorage):
    def __init__(self, dataset_path: str, **lerobot_kwargs) -> None:
        super().__init__(
            dataset_path=dataset_path, hdf5_path=None, schema=g1_omnipicker_schema()
        )
        if lerobot_kwargs:
            self.setup_lerobot(**lerobot_kwargs)


class G1PickOscLeRobotStorage(LeRobotSimSyncMixin, G1PickOscDataStorage):
    def __init__(self, dataset_path: str, **lerobot_kwargs) -> None:
        super().__init__(
            dataset_path=dataset_path, hdf5_path=None, schema=g1_pick_osc_schema()
        )
        if lerobot_kwargs:
            self.setup_lerobot(**lerobot_kwargs)
