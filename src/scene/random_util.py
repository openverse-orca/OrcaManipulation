import numpy as np
from scipy.spatial.transform import Rotation

def get_random_qpos(qpos_bound: np.ndarray, dof: int)-> np.array:
    if dof == 1 or dof == 3:
        qpos = np.random.uniform(qpos_bound[0][0], qpos_bound[0][1])
    elif dof == 6:
        position = np.array([np.random.uniform(qpos_bound[0][0], qpos_bound[0][1]) if qpos_bound[0][0] != qpos_bound[0][1] else qpos_bound[0][0], 
                             np.random.uniform(qpos_bound[1][0], qpos_bound[1][1]) if qpos_bound[1][0] != qpos_bound[1][1] else qpos_bound[1][0], 
                             np.random.uniform(qpos_bound[2][0], qpos_bound[2][1]) if qpos_bound[2][0] != qpos_bound[2][1] else qpos_bound[2][0]])
        rotation = np.array([np.random.uniform(qpos_bound[3][0], qpos_bound[3][1]) if qpos_bound[3][0] != qpos_bound[3][1] else qpos_bound[3][0], 
                             np.random.uniform(qpos_bound[4][0], qpos_bound[4][1]) if qpos_bound[4][0] != qpos_bound[4][1] else qpos_bound[4][0], 
                             np.random.uniform(qpos_bound[5][0], qpos_bound[5][1]) if qpos_bound[5][0] != qpos_bound[5][1] else qpos_bound[5][0]])
        quat = Rotation.from_euler('xyz', rotation).as_quat()
        qpos = np.concatenate([position, quat[[3, 0, 1, 2]]])
    return qpos


def get_random_transform(position_bound: np.array, rotation_bound: np.array)-> np.array:
    position = np.array([np.random.uniform(position_bound[0][0], position_bound[0][1]) if position_bound[0][0] != position_bound[0][1] else position_bound[0][0], 
                         np.random.uniform(position_bound[1][0], position_bound[1][1]) if position_bound[1][0] != position_bound[1][1] else position_bound[1][0], 
                         np.random.uniform(position_bound[2][0], position_bound[2][1]) if position_bound[2][0] != position_bound[2][1] else position_bound[2][0]])
    rotation = np.array([np.random.uniform(rotation_bound[0][0], rotation_bound[0][1]) if rotation_bound[0][0] != rotation_bound[0][1] else rotation_bound[0][0], 
                         np.random.uniform(rotation_bound[1][0], rotation_bound[1][1]) if rotation_bound[1][0] != rotation_bound[1][1] else rotation_bound[1][0], 
                         np.random.uniform(rotation_bound[2][0], rotation_bound[2][1]) if rotation_bound[2][0] != rotation_bound[2][1] else rotation_bound[2][0]])
    quat = Rotation.from_euler('xyz', rotation).as_quat()
    return np.concatenate([position, quat])

def choose_random_indices(nums: int, nums_range: list[int])-> list[int]:
    pick_nums = (np.random.randint(nums_range[0], nums_range[1]) 
                if nums_range[0] != nums_range[1] else nums_range[0])
    return np.random.choice(range(nums), pick_nums, replace=False)