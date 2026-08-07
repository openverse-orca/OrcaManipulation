import abc
import numpy as np
from scipy.interpolate import CubicSpline
from orca_gym.log import OrcaLog

orca_logger = OrcaLog.get_instance()

class AbstractInterpolator(metaclass=abc.ABCMeta):

    def __init__(self, noise_value: float):
        self.noise_value = noise_value

    @abc.abstractmethod
    def interpolate(self, dataset: np.array, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_interpolation_paths(self) -> list[str]:
        """返回需要插值的数据集路径列表"""
        raise NotImplementedError


class OpenLoongInterpolator(AbstractInterpolator):
    def __init__(self, noise_value: float):
        super().__init__(noise_value)
        self._insertion_indices = None  # 缓存插值位置

    def get_interpolation_paths(self) -> list[str]:
        """返回需要插值的数据集路径列表"""
        return [
            "/action/effector/motor",
            "/action/end/position",
            "/action/end/orientation"
        ]

    def interpolate(self, dataset: np.array, **kwargs):
        dataset_path = kwargs.get("dataset_path", None)
        if dataset_path is None:
            raise ValueError("dataset_path is required")
        if dataset_path == "/action/effector/motor":
            return self.interpolate_effector_motor(dataset)
        elif dataset_path == "/action/end/position":
            return self.interpolate_end_position(dataset)
        elif dataset_path == "/action/end/orientation":
            return self.interpolate_end_orientation(dataset)
        

    def interpolate_effector_motor(self, dataset: np.array):
        """
        插值抓夹控制值
        维度：(N, 2) - 两个抓夹的控制值
        注意：motor值为离散整数，插值后需要取整
        """
        data_array = np.array(dataset)
        result = self._interpolate_linear(data_array, save_indices=False)
        # Motor值为离散整数，取整
        result = np.round(result).astype(np.int32)
        return result

    def interpolate_end_position(self, dataset: np.array):
        """
        插值末端位置
        支持flatten后的多臂数据：(N, 6) -> 两个3D位置 (N, 2, 3)
        """
        data_array = np.array(dataset)
        
        # 检测是否是flatten的多臂位置数据
        if len(data_array.shape) == 2 and data_array.shape[1] % 3 == 0:
            num_arms = data_array.shape[1] // 3
            if num_arms > 1:
                # 直接对flatten的数据进行插值
                # 因为距离计算和统计量计算对flatten数据也是有效的
                result = self._interpolate_linear(data_array, save_indices=True)
                return result
        
        # 单臂或未flatten的情况
        result = self._interpolate_linear(dataset, save_indices=True)
        return result

    def interpolate_end_orientation(self, dataset: np.array):
        """
        插值末端方向(四元数)
        支持flatten后的多臂数据：(N, 8) -> 两个四元数 (N, 2, 4)
        """
        data_array = np.array(dataset)
        
        # 检测是否是flatten的多臂四元数数据
        if len(data_array.shape) == 2 and data_array.shape[1] % 4 == 0:
            num_arms = data_array.shape[1] // 4
            if num_arms > 1:
                # Reshape: (N, num_arms*4) -> (N, num_arms, 4)
                reshaped = data_array.reshape(data_array.shape[0], num_arms, 4)
                
                # 对每个arm分别插值
                interpolated_arms = []
                for arm_idx in range(num_arms):
                    arm_data = reshaped[:, arm_idx, :]
                    interpolated = self._interpolate_quaternion(
                        arm_data, 
                        use_saved_indices=True
                    )
                    interpolated_arms.append(interpolated)
                
                # 合并结果并flatten: (N', num_arms, 4) -> (N', num_arms*4)
                result = np.stack(interpolated_arms, axis=1)
                result = result.reshape(result.shape[0], -1)
                return result
        
        # 单臂或未flatten的情况
        result = self._interpolate_quaternion(dataset, use_saved_indices=True)
        return result
    
    def _interpolate_linear(self, dataset: np.array, save_indices: bool = False) -> np.array:
        """
        线性数据插值：每4个值插入1个新值
        根据相邻点之间的距离，在变化最大的位置插值
        使用平均值 + 标准差 + random(0, noise)
        """
        data_array = np.array(dataset)
        
        if len(data_array) < 4:
            return data_array
        
        result = []
        num_groups = len(data_array) // 4
        new_values = []
        insertion_indices = []
        
        for i in range(num_groups):
            group = data_array[i*4:(i+1)*4]
            
            # 计算相邻点之间的欧氏距离
            distances = []
            for j in range(3):
                dist = np.linalg.norm(group[j+1] - group[j])
                distances.append(dist)
            
            # 找到距离最大的位置（变化最剧烈）
            max_dist_idx = np.argmax(distances)
            insertion_indices.append(max_dist_idx)
            
            # 计算插值值
            mean = np.mean(group, axis=0)
            std = np.std(group, axis=0)
            noise = np.random.uniform(0, self.noise_value, size=group[0].shape)
            interpolated = mean + std + noise
            new_values.append(interpolated)
            
            # 在合适的位置插入新值
            for j in range(4):
                result.append(group[j])
                if j == max_dist_idx:
                    result.append(interpolated)
            
        
        # 保存插值位置供orientation使用
        if save_indices:
            self._insertion_indices = insertion_indices
        
        remaining = len(data_array) % 4
        if remaining > 0:
            result.extend(data_array[num_groups*4:])
        
        if new_values:
            new_values_array = np.array(new_values)
        
        return np.array(result)
    
    def _interpolate_quaternion(self, dataset: np.array, use_saved_indices: bool = False) -> np.array:
        """
        四元数插值：使用SLERP，每4个值插入1个新值
        使用与position相同的插值位置，保持数据一致性
        """
        data_array = np.array(dataset)
        
        if len(data_array) < 4:
            return data_array
        
        result = []
        num_groups = len(data_array) // 4
        new_values = []
        
        for i in range(num_groups):
            group = data_array[i*4:(i+1)*4]
            
            # 决定插值位置
            if use_saved_indices and self._insertion_indices is not None and i < len(self._insertion_indices):
                # 使用position计算的插值位置
                insert_idx = self._insertion_indices[i]
            else:
                # 计算相邻四元数之间的角度差异
                angle_diffs = []
                for j in range(3):
                    dot_product = np.abs(np.sum(group[j] * group[j+1], axis=-1))
                    dot_product = np.clip(dot_product, 0.0, 1.0)
                    angle = np.arccos(dot_product)
                    angle_diffs.append(angle)
                
                # 找到角度变化最大的位置
                insert_idx = np.argmax(angle_diffs)
            
            # 计算插值值
            q_interp = self._slerp_multiple(group, 0.5)
            noise = np.random.uniform(-self.noise_value, self.noise_value, size=q_interp.shape)
            q_interp = q_interp + noise
            q_interp = q_interp / np.linalg.norm(q_interp, axis=-1, keepdims=True)
            new_values.append(q_interp)
            
            # 在合适的位置插入新值
            for j in range(4):
                result.append(group[j])
                if j == insert_idx:
                    result.append(q_interp)
        
        remaining = len(data_array) % 4
        if remaining > 0:
            result.extend(data_array[num_groups*4:])
        
        if new_values:
            new_values_array = np.array(new_values)
            norms = np.linalg.norm(new_values_array, axis=-1)
        
        return np.array(result)
    
    def _slerp_multiple(self, quaternions: np.array, t: float) -> np.array:
        """对多个四元数进行球面线性插值"""
        q_avg = np.mean(quaternions, axis=0)
        q_avg = q_avg / np.linalg.norm(q_avg, axis=-1, keepdims=True)
        return q_avg


class OpenLoongInterpolatorAdvanced(AbstractInterpolator):
    """改进版插值器：使用三次样条插值和SLERP"""
    
    def __init__(self, noise_value: float, interpolation_factor: int = 3):
        """
        @param noise_value: 噪声值
        @param interpolation_factor: 插值倍数，每两个点之间插入的点数
        """
        super().__init__(noise_value)
        self.interpolation_factor = interpolation_factor

    def get_interpolation_paths(self) -> list[str]:
        """返回需要插值的数据集路径列表"""
        return [
            "/action/effector/motor",
            "/action/end/position",
            "/action/end/orientation"
        ]

    def interpolate(self, dataset: np.array, **kwargs):
        dataset_path = kwargs.get("dataset_path", None)
        if dataset_path is None:
            raise ValueError("dataset_path is required")
        if dataset_path == "/action/effector/motor":
            return self.interpolate_effector_motor(dataset)
        elif dataset_path == "/action/end/position":
            return self.interpolate_end_position(dataset)
        elif dataset_path == "/action/end/orientation":
            return self.interpolate_end_orientation(dataset)
        
    def interpolate_effector_motor(self, dataset: np.array):
        """使用三次样条插值抓夹控制值"""
        return self._cubic_spline_interpolate(dataset)

    def interpolate_end_position(self, dataset: np.array):
        """使用三次样条插值末端位置"""
        return self._cubic_spline_interpolate(dataset)

    def interpolate_end_orientation(self, dataset: np.array):
        """使用SLERP插值末端方向(四元数)"""
        return self._slerp_interpolate(dataset)
    
    def _cubic_spline_interpolate(self, dataset: np.array) -> np.array:
        """使用三次样条插值，生成平滑曲线"""
        if len(dataset) < 2:
            return dataset
        
        original_shape = dataset.shape
        if len(original_shape) == 1:
            dataset = dataset.reshape(-1, 1)
        
        n = len(dataset)
        t_original = np.arange(n)
        t_new = np.linspace(0, n-1, (n-1) * self.interpolation_factor + 1)
        
        result = []
        for dim in range(dataset.shape[-1]):
            cs = CubicSpline(t_original, dataset[:, dim])
            interpolated = cs(t_new)
            noise = np.random.uniform(-self.noise_value, self.noise_value, size=interpolated.shape)
            result.append(interpolated + noise)
        
        result = np.array(result).T
        if len(original_shape) == 1:
            result = result.flatten()
        
        return result
    
    def _slerp_interpolate(self, dataset: np.array) -> np.array:
        """使用SLERP对四元数进行球面线性插值"""
        if len(dataset) < 2:
            return dataset
        
        n = len(dataset)
        result = [dataset[0]]
        
        for i in range(n - 1):
            q1 = dataset[i]
            q2 = dataset[i + 1]
            
            for j in range(1, self.interpolation_factor + 1):
                t = j / self.interpolation_factor
                if j < self.interpolation_factor:
                    q_interp = self._slerp(q1, q2, t)
                    noise = np.random.uniform(-self.noise_value, self.noise_value, size=q_interp.shape)
                    q_interp = q_interp + noise
                    q_interp = q_interp / np.linalg.norm(q_interp, axis=-1, keepdims=True)
                    result.append(q_interp)
                else:
                    result.append(q2)
        
        return np.array(result[:-1])
    
    def _slerp(self, q1: np.array, q2: np.array, t: float) -> np.array:
        """球面线性插值"""
        dot = np.sum(q1 * q2, axis=-1, keepdims=True)
        
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        dot = np.clip(dot, -1.0, 1.0)
        
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        if sin_theta < 1e-6:
            return q1
        
        w1 = np.sin((1.0 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2
