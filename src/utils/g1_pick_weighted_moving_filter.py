"""Weighted moving average filter for dual-arm joint commands.

Ported from unitreerobotics/xr_teleoperate teleop/utils/weighted_moving_filter.py
(visualization helpers omitted).
"""
from __future__ import annotations

import numpy as np


class WeightedMovingFilter:
    def __init__(self, weights, data_size: int = 14):
        self._window_size = len(weights)
        self._weights = np.asarray(weights, dtype=np.float64)
        assert np.isclose(np.sum(self._weights), 1.0), (
            "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        )
        self._data_size = int(data_size)
        self._filtered_data = np.zeros(self._data_size, dtype=np.float64)
        self._data_queue: list[np.ndarray] = []

    def _apply_filter(self) -> np.ndarray:
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.asarray(self._data_queue, dtype=np.float64)
        temp_filtered_data = np.zeros(self._data_size, dtype=np.float64)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(
                data_array[:, i], self._weights, mode="valid"
            )[-1]
        return temp_filtered_data

    def add_data(self, new_data) -> None:
        new_data = np.asarray(new_data, dtype=np.float64).reshape(-1)
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return

        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data.copy())
        self._filtered_data = self._apply_filter()

    def reset(self) -> None:
        self._data_queue.clear()
        self._filtered_data = np.zeros(self._data_size, dtype=np.float64)

    @property
    def filtered_data(self) -> np.ndarray:
        return self._filtered_data
