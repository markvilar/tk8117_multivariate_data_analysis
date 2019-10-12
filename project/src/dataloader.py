import os
import numpy as np
from typing import Tuple

class DataLoader:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path;
        self._hyper = None;
        self._imu = None;
        self._rgb = None;
        self._hyper_idx = -1;
        self._imu_idx = -1;
        self._rgb_idx = -1;
        raise NotImplementedError

    def get_hyper(self) -> np.ndarray:
        raise NotImplementedError

    def get_imu(self) -> np.ndarray:
        raise NotImplementedError

    def get_rgb(self) -> np.ndarray:
        raise NotImplementedError


def test():
    print('hello')

if __name__ == '__main__':
    test()
