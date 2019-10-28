#!/usr/bin/env Python
import h5py
import os
import numpy as np
from typing import List, Dict
from queue import Queue

def read_h5_file(dir_path: str, file_name: str, search_keys: List[str], queue_size: int) -> Dict[str, np.ndarray]:
    file_path = dir_path + '/' + file_name
    assert file_name.lower().endswith('.h5'), 'File is not a h5 file: {}'.format(file_name)
    assert os.path.exists(dir_path), 'Folder does not exist: {}'.format(os.path.abspath(dir_path))
    assert os.path.isfile(file_path), 'File does not exist: {}'.format(os.path.abspath(file_path))

    search_keys = [x.lower() for x in search_keys]
    datasets = dict()
    group_queue = Queue(queue_size)

    with h5py.File(file_path, 'r') as f:
        for key, value in f.items():
            if isinstance(value, h5py.Group):
                group_queue.put(value)
            elif isinstance(value, h5py.Dataset):
                if key.lower() in search_keys:
                    datasets[key] = value
        while not group_queue.empty():
            group = group_queue.get()
            name = group.name
            for key, value in group.items():
                if isinstance(value, h5py.Group):
                    group_queue.put(value)
                elif isinstance(value, h5py.Dataset):
                    print('name: {}, key: {}, value: {}'.format(name, key, value))
                    if key.lower() in search_keys:
                        key = name + '/' + key;
                        arr = np.empty(shape=value.shape, dtype=value.dtype)
                        value.read_direct(arr)
                        datasets[key] = arr
    
    return datasets

def remove_dict_keys(data: Dict, to_remove: List):
    ''' Remove the dictionary entries based on the entry key. '''
    for key in to_remove:
        del data[key]
    return data

def read_h5_example():
    datasets = read_h5_file('../datasets/archaeology', 'ark_20170927_152616_1.h5', ['roll', 'rollrate', 'pitch', 'imu', 'rgbframes', 'timestamp', 'timestampmeasured'], 500)
    for key, value in datasets.items():
        print('key: {}, value: {}'.format(key, value.dtype))

if __name__ == "__main__":
    read_h5_example()
