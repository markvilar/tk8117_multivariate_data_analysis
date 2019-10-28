#!/usr/bin/env Python

from utilities import read_h5_file, print_dict_entries

def example_read_h5_file():
    datasets = read_h5_file('../datasets/archaeology', 'ark_20170927_152616_1.h5', ['roll', 'rollrate', 'pitch', 'imu', 'rgbframes', 'timestamp', 'timestampmeasured'], 500)
    print_dict_entries(datasets)


