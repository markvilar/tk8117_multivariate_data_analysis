#!/usr/bin/env Python
import h5py
import numpy as np

def main():
    data_folder = '../data'
    file_name = 'Ark_20170630_081758_1.h5'
    file_path = data_folder + '/' + file_name
    with h5py.File(file_path, 'r') as data_file:
        for file_key in data_file.keys():
            group = data_file[file_key]
            print("key: ", file_key)
            for group_key in group.keys():
                print(group_key)

if __name__ == "__main__":
    main()
