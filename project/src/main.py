import numpy as np
import matplotlib.pyplot as plt

from utilities import read_h5_file

def main():
    datasets = read_h5_file('../data', 'Ark_20170630_085421_1.h5', ['xacceleration', 'yacceleration', 'zacceleration'], 500)
    plt.plot(datasets['/rawdata/navigation/imu/ZAcceleration'])

if __name__ == '__main__':
    main()
