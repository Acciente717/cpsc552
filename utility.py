import numpy as np
import matplotlib.pyplot as plt

def MovingAveragePlot(input_list, window_size):
    plt.subplots(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(input_list)), input_list)

    plt.subplot(1, 2, 2)
    window = np.ones(int(window_size))/float(window_size)
    ave_values = np.convolve(input_list, window, 'valid')
    plt.plot(range(len(ave_values)), ave_values)

    plt.show()
