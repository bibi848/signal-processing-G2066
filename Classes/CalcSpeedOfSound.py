import numpy as np
import matplotlib.pyplot as plt

def calcSpeedOfSound(time_seconds, time_data, time_threshold, threshold_shift,
                     depth, displayBool = False, elements = [0]):
    '''
    time_seconds:    numpy array of times
    time_data:       numpy array of signal data for all elements
    time_threshold:  ignore all signals up till that time
    threshold_shift: ignore all signals up till time_threshold + threshold_shift
    depth:           characteristic length for measurement
    displayBool:     to display picture
    elements:        to average over the element indices
    '''

    sound_speed_list = []

    for ele in elements:
        mask = time_seconds > time_threshold
        time_after = time_seconds[mask]
        signal_after = time_data[ele][mask]

        max_idx1  = np.argmax(signal_after)
        max_time1 = time_after[max_idx1]
        max_val1  = signal_after[max_idx1]

        mask = time_seconds > (time_threshold + threshold_shift)
        time_after = time_seconds[mask]
        signal_after = time_data[ele][mask]

        max_idx2  = np.argmax(signal_after)
        max_time2 = time_after[max_idx2]
        max_val2  = signal_after[max_idx2]

        sound_speed = 2*(depth / (max_time2 - max_time1))

        sound_speed_list.append(sound_speed)
    
    if displayBool:
        plt.plot(time_seconds, time_data[elements[-1]])
        plt.scatter(max_time1, max_val1, c='r')
        plt.scatter(max_time2, max_val2, c='b')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

    return np.mean(sound_speed_list)