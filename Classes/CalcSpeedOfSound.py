import numpy as np
import matplotlib.pyplot as plt

def get_interp_time(t1, t2, y1, y2, threshold):
    return t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)

def calcSpeedOfSound(time_seconds, time_data, time_threshold, threshold_shift,
                     depth, amplitude_threshold, displayBool = False, elements = [0]):

    sound_speed_list = []

    for ele in elements:
        mask1 = (time_seconds > time_threshold) & (time_seconds < (time_threshold + threshold_shift))
        time_win1 = time_seconds[mask1]
        signal_win1 = time_data[ele][mask1]
        above_threshold1 = np.where(signal_win1 >= amplitude_threshold)[0]

        if len(above_threshold1) == 0:
            print(f"Threshold not met in first window for element {ele}")
            return 0

        idx1 = above_threshold1[0]
        
        if idx1 > 0:
            interp_time1 = get_interp_time(time_win1[idx1-1], time_win1[idx1], 
                                          signal_win1[idx1-1], signal_win1[idx1], 
                                          amplitude_threshold)
        else:
            interp_time1 = time_win1[idx1]

        mask2 = time_seconds > (time_threshold + threshold_shift)
        time_win2 = time_seconds[mask2]
        signal_win2 = time_data[ele][mask2]
        above_threshold2 = np.where(signal_win2 >= amplitude_threshold)[0]
        
        if len(above_threshold2) == 0:
            print(f"Threshold not met in second window for element {ele}")
            return 0

        idx2 = above_threshold2[0]

        if idx2 > 0:
            interp_time2 = get_interp_time(time_win2[idx2-1], time_win2[idx2], 
                                          signal_win2[idx2-1], signal_win2[idx2], 
                                          amplitude_threshold)
        else:
            interp_time2 = time_win2[idx2]

        sound_speed = 2 * (depth / (interp_time2 - interp_time1))
        sound_speed_list.append(sound_speed)
    
    if displayBool and len(sound_speed_list) > 0:
        plt.plot(time_seconds, time_data[elements[-1]])
        plt.axhline(y=amplitude_threshold, color='g', linestyle='--')
        plt.scatter(interp_time1, amplitude_threshold, c='r', zorder=5, label='Trigger 1')
        plt.scatter(interp_time2, amplitude_threshold, c='b', zorder=5, label='Trigger 2')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

        plt.plot(time_seconds, time_data[elements[-1]])
        plt.axhline(y=amplitude_threshold, color='g', linestyle='--')
        plt.scatter(interp_time1, amplitude_threshold, c='r', zorder=5, label='Trigger 1')
        plt.xlim(interp_time1 - 0.1e-5, interp_time1 + 0.2e-5)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

        plt.plot(time_seconds, time_data[elements[-1]])
        plt.axhline(y=amplitude_threshold, color='g', linestyle='--')
        plt.scatter(interp_time2, amplitude_threshold, c='b', zorder=5, label='Trigger 2')
        plt.xlim(interp_time2 - 0.1e-5, interp_time2 + 0.2e-5)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    return np.mean(sound_speed_list)