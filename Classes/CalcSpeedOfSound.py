'''
Author: OD
'''

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

def get_interp_time(t1, t2, y1, y2, threshold):
    return t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)

def calcSpeedOfSound(time_seconds, time_data, time_threshold, threshold_shift,
                     depth, amplitude_threshold, calculation_type='interp', 
                     displayBool = False, elements = [0], savePicBool = True):

    sound_speed_list = []

    for ele in elements:
        mask1 = (time_seconds > time_threshold) & (time_seconds < (time_threshold + threshold_shift))
        time_win1 = time_seconds[mask1]
        signal_win1 = time_data[ele][mask1]

        mask2 = time_seconds > (time_threshold + threshold_shift)
        time_win2 = time_seconds[mask2]
        signal_win2 = time_data[ele][mask2]

        if calculation_type == 'peak':
            idx1 = np.argmax(signal_win1)
            t1 = time_win1[idx1]
            y1 = signal_win1[idx1]

            idx2 = np.argmax(signal_win2)
            t2 = time_win2[idx2]
            y2 = signal_win2[idx2]

        elif calculation_type == 'sample':
            above1 = np.where(signal_win1 >= amplitude_threshold)[0]
            above2 = np.where(signal_win2 >= amplitude_threshold)[0]

            if len(above1) == 0 or len(above2) == 0:
                return 0

            idx1 = above1[0]
            t1 = time_win1[idx1]
            y1 = signal_win1[idx1]

            idx2 = above2[0]
            t2 = time_win2[idx2]
            y2 = signal_win2[idx2]

        elif calculation_type == 'interp':
            above1 = np.where(signal_win1 >= amplitude_threshold)[0]
            above2 = np.where(signal_win2 >= amplitude_threshold)[0]

            if len(above1) == 0 or len(above2) == 0:
                return 0

            idx1 = above1[0]
            if idx1 > 0:
                t1 = get_interp_time(time_win1[idx1-1], time_win1[idx1], 
                                     signal_win1[idx1-1], signal_win1[idx1], 
                                     amplitude_threshold)
            else:
                t1 = time_win1[idx1]
            y1 = amplitude_threshold

            idx2 = above2[0]
            if idx2 > 0:
                t2 = get_interp_time(time_win2[idx2-1], time_win2[idx2], 
                                     signal_win2[idx2-1], signal_win2[idx2], 
                                     amplitude_threshold)
            else:
                t2 = time_win2[idx2]
            y2 = amplitude_threshold

        sound_speed = 2 * (depth / (t2 - t1))
        sound_speed_list.append(sound_speed)
    
    if displayBool and len(sound_speed_list) > 0:
            
            # Overview Plot
            plt.figure(figsize=(10, 5))
            plt.plot(time_seconds, time_data[elements[-1]], c='b')
            if calculation_type != 'peak':
                plt.axhline(y=amplitude_threshold, color='g', linestyle='--')
            plt.scatter(t1, y1, c='crimson', zorder=5, label='Trigger 1')
            plt.scatter(t2, y2, c='goldenrod', zorder=5, label='Trigger 2')
            plt.xlim(0, 4.0e-5)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()
            if savePicBool:
                plt.savefig(f'Images/speedofsoundtrace.png')
            plt.show()


            plt.figure(figsize=(11, 5))
            plt.subplot(1, 2, 1)
            plt.plot(time_seconds, time_data[elements[-1]], c='b')
            if calculation_type != 'peak':
                plt.axhline(y=amplitude_threshold, color='g', linestyle='--')
            plt.scatter(t1, y1, c='crimson', zorder=5, label='Trigger 1')
            plt.xlim(t1 - 0.1e-5, t1 + 0.2e-5)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.plot(time_seconds, time_data[elements[-1]], c='b')
            if calculation_type != 'peak':
                plt.axhline(y=amplitude_threshold, color='g', linestyle='--')
            plt.scatter(t1, y1, c='crimson', zorder=5, label='Trigger 1')
            plt.scatter(t2, y2, c='goldenrod', zorder=5, label='Trigger 2')
            plt.xlim(t2 - 0.1e-5, t2 + 0.2e-5)
            plt.xlabel('Time [s]')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            if savePicBool:
                plt.savefig(f'Images/speedofsoundtriggers.png')
            plt.show()

    return np.mean(sound_speed_list)