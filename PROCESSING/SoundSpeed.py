#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import os
from scipy.signal.windows import tukey
import sys

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from Classes.CalcSpeedOfSound import calcSpeedOfSound

# Point the script to the correct subfolder.
in_data_type = '1D Processed Data'
in_data_name = 'Al Pure 10MHz Ex 09032026'
cwd          = Path.cwd().parent
filter_data  = True
depth        = 53.3e-3 #mm
t_threshold  = 1e-5
threshold_shift = 1e-5

# Input and Output paths.
if filter_data:
    IN_DIR = os.path.join(cwd, 'DATA', in_data_type, (in_data_name + ' Filtered'))
else:
    IN_DIR  = os.path.join(cwd, 'DATA', in_data_type, in_data_name)

# Find all files in directory which are . files. 
data_folders = sorted([
    p.name for p in Path(IN_DIR).iterdir()
    if p.is_dir()
])

print('Folders available in directory:')
print(data_folders)
print()

#%%
# Keeping necessary files
data_folders = [x for x in data_folders if 'Speed of Sound' in x]
print('Relevant Data')
print(data_folders)
print()

#%%
# Extracting speed of sound from data
for folder in data_folders:
    # Locate Data
    loc = os.path.join(IN_DIR, folder)

    time_path = loc + '/time.csv'
    h5_path   = loc + '/time_data.h5'

    time_df = pd.read_csv(time_path)
    time_np = time_df['time_seconds'].to_numpy()

    with h5py.File(h5_path, 'r') as f:
        time_data = np.array(f["time_data"])

    # Find backwall reflection 1
    mask = time_np > t_threshold
    time_after = time_np[mask]
    signal_after = time_data[0][mask]

    max_idx1  = np.argmax(signal_after)
    max_time1 = time_after[max_idx1]
    max_val1  = signal_after[max_idx1]

    plt.plot(time_np, time_data[0])
    plt.scatter(max_time1, max_val1, c='r')

    # Find backwall reflection 1
    mask = time_np > (t_threshold + threshold_shift)
    time_after = time_np[mask]
    signal_after = time_data[0][mask]

    max_idx2  = np.argmax(signal_after)
    max_time2 = time_after[max_idx2]
    max_val2  = signal_after[max_idx2]

    plt.scatter(max_time2, max_val2, c='b')

    sound_speed = 2*(depth / (max_time2 - max_time1))

    print(f'Speed of Sound: {sound_speed:.2f} m/s')
    print()
    break

#%%
plt.plot(time_np, time_data[0])
plt.scatter(max_time1, max_val1, c='r')
plt.scatter(max_time2, max_val2, c='r')
plt.show()

plt.plot(time_np, time_data[0])
plt.scatter(max_time1, max_val1, c='r')
plt.xlim(1.6e-5, 2e-5)
plt.show()

plt.plot(time_np, time_data[0])
plt.scatter(max_time2, max_val2, c='r')
plt.xlim(3.2e-5, 3.6e-5)
plt.show()

#%%
# Function Test

sound_speed = calcSpeedOfSound(time_np, time_data, t_threshold, threshold_shift,
                               depth, displayBool=True, elements=[0,1,2])

#%%