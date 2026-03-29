#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import os
import sys

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from Classes.CalcSpeedOfSound import calcSpeedOfSound

# Point the script to the correct subfolder.
in_data_type = '2D Processed Data'
in_data_name = 'Cu Pure 7.5MHz Ex 16032026'
cwd          = Path.cwd().parent
filter_data  = True
depth        = 50e-3 #mm

t_threshold         = 1e-5
threshold_shift     = 2e-5
amplitude_threshold = 0.025
elements            = [4]

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

speed_sound_list = []

for folder in data_folders:
    # Locate Data
    loc = os.path.join(IN_DIR, folder)

    time_path = loc + '/time.csv'
    h5_path   = loc + '/time_data.h5'

    time_df = pd.read_csv(time_path)
    time_np = time_df['time_seconds'].to_numpy()

    with h5py.File(h5_path, 'r') as f:
        time_data = np.array(f["time_data"])
    
    sound_speed = calcSpeedOfSound(time_np, time_data, t_threshold, threshold_shift,
                               depth, amplitude_threshold, displayBool=True, elements=elements)
    
    print('Speed of Sound:', sound_speed)
    speed_sound_list.append(sound_speed)

print()
print('Speed of Sound:', np.mean(speed_sound_list))

#%%