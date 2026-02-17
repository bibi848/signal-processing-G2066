#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import os
from scipy.signal.windows import tukey

# Point the script to the correct subfolder.
in_data_type = '1D Processed Data'
in_data_name = 'Cu Pure 2.5MHz 17022026'
cwd          = Path.cwd().parent
filter_data  = True
depth        = 27e-3 #mm

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
# Extracting speed of sound from data
for folder in sorted(p for p in Path(IN_DIR).iterdir() if p.is_dir()):
    print(f'Processing: {folder.name}')

    time_path = folder / 'time.csv'
    if time_path.exists():
        time_df = pd.read_csv(time_path)
        time_np = time_df['time_seconds'].to_numpy()
    else:
        print('Time Not Found')
        break    
    
    h5_path = folder / 'time_data.h5'
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            time_data = np.array(f["time_data"])
    else:
        print('Time Data Not Found')
        break
    
    # Find backwall reflection
    max_idx = np.where(time_data[0] == max(time_data[0]))
    max_time = time_np[max_idx]

    plt.plot(time_np, time_data[0])
    plt.scatter(max_time, max(time_data[0]), c='r')

    sound_speed = 2*(depth / max_time)

    print(f'Speed of Sound: {sound_speed[0]:.2f} m/s')
    print()