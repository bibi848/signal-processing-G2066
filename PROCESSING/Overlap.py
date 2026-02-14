#%%
# Importing Functions and Defining Correct Path
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os

from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from itertools import combinations


# Point the script to the correct subfolder.
in_data_type = '1D TFM Data'
in_data_name = 'Al Pure 10MHz 12022026'
cwd          = Path.cwd().parent
save_picture = True
filter_data  = True
grouping     = 3

cmap1 = 'viridis'
cmap2 = 'hot'

# Input and Output paths.
if filter_data:
    IN_DIR = os.path.join(cwd, 'DATA', in_data_type, (in_data_name + ' Filtered'))
else:
    IN_DIR  = os.path.join(cwd, 'DATA', in_data_type, in_data_name)

# Find all files in directory which are .mat files. 
image_files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith(".png")
    and os.path.isfile(os.path.join(IN_DIR, f))
]
image_files = np.sort(image_files)

print('Files available in directory:')
print(image_files)
print()

#%%
# Function Definition
def load_image(path):
    img = imread(path, as_gray=True)
    img = img.astype(np.float64)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def ssim_percentage(img1, img2):
    score, _ = ssim(img1, img2, data_range=1.0, full=True)
    return score * 100


#%%
# Similarity Comparison
results = []

for i in range(0, len(image_files), grouping):
    group_files = image_files[i:i+grouping]

    images = [
        load_image(os.path.join(IN_DIR, f))
        for f in group_files
    ]

    for (idx1, img1), (idx2, img2) in combinations(enumerate(images), 2):
        similarity = ssim_percentage(img1, img2)

        results.append({
            'group': i // grouping + 1,
            'image_1': group_files[idx1],
            'image_2': group_files[idx2],
            'ssim_percent': similarity
        })

results_df = pd.DataFrame(results)
print(results_df)

#%%
# Image Statistics
mean_images = {}

for i in range(0, len(image_files), grouping):

    if i > 8:
        break

    group_files = image_files[i:i+grouping]

    images = np.stack([
        load_image(os.path.join(IN_DIR, f))
        for f in group_files
    ], axis=0)

    mean_img = np.mean(images, axis=0)
    group_id = i // grouping + 1
    mean_images[group_id] = mean_img

    # Displaying Data

    fig, ax1 = plt.subplots()
    ax1.imshow(mean_img, cmap=cmap1)
    ax1.set_title(f'Mean Group {group_id}')

    if save_picture:
        out_path = os.path.join(IN_DIR, f'Averaged_Group_{group_id}.png')
        plt.imsave(out_path,
                   mean_img,
                   cmap=cmap1,
                   vmin=0.0,
                   vmax=1.0)

    plt.show()
    plt.close(fig)

    std_img = np.std(images, axis=0)

    fig, ax2 = plt.subplots()
    ax2.imshow(std_img, cmap=cmap2)
    ax2.set_title(f'Standard Deviation Group {group_id}')
    plt.show()
    plt.close(fig)

    peak_amp = np.max(mean_img)
    db_threshold = peak_amp * 10**(-8/20)

    fig, ax = plt.subplots()
    ax.imshow(mean_img, cmap=cmap1, vmin=0.0, vmax=1.0)
    ax.contour(mean_img,
               levels=[db_threshold],
               colors='red',
               linewidths=1)
    ax.axis('off')

    if save_picture:
        out_path = os.path.join(IN_DIR, f'Contoured_Group_{group_id}.png')
        fig.savefig(out_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0)

    plt.show()
    plt.close(fig)

#%%