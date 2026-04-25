# Sensing, Imaging and Signal Processing G2066

Welcome to the 2026 Group Industrial Project GitHub repository for Sensing, Imaging and Signal Processing. This repository will be used to collaborate on code written, and serve as a record of how the project grows over time. 

## Repository Summary
Folders:
* **3D RECONSTRUCTION**: Contains the frameworks for combining 2D TFM slices into full 3D volumes. 
* **CPP**: Contains the frameworks for accelerated TFM calculations, using both OpenMP for CPU multi-threading and HIP/ROCm for GPU leveraged computing based on a C++ architecture. A full explanation is included in the folder itself. 
* **Classes**: Contains functions for calculating the speed of sound, filtering, stitching in 2D or 3D and a python TFM implementation.
* **DATA**: Contains raw and processed data, as well as finalised TFM images used in stitching operations.
* **PROCESSING**: Contains experimental procedures as part of the project, function demos (filtering, speed of sound calculations...etc) and a batch GPU TFM implementation. More information is included in the folder. 
* **STITCHING**: Contains programs exploring different stitching algorithms applied to the TFM images.
* **SYNTHETIC DATA**: Contains programs exploring the synthetic data generation pipeline. 

Files:
* `.gitignore`: Used to ignore files or folders from commits, such as propriatary datasets.
* `CMakeLists.txt`: Detects whether the user is using Windows, Linux or Mac, to set up the OpenMP and HIP/ROCm environments. More details in the CPP folder. 
* `Display3DData.py`: Uses the napari library to view the 3D TFM data.
* `Imaging.py`: Connects the processed data to the appropriate TFM calculation, which can include a Python-based calculation held in the folder **Classes**, an OpenMP accelerated C++ function contained in the **CPP** folder, or a GPU leveraged computation held in the **CPP** folder. This is for producing 2D TFM images.
* `Imaging3D.py`: Connects the processed data to the appropriate TFM calculation, which includes an OpenMP accelerated C++ function in the **CPP** folder, or a GPU leveraged computation held in the **CPP** folder. This is for producing 3D TFM images.
* `MATtoCSV.py`: Converts the raw data collected in the UNDT lab (.mat files) to processed and readable data (.xlsx files) to then be imaged by `Imaging.py` or `Imaging3D.py`. 
* `README.md`: The file you are reading right now!
* `requirements.txt`: Contains all libraries used in the project.

Folders that contain a lot of information will themselves have a `README.md` file available. 

---

### Citation

Djuric, O., Bruce-Gardyne, O., & Tabet, T. (2026).
signal-processing-G2066. GitHub.
https://github.com/bibi848/signal-processing-G2066
