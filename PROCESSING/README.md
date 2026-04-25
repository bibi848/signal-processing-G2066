# PROCESSING

Folder used to contain scripts written by OD. 

## Folder Summary
Folders:
* **Images**: Various images produced for the report. 

Files:
* `Aluminium Experiment.py`: Experimental procedure done on an aluminium sample imaged with a 1D 10MHz 128 element array using a 3D printed guide. More details in the script.
* `Angular Filter.py`: Test script showing no benefit to using an angular filter to limit pixels to only be imaged by certain array elements.
* `Copper Experiment 15042026.py`: Experimental procedure done on a copper sample imaged with a 2D 7.5MHz 128 element array using a 3D printed guide. More details in the script.
* `Copper Experiment 22042026.py`: Experimental procedure done on a copper sample imaged with a 2D 7.5MHz 128 element array using a 3D printed guide. More details in the script.
* `Filter_Demo.py`: Small demo showing the use of the filter function visually on element a-scans.
* `Imaging_ultra.py`: Test script investigating the use of GPU batching to reduce kernel overhead when imaging using the GPU.
* `Overlap.py`: Test script investigating if taking the average over different scans is worth it when imaging microstructural backscatter. Results found that it was negligible compared to just using 64 averages on the MicroPulse FMC machine in the lab.
* `SoundSpeed.py`: Small demo showing the use of the speed of sound function to visually show how the speed of sound is calculated in different materials.

---
