"""Single-volume run of the 3D engine with tuned grain/attenuation/dB-range.

Reuses every parameter defined in run_engine_3d.py and only overrides:
    * GRAIN_IMP_VAR   : 0.025 → 0.05     (wider impedance spread)
    * MATERIAL atten. : 1.0   → 0.3      (lower longitudinal attenuation)
    * TFM_DB_RANGE    : 40.0  → 20.0     (dB display range)

SCAN_ANGLES_DEG is forced to [0.0] so exactly one 3D volume is produced.
"""
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_engine_3d
from engine.materials import COPPER

run_engine_3d.GRAIN_IMP_VAR   = 0.05
run_engine_3d.TFM_DB_RANGE    = 20.0
run_engine_3d.MATERIAL        = replace(COPPER, attenuation_L=0.3, attenuation_S=1.2)
run_engine_3d.Z_MIN_MM        = 15.0
run_engine_3d.Z_MAX_MM        = 35.0
run_engine_3d.SCAN_ANGLES_DEG = [0.0]
run_engine_3d.OUTPUT_DIR      = HERE / "output" / "engine_3d_tuned"

if __name__ == "__main__":
    run_engine_3d.main()
