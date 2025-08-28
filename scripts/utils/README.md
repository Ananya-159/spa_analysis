# Scripts/Utils (Legacy)

This folder contains **legacy helper scripts** used only during early smoke tests and Phase-2 development.  
Example: `logger.py` was used to write lightweight CSV logs (`phase2_checks.csv`).

## Purpose
- Provided CSV logging (`log_breakdown`) and file hashing (`file_sha256`).
- Supported debugging and quick pipeline checks in Phase-2.
- Outputs were written to `results/phase2_checks.csv`.

## Current Status
- These utilities are **not used** in the final GA/HC/HH/Second-round pipeline.  
- Kept here **for provenance only**, to preserve the development/debugging history.

## Notes
- If you re-run old smoke tests, `logger.py` may still be required.  
- For all production runs, use the main code in `core/` (which now includes all required utilities).
