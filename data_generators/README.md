### data_generators/

Script to generate or transform the synthetic dataset for the Student–Project Allocation (SPA) problem.

### File
`generate_spa_dataset.py`-Generates a synthetic dataset and saves all outputs to `/data/`
`README.md` - this file

### Usage
```bash
python data_generators/generate_spa_dataset.py
```
Outputs Excel to `data/`, which is then used by rest of the scripts.

### Note
This script is self-contained and does not read from any external config files.



