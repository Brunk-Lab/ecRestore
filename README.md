# ecRestore
This repository has the scripts for stochastic simulation of ecDNA re-distribution in the FACS-sorted NCIH2170 subpopulations.

## Repository Contents

- `Simulation_Stable_Recenter.py` – Includes the Gillespie Algorithm that is used to model stochastic cell division and death
- `Genetic_Stable_Recenter.py` – Includes the Genetic Algorithm that is used to optimize parameters for cell death, division, and split probabilities
- `run_genetic_recenter.sh` – Bash script to execute both Python scripts.

