#!/bin/env bash
#SBATCH --job-name=multiprocess
#SBATCH --output=logs/multiprocess_%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=122

# #SBATCH --nodes=32


python - << EOF
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stats_grid

stats_grid.get_stats_on_grid(
    output="run03",
    passive_range=(0.01,0.15),
    npc_traverse_range=(1,1000),
    k_on_range=(0.001,5),
    nx=33,
    ny=33,
    n_passive=22,
    cargo_concentration_M=0.1e-6,
    Ran_concentration_M=20e-6,
    pickle_file="run03.pkl")
EOF
