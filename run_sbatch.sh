#!/bin/bash
#SBATCH -N 1
#SBATCH -J EigenMaps
#SBATCH -t 2-00:00:00
#SBATCH -p batch
#SBATCH --mem 100000

cd $HOME/Eigenmaps/
mycommand="python -W ignore gem_eigenmaps.py --dataset paper-cite --dimension 256"
# mycommand="python -W ignore cost_eigenmaps_unconstrained.py --dataset texas --dimension 128 --cost num_lap"
echo $mycommand
$mycommand
