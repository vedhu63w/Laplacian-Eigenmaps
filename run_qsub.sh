#PBS -N Emb_gen
#PBS -l walltime=72:00:00
#PBS -l nodes=1
#PBS -l mem=100GB
#PBS -j oe
#PBS -m abe
#PBS -A PAA0205

source ~/.bashrc
cd /users/PAS1421/osu10530/EigenMaps

python -W ignore gem_eigenmaps.py --dataset microsoft --dimension 128
