#!/bin/bash
#SBATCH -J test
#SBATCH -p intel96
#SBATCH -N 1
#SBATCH --cpus-per-task=96

export OMP_NUM_THREADS=1
path="/public/home/xjf/lammps-10Feb21/build-reann-mpi/"
mpirun -np 8 $path/lmp_mpi -in in.lmp > out-lmp
