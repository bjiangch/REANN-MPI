export OMP_NUM_THREADS=1
path="/public/home/xjf/proj-22-0117/lammps/test_mpi/lmp/lammps-10Feb21/tt/lammps-10Feb21/build-reann-mpi/"
mpirun -np 8 $path/lmp_mpi -in in.lmp > out-lmp
