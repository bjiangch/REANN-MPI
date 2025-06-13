# REANN-MPI
REANN-MPI is a MPI parallelized implementation of our REANN model, which is rewritten in C++ with LibTorch and supports MPI parallel  MD simulations in LAMMPS.


- [Installation&compilation](#Installation&compilation)

- [Usage](#Usage)

- [Reference](#Reference)

## Installation&compilation

### Pre-installation requirement
Before compilation, you need to make sure softwares below has been installed (versions in the bracket have been tested).

- GCC (9.1.0)
- Intel MPI_CXX (3.1)
- OpenMP_CXX (4.5)
- cuda (11.3) and cudnn, only if using GPU

### Download LAMMPS and Libtorch

You need to download LAMMPS from the LAMMPS website. Due to the update of LAMMPS, we recommend using the 10Feb2021 version of LAMMPS. Code supporting newer versions of LAMMPS will be released soon.

```bash
#download LAMMPS
wget https://download.lammps.org/tars/lammps-10Feb2021.tar.gz
```

You also need to download the Torch C++ frontend (Libtorch) from the PyTorch website consistent with the environments in the targeted device (CPU or GPU with different versions of CUDA and cudnn). In addition, the downloaded Libtorch package needs to be uncompress in the  “pkg” folder, which is located in the same directory with the src of  LAMMPS, for example, "pkg/libtorch-1.12.1-cpu". 

```bash
#download LibTorch-CPU
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip
#download LibTorch-GPU if using GPU
wget https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.12.1%2Bcu116.zip
```
### Patch LAMMPS with REANN-MPI code

From the `REANN-MPI` directory, run

```bash
sh patch_reann2lmp.sh path_lammps
```

Then `path_lammps` contains the directories as below,

```bash
[xjf]$ cd lammps-10Feb21/
[xjf]$ ls
bench  build-reann-mpi  cmake  doc  examples  fortran  lib  LICENSE  pkg  potentials  python  README  src  tools  unittest
```

### Configure LAMMPS with CMake

Change the paths of LibTorch (e.g., `${LAMMPS_DIR}/pkg/libtorch-1.12.1-cpu`) and CUDA/CUDNN-related libraries (e.g., `/usr/local/cuda-12.3/`,if using GPU version LibTorch) in cmake/CMakeLists.txt to the paths you installed.

```CMake
if(BUILD_CUDA) #Using GPU
  SET(CMAKE_PREFIX_PATH     ${LAMMPS_DIR}/pkg/libtorch-1.12.1-gpu)   # add to link the libtorch-gpu
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.3/")
  set(CUDNN_LIBRARY "/usr/local/cuda-12.3/lib64/")
  set(CUDNN_INCLUDE_DIR "/usr/local/cuda-12.3/include/")
  message(STATUS "Building with libtorch ON GPU
    -- libtorch        :    ${CMAKE_PREFIX_PATH}
    -- CUDA  ROOT      :    ${CUDA_TOOLKIT_ROOT_DIR}
    -- CUDNN libraries :    ${CUDNN_LIBRARY}
    -- CUDNN includes  :    ${CUDNN_INCLUDE_DIR};
  ")
  add_definitions(-D MODE_CUDA=true)
else() #Using CPU
  SET(CMAKE_PREFIX_PATH     ${LAMMPS_DIR}/pkg/libtorch-1.12.1-cpu)   # add to link the libtorch-cpu
  message(STATUS "Building with libtorch ON CPU
    -- Libtorch        :    ${CMAKE_PREFIX_PATH}
  ")
  add_definitions(-D MODE_CUDA=false)
endif()
```

### Compilation 
After preparing the software and libraries, you can compile LAMMPS with REANN-MPI as below,

```bash
cd build-reann-mpi
sh build.cpu.sh  #if using CPU
#sh build.gpu.sh #if using GPU
#waiting for a monment of building, if no error report, then
make
```
For LAMMPS with REANN-MPI running on GPU, the script above should be run with `sh build.gpu.sh` on a computational node with one GPU and software/libraries mentioned above.



## Usage
There is an example in the example/ folder, along with the corresponding REANN model, LAMMPS input files, and running script.

### LAMMPS input script 

Define the pair style in the LAMMPS input script,
```bash
comm_style brickreann
processors px py pz
...
pair_style reann_mpi
pair_coeff * * cutoff_radius datatype
```
- Define the comm_style brickreann for reann_mpi.
- The command processors specifies how processors are mapped as a regular 3d grid to the global simulation box, details can be seen in [the document of LAMMPS ](https://docs.lammps.org/processors.html).
- Define the pair style `reann_mpi`
- Define the parameters for `reann_mpi`.  cutoff_radius is that used in training REANN model, and datatype is **float** (or **double**), which specifies the file `REANN_LAMMPS_FLOAT.pt` (or `REANN_LAMMPS_DOUBLE.pt`) to load the model.

### Running

The basic command to run LAMMPS is as below, in which numprocesses=px\*py\*pz is the total number of processes used in the simulation,

```bash
mpirun -np numprocesses path_of_lammps/lmp_mpi -in in.lmp
```

For example, if you run one simulation with `processors 2 2 2`, numprocesses shoule be 8. The allocation of corresponding computing resources is usually done by a job system such as Slurm. We also provide an example `job-md.slurm`.




## Reference
J. Xia, B. Jiang Efficient Parallelization of Message Passing Neural Network Potentials for Large-scale Molecular Dynamics, arXiv preprint arXiv:2505.06711, 2025.
