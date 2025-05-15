#ifdef PAIR_CLASS

PairStyle(reann_mpi,PairREANN_MPI)   // reann_mpi is the name in the input script

#else

#ifndef LMP_PAIR_REANN_MPI_H
#define LMP_PAIR_REANN_MPI_H

#include "pair.h"
#include <torch/torch.h>
#include <string>
#include "PES.h"
#include "comm.h"

namespace LAMMPS_NS {
  class PairREANN_MPI : public Pair { 
     public:
       PES module{nullptr};
       PairREANN_MPI(class LAMMPS *);
       virtual ~PairREANN_MPI();
       virtual void compute(int, int);
       virtual void init_style();
       virtual double init_one(int, int);
       virtual void settings(int, char **);
       virtual void coeff(int, char **);
     protected:
       virtual void allocate();
       virtual int select_gpu();
       double cutoff; 
       double cutoffsq;
       std::string datatype;
       torch::Dtype tensor_type=torch::kDouble;
       torch::TensorOptions option1=torch::TensorOptions().dtype(torch::kDouble);
       torch::TensorOptions option2=torch::TensorOptions().dtype(torch::kLong);
       torch::Tensor tensor_device=torch::empty(1);
       int id_gpu_using=-1;
  };
}

#endif
#endif
