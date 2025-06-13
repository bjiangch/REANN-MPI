#pragma once
#include <torch/torch.h>
#include "set_parameters.h"
#include "activate.h"
#include "model.h"
#include "density.h"
#include "get_neigh.h"
#include <iostream>
#include <fstream> // added for cout to file

using LAMMPS_NS::LAMMPS;


class PESImpl : public torch::nn::Module {
  public:
    PESImpl(int nlinked);
    LAMMPS *lmp;
    std::unordered_map<int, int>atom_map;
int count_j;


void forward(torch::Tensor &cart, torch::Tensor &atom_index, torch::Tensor &local_species, torch::Tensor &neigh_species, torch::Tensor &neigh_list, torch::Tensor &energy, torch::Tensor &force, torch::Tensor &output);
    GetDensity density{nullptr};
 
  private:
    double     cutoff;
    NNMod      nnmod{nullptr};
   int nwave;
   torch::Tensor g_dT_G_sum;
   torch::Tensor g_dT_cT_sum;
   torch::Tensor g_dT_cT, g_dT_G;
};

TORCH_MODULE(PES);
