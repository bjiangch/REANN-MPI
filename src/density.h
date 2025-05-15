#pragma once

#include "pointers.h"  // IWYU pragma: export
#include "lmptype.h"
#include "lammps.h"  
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream> // added for cout to file
#include <cmath>
#include "model.h"
using LAMMPS_NS::LAMMPS;

class GetDensityImpl : public torch::nn::Module {
  public:
    GetDensityImpl(torch::Tensor rs, torch::Tensor inta, double cutoff, int nipsin, int norbit, torch::nn::Sequential &ocmod_list);
    void forward(torch::Tensor &cart, torch::Tensor &atom_index, torch::Tensor &local_species, torch::Tensor &neigh_species, torch::Tensor &neigh_list,/* torch::Tensor &density_p*/
   std::vector<torch::Tensor> &density_list,
   std::vector<torch::Tensor> &orb_coeff_list,
   std::vector<torch::Tensor> &orb_coeff_m_list,
   std::vector<torch::Tensor> &orbital_list
);
    torch::Tensor         cutoff;
    torch::Tensor         pi_cut;
    torch::Tensor         nipsin;
    torch::Tensor         index_para;

    LAMMPS *lmp;

int count_j;

    std::unordered_map<int, int>atom_map;


  private:
    torch::Tensor         rs;
    torch::Tensor         inta;
    torch::Tensor         params;
    torch::Tensor         hyper;
    //int                   norbit;
    torch::Tensor mask, ele_index, part_radial;
    //torch::Tensor expandpara,worbital,sum_worbital, hyper_worbital;
    torch::nn::ModuleDict ocmod;
    //torch::OrderedDict<const std::string, std::shared_ptr<torch::nn::Module>> ocmod;
    //torch::OrderedDict<std::string, torch::nn::Sequential> ocmod;

    /*void*/torch::Tensor gaussian(torch::Tensor &distances, torch::Tensor &species_/*, torch::Tensor &radial*/);
    /*void*/torch::Tensor cutoff_cosine(torch::Tensor &distances/*, torch::Tensor &f_cut*/);
    /*void*/torch::Tensor angular(torch::Tensor &dist_vec, torch::Tensor /*&*/f_cut/*, torch::Tensor &angular_p*/);
    void obtain_orb_coeff(int &&iteration, int &numatom, torch::Tensor &orbital, torch::Tensor &/*&*/center_list, torch::Tensor &neigh_list, torch::Tensor &orb_coeff, torch::Tensor &density_p);
};

TORCH_MODULE(GetDensity);
