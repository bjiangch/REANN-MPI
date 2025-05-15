#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "omp.h"

#include "activate.h"
#include "ResBlock.h"

class NNModImpl : public torch::nn::Module {
  public:
    NNModImpl(
      int                      maxnumtype_p,
      int                      outputneuron,
      std::vector<std::string> atomtype,
      int                      nblock,
      std::vector<int>         nl,
      std::vector<double>      dropout_p,
      std::string              activate,
      double                   initpot,
      bool                     table_norm);
      torch::Tensor forward(torch::Tensor *density, torch::Tensor *species);
      torch::Tensor initpot;
      torch::Tensor output,mask,ele_index,ele_den,species_ui8;

  private:
    int outputneuron;
    torch::nn::Linear linear{nullptr};
    torch::nn::ModuleDict elemental_nets;
    std::vector<std::string> myatomtype;

};

TORCH_MODULE(NNMod);
