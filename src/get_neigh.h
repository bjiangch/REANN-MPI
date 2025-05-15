#pragma once

#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

class Neigh_ListImpl : public torch::nn::Module {
  public:
    Neigh_ListImpl(double cutoff,int nlinked);
    void forward(torch::Tensor &period_table,torch::Tensor &coordinates,torch::Tensor &cell,torch::Tensor &mass, torch::Tensor &neigh_list_p, torch::Tensor &shifts_p);
 
  private:
    double cutoff;
    double cell_list;
    int    nlinked;
    torch::Tensor linked;
};

TORCH_MODULE(Neigh_List);
