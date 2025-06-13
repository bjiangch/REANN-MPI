#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "activate.h"

class ResBlockImpl : public torch::nn::Module {
  public:
    ResBlockImpl(std::vector<int> nl, std::vector<double> dropout_p, std::string activate, bool table_norm);
 
    torch::Tensor forward(torch::Tensor &&x);
 
  private:
    torch::nn::Sequential resblock{nullptr};
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(ResBlock);
