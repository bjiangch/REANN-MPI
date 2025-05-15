#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "activate.h"


Relu_likeImpl::Relu_likeImpl(int neuron1, int neuron) {
  this->alpha=register_parameter("alpha",torch::ones({1,neuron}));
  this->beta =register_parameter("beta",torch::ones({1,neuron})/float(neuron1));
}

torch::Tensor Relu_likeImpl::forward(torch::Tensor &&x) {
  return this->alpha*torch::silu(x*this->beta);
}

/******************************************/
Tanh_likeImpl::Tanh_likeImpl(int neuron1, int neuron) {
  this->alpha=register_parameter("alpha",torch::ones({1,neuron})/torch::sqrt(torch::tensor({float(neuron1)},torch::kFloat)));
  this->beta =register_parameter("beta",torch::ones({1,neuron})/float(neuron1));
}

torch::Tensor Tanh_likeImpl::forward(torch::Tensor x) {
  return this->alpha*x/torch::sqrt(1.0+torch::square(x*this->beta));
}
