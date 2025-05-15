#include "ResBlock.h"
using namespace torch::indexing;

ResBlockImpl::ResBlockImpl(
    std::vector<int>    nl,
    std::vector<double> dropout_p,
    std::string         activate,
    bool                table_norm) {

  int nhid=nl.size()-1;
  double sumdrop=0;
  for (size_t i=0; i<dropout_p.size(); ++i) {
    sumdrop+=dropout_p[i];
  }

  resblock=torch::nn::Sequential();
  for (int i=1; i<nhid; ++i) {
    if (activate=="Tanh_like") {
      resblock->push_back(Tanh_like(nl[i-1],nl[i]));
    }else {
      resblock->push_back(Relu_like(nl[i-1],nl[i]));
    }
    if (table_norm) {
      std::vector<long> normalized_shape={nl[i]};
      resblock->push_back(torch::nn::LayerNorm(normalized_shape));
    }
    if (sumdrop>=0.0001) resblock->push_back(torch::nn::Dropout(dropout_p[i-1]));
    linear=torch::nn::Linear(nl[i],nl[i+1]);
    if (i==nhid-1) {
      torch::nn::init::zeros_(linear->weight);
    }else {
      torch::nn::init::xavier_uniform_(linear->weight);
    }
    torch::nn::init::zeros_(linear->bias);
    resblock->push_back(linear);
  }
  this->resblock=register_module("resblock",resblock);
  
}

torch::Tensor ResBlockImpl::forward(torch::Tensor &&x) {
  return this->resblock->forward(x)+x;
}
