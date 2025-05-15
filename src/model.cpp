#include "model.h"
using namespace torch::indexing;

/***************************************************/
NNModImpl::NNModImpl(
  int                      maxnumtype_p,
  int                      outputneuron,
  std::vector<std::string> atomtype,
  int                      nblock,
  std::vector<int>         nl,
  std::vector<double>      dropout_p,
  std::string              activate,
  double                   initpot,
  bool                     table_norm) {
  this->myatomtype=atomtype;

  this->initpot=register_buffer("initpot",torch::tensor({initpot},torch::kDouble).to(torch::kFloat));
  this->outputneuron=outputneuron;

  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> elemental_nets;

  double sumdrop=0;
  for (size_t i=0; i<dropout_p.size(); ++i) {
    sumdrop+=dropout_p[i];
  }
  {//same as with torch.no_grad() in pytorch
  torch::NoGradGuard no_grad;

  nl.push_back(nl[1]);
  int nhid=nl.size()-1;

  for (const auto& ele : atomtype) {
    torch::nn::Sequential modules;//{nullptr};
    linear=torch::nn::Linear(nl[0],nl[1]);
    torch::nn::init::xavier_uniform_(linear->weight);
    modules->push_back(linear);
    for (int iblock=0; iblock< nblock; ++iblock) {
      modules->push_back(ResBlock(nl,dropout_p,activate,table_norm)); 
    }
    if (activate=="Tanh_like") {
      modules->push_back(Tanh_like(nl[nhid-1],nl[nhid]));
    }else {
      modules->push_back(Relu_like(nl[nhid-1],nl[nhid]));
    }

    linear=torch::nn::Linear(nl[nhid],this->outputneuron);
    torch::nn::init::zeros_(linear->weight);
    if (abs(this->initpot[0].to(torch::kDouble).item<double>())>1e-6) torch::nn::init::zeros_(linear->bias);
    modules->push_back(linear);
    elemental_nets.insert(ele, modules.ptr());
  }
  }

  torch::nn::ModuleDict elemental_nets_tmp(elemental_nets);
  this->elemental_nets=register_module("elemental_nets",elemental_nets_tmp);
}

torch::Tensor NNModImpl::forward(
  torch::Tensor *density,
  torch::Tensor *species) {
  #if MODE_CUDA
  /*torch::Tensor*/ output = torch::zeros({(*density).sizes()[0],this->outputneuron},(*density).dtype()).to((*density).device());
  #else
  /*torch::Tensor*/ output = torch::zeros({(*density).sizes()[0],this->outputneuron},(*density).dtype());
  #endif

  int itype=0;

  /*torch::Tensor*/ species_ui8=(*species).to(torch::kUInt8);
  for (const auto& item : *elemental_nets) {
    auto m=item.value()->as<torch::nn::Sequential>();
    /*torch::Tensor*/ ele_index = torch::nonzero((species_ui8) == itype).view({-1});
    if (ele_index.sizes()[0] > 0) {
      /*torch::Tensor*/ ele_den = (*density).index({ele_index});
      output.index_put_({ele_index}, m->forward(ele_den));
    }
    ++itype;
  }

  return output;
}
