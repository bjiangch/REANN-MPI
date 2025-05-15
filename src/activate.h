#pragma once
#include <torch/torch.h>

class Relu_likeImpl : public torch::nn::Module {
  public:
    Relu_likeImpl(int neuron1,int neuron);
    torch::Tensor forward(torch::Tensor &&x);
    torch::Tensor alpha, beta;
};

TORCH_MODULE(Relu_like);

///-------------------------------------------------

class Tanh_likeImpl : public torch::nn::Module {
  public:
    Tanh_likeImpl(int neuron1,int neuron);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor alpha, beta;
};

TORCH_MODULE(Tanh_like);

