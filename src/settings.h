#pragma once

#include <vector>
#include <unordered_map>
#include <torch/torch.h>

#ifndef _SETTINGS_H_
#define _SETTINGS_H_
//namespace reann{
//default settings of input_nn
  extern int                 start_table    ;
  extern int                 table_coor     ;
  // NN structure
  extern std::vector<int>    nl             ;
  //nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
  extern int                 nblock         ;
  //dropout probability for each hidden layer
  extern std::vector<double> dropout_p      ;
  extern int                 table_init     ;
  extern int                 nkpoint        ;
  extern int                 Epoch          ;
  extern int                 patience_epoch ;
  extern double              decay_factor   ;
  extern double              start_lr       ;
  extern double              end_lr         ;
  extern double              re_ceff        ;
  extern double              ratio          ;
  extern int                 batchsize_train;
  extern int                 batchsize_test ;
  extern double              e_ceff         ;
  extern double              init_f         ;
  extern double              final_f        ;
  extern int                 queue_size     ;
  extern int                 print_epoch    ;
  extern bool                table_norm     ;
  extern std::string         DDP_backend    ;
  extern std::string         activate       ;
  extern std::string         dtype          ;
  //neural network architecture   
  extern std::vector<int>    oc_nl          ;
  extern int                 oc_nblock      ;
  extern std::vector<double> oc_dropout_p   ;
  extern std::string         oc_activate    ;
  extern bool                oc_table_norm  ;
  extern int                 oc_loop        ;
  extern std::string         floder         ;

//default settings of input_density
  extern int                              neigh_atoms    ;
  extern double                           cutoff         ;
  extern int                              nipsin         ;
  extern std::vector<std::string>         atomtype       ;
  extern std::unordered_map<std::string, int> species_map;
  extern int                              nwave          ;
//
  extern int                              norbit;
  extern int                              outputneuron;
  extern int                              maxnumtype;
  extern torch::Tensor inta;
  extern torch::Tensor rs;
  // chk initpot as it may come from .pt
  extern double initpot;
  //
  extern torch::DeviceType tensor_device;
  extern torch::Dtype tensor_type;
  extern torch::TensorOptions tensor_option;
//}
#endif
