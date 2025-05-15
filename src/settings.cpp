#include "settings.h"

//namespace reann{
//default settings of input_nn
  int                 start_table     = 1;
  int                 table_coor      = 0;
  // NN structure
  std::vector<int>    nl              = {128,128};
  //nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
  int                 nblock          = 1;
  //dropout probability for each hidden layer
  std::vector<double> dropout_p       = {0.0,0.0,0.0,0.0};
  int                 table_init      = 0;
  int                 nkpoint         = 1;
  int                 Epoch           = 400000;
  int                 patience_epoch  = 50;
  double              decay_factor    = 0.5;
  double              start_lr        = 0.005;
  double              end_lr          = 1e-5;
  double              re_ceff         = 0;
  double              ratio           = 0.9;
  int                 batchsize_train = 16;
  int                 batchsize_test  = 16;
  double              e_ceff          = 0.1;
  double              init_f          = 50;
  double              final_f         = 0.5;
  int                 queue_size      = 10;
  int                 print_epoch     = 5;
  bool                table_norm      = true;
  std::string         DDP_backend     = "nccl";
  std::string         activate        = "Relu_like";
  std::string         dtype           = "float32";
  //neural network architecture   
  std::vector<int>    oc_nl           = {16,16};
  int                 oc_nblock       = 1;
  std::vector<double> oc_dropout_p    = {0,0,0,0};
  std::string         oc_activate     = "Relu_like";
  bool                oc_table_norm   = true;
  int                 oc_loop         = 0;
  std::string         floder          = "/data/H2O/";

//default settings of input_density
  int                              neigh_atoms     = 150;
  double                           cutoff          = 6.2;
  int                              nipsin          = 2;
  std::vector<std::string>         atomtype        = {"O","H"};
  std::unordered_map<std::string, int> species_map;
  int                              nwave           = 8;
//
  int                              norbit;
  int                              outputneuron;
  int                              maxnumtype;
  torch::Tensor inta;
  torch::Tensor rs;
  // chk initpot as it may come from .pt
  double initpot;
  //
  torch::DeviceType tensor_device;
  torch::Dtype tensor_type;
  torch::TensorOptions tensor_option;
  
//}
