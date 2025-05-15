#include "set_parameters.h"

set_parameters::set_parameters() {
  //std::cout<<" !!setting parameters!! "<<std::endl;
}
void set_parameters::set() {
  /// part of extern
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
  //extern double initpot;

  ///======================read input_nn==================================
  readinput read_input_nn("para/input_nn");

  start_table     = read_input_nn.get_string2int();       //1
  table_coor      = read_input_nn.get_string2int();       //0
  nl              = read_input_nn.get_string2int_arr();   //[128,128]
  nblock          = read_input_nn.get_string2int();       //1
  dropout_p       = read_input_nn.get_string2double_arr();//[0.0,0.0,0.0,0.0]
  table_init      = read_input_nn.get_string2int();       //0
  nkpoint         = read_input_nn.get_string2int();       //1
  Epoch           = read_input_nn.get_string2int();       //400000
  patience_epoch  = read_input_nn.get_string2int();       //50
  decay_factor    = read_input_nn.get_string2double();    //0.5
  start_lr        = read_input_nn.get_string2double();    //0.005
  end_lr          = read_input_nn.get_string2double();    //1e-5
  re_ceff         = read_input_nn.get_string2double();    //0
  ratio           = read_input_nn.get_string2double();    //0.9
  batchsize_train = read_input_nn.get_string2int();       //16
  batchsize_test  = read_input_nn.get_string2int();       //16
  e_ceff          = read_input_nn.get_string2double();    //0.1
  init_f          = read_input_nn.get_string2double();    //50
  final_f         = read_input_nn.get_string2double();    //0.5
  queue_size      = read_input_nn.get_string2int();       //10
  print_epoch     = read_input_nn.get_string2int();       //5
  table_norm      = read_input_nn.get_string2bool();      //True
  DDP_backend     = read_input_nn.get_string();           //"nccl"
  activate        = read_input_nn.get_string();           //"Relu_like"
  dtype           = read_input_nn.get_string();           //"float32"
  oc_nl           = read_input_nn.get_string2int_arr();   //[16,16]
  oc_nblock       = read_input_nn.get_string2int();       //1
  oc_dropout_p    = read_input_nn.get_string2double_arr();//[0,0,0,0]
  oc_activate     = read_input_nn.get_string();           //"Relu_like"
  oc_table_norm   = read_input_nn.get_string2bool();      //True
  oc_loop         = read_input_nn.get_string2int();       //0
  floder          = read_input_nn.get_string();           //"/data/H2O/"

  /// define the outputneuron of NN
  outputneuron=1;
  //
  readinput read_input_density("para/input_density");

  neigh_atoms     = read_input_density.get_string2int();       //150
  cutoff          = read_input_density.get_string2double();    //6.2
  nipsin          = read_input_density.get_string2int();       //2
  atomtype        = read_input_density.get_string2string_arr();//["O","H"]
  nwave           = read_input_density.get_string2int();       //8

  maxnumtype=atomtype.size();
  for (int i=0;i<maxnumtype;++i) {
    species_map[atomtype[i]]=i;
  }

  inta=torch::ones({maxnumtype,nwave});
  std::vector<torch::Tensor> rs_tmp;
  for (int itype=0;itype<maxnumtype;itype++) {
    rs_tmp.push_back(torch::linspace(0,cutoff,nwave));
  }
  rs=torch::stack(rs_tmp,0);
  
  ///======================for orbital================================
  nipsin+=1;
  norbit=nwave*(nwave+1)/2*nipsin;
  ///========================nn structure========================
  nl.insert(std::begin(nl), norbit);
  oc_nl.insert(std::begin(oc_nl), norbit);

}
