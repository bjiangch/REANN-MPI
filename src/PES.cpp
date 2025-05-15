#include "PES.h"
#include "comm.h"
#include "atom.h"
#include <mpi.h>
using namespace torch::indexing;
using namespace LAMMPS_NS;

PESImpl::PESImpl(int nlinked) {

  //extern int                 table_coor     ;
  extern std::vector<int>    nl             ;
  extern int                 nblock         ;
  extern std::vector<double> dropout_p      ;
  extern bool                table_norm     ;
  extern std::string         DDP_backend    ;
  extern std::string         activate       ;
  extern std::string         dtype          ;
  extern std::vector<int>    oc_nl          ;
  extern int                 oc_nblock      ;
  extern std::vector<double> oc_dropout_p   ;
  extern std::string         oc_activate    ;
  extern bool                oc_table_norm  ;
  extern int                 oc_loop        ;
  //
  //extern int                              neigh_atoms    ;
  extern double                           cutoff         ;
  extern int                              nipsin         ;
  extern std::vector<std::string>         atomtype       ;
  //extern std::unordered_map<std::string, int> species_map;
  extern int                              nwave          ;
  extern int                              norbit;
  extern int                              outputneuron;
  extern int                              maxnumtype;
  extern torch::Tensor                    inta;
  extern torch::Tensor                    rs;
  extern double                           initpot;
  
  set_parameters set_params;
  set_params.set();
  ///================read the periodic boundary condition, element and mass=========
  this->cutoff=cutoff;
  this->nwave=nwave;

  torch::nn::Sequential ocmod_list;
  for (int ioc_loop=0; ioc_loop<oc_loop; ++ioc_loop) {
    ocmod_list->push_back(NNMod(maxnumtype,nwave,atomtype,oc_nblock,oc_nl, \
      oc_dropout_p,oc_activate,initpot,oc_table_norm));
  }
  GetDensity density(rs,inta,cutoff,nipsin,norbit,ocmod_list);
  this->density=register_module("density",density);
  NNMod nnmod(maxnumtype,outputneuron,atomtype,nblock,nl,dropout_p,activate,initpot,table_norm);
  this->nnmod=register_module("nnmod",nnmod);
};

void PESImpl::forward(
  torch::Tensor &cart,
  torch::Tensor &atom_index,
  torch::Tensor &local_species,
  torch::Tensor &neigh_species,
  torch::Tensor &neigh_list,
  torch::Tensor &energy,
  torch::Tensor &force,
  torch::Tensor &output) {

  atom_index = atom_index.t().contiguous();
  cart=cart.detach().clone();

  cart.requires_grad_(true);
  torch::Tensor density;
  std::vector<torch::Tensor> density_list;
  std::vector<torch::Tensor> orb_coeff_list;
  std::vector<torch::Tensor> orbital_list;
  std::vector<torch::Tensor> orb_coeff_m_list;
  this->density(cart,atom_index,local_species,neigh_species,neigh_list,density_list,orb_coeff_list,orb_coeff_m_list,orbital_list);

  auto T=density_list.size()-1;
  int nlocal=local_species.sizes()[0];
  int nlocal_lmp = lmp->atom->nlocal;

  density=(density_list[T]).index({Slice(0,nlocal_lmp)});
  local_species=local_species.index({Slice(0,nlocal_lmp)});

  output = this->nnmod(&density,&local_species)+this->nnmod->initpot;

  torch::Tensor varene = torch::sum(output/*.index({Slice(0,nlocal_lmp)})*/);

  c10::optional<bool> retain_graph =true;
     
  torch::Tensor /*g_cTp1_dT*/ g_cmTp1_dT = (torch::autograd::grad({varene},{density_list[T]},{},false/*retain_graph*/))[0];

  #if MODE_CUDA
density=torch::empty({1});//clean mem
output=torch::empty({1});//clean mem
  #endif

if (T>0) {
  lmp->comm->forward_comm_tensor(1,atom_map,g_cmTp1_dT, 0);
}

if (T>0) {
  g_dT_cT_sum=torch::zeros({nlocal,nwave},torch::dtype(varene.dtype())).contiguous();
  #if MODE_CUDA
  g_dT_cT_sum=g_dT_cT_sum.to(varene.device());
  #endif

  g_dT_G_sum=torch::zeros_like(orbital_list[0],torch::dtype(varene.dtype())).contiguous();
  #if MODE_CUDA
  g_dT_G_sum=g_dT_G_sum.to(varene.device());
  #endif
}

  for (size_t t=T;t>0; --t) { 
    auto dT=(density_list[t]);
    auto cT=(orb_coeff_list[t-1]); // c_{T+1} in next step

    g_dT_cT = (torch::autograd::grad({dT.index({Slice(0,nlocal_lmp)})},{cT},{g_cmTp1_dT.index({Slice(0,nlocal_lmp)})},retain_graph))[0];

    lmp->comm->forward_comm_tensor_reduce(1,atom_map,g_dT_cT, 0);

    g_dT_cT_sum.add_(g_dT_cT);//g_dT_cT_sum+=g_dT_cT;
  #if MODE_CUDA
g_dT_cT=torch::empty({1});//clean mem
  #endif

    g_dT_G = (torch::autograd::grad({dT},{/*orbital_c*/orbital_list[t]},{g_cmTp1_dT},false/*retain_graph*/))[0];

  #if MODE_CUDA
dT=torch::empty({1});//clean mem
density_list[t]=torch::empty({1});//clean mem
 #endif

    g_dT_G_sum.add_(g_dT_G);//g_dT_G_sum+=g_dT_G;  
  #if MODE_CUDA
g_dT_G=torch::empty({1});//clean mem
  #endif

    g_cmTp1_dT = (torch::autograd::grad({cT.index({Slice(0,nlocal_lmp)})},{/*dTm1*/density_list[t-1]},{g_dT_cT_sum.index({Slice(0,nlocal_lmp)})},false/*retain_graph*/))[0];

    lmp->comm->forward_comm_tensor(1,atom_map,g_cmTp1_dT, 0);
  }  // end of loop over T


  auto orbital_c=orbital_list[0];

torch::Tensor grad_mine;
if (T>0) {
  g_dT_G_sum.add_( (torch::autograd::grad({/*dT*/density_list[0]},{orbital_c},{g_cmTp1_dT},false/*retain_graph*/))[0] );
  /*auto*/ grad_mine= (torch::autograd::grad({orbital_c},{cart},{g_dT_G_sum},retain_graph))[0];
}else{
  /*auto*/ grad_mine= (torch::autograd::grad({density_list[0]},{cart},{g_cmTp1_dT},false/*retain_graph*/))[0];
}

  #if MODE_CUDA
density_list[0]=torch::empty({1});//clean mem
orbital_c=torch::empty({1});//clean mem
g_cmTp1_dT=torch::empty({1});//clean mem
g_dT_G=torch::empty({1});//clean mem
g_dT_G_sum=torch::empty({1});//clean mem
  #endif

  energy=varene.detach();
  force =-grad_mine.detach();  // DIFFENERENT IN USING THIS AND BELOW
};
