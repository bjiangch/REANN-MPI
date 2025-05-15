#include "density.h"

#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "atom.h"
#include <mpi.h>

using namespace torch::indexing;
class cutoff_cosine : public torch::autograd::Function<cutoff_cosine> {
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor &distances,
    torch::Tensor &cutoff) {

    torch::Tensor mycut= (M_PI / cutoff);
    torch::Tensor myd=distances * mycut;
    torch::Tensor mycos=0.5 * torch::cos(myd) + 0.5;
    torch::Tensor f_cut=torch::square(mycos);

    torch::Tensor mygrad=(mycos)*(-torch::sin(myd)*mycut);

    ctx->save_for_backward({/*distances, mycut, mycos, myd*/mygrad});
    return f_cut;
  }

  static torch::autograd::tensor_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs)
  {
    auto saved = ctx->get_saved_variables();
    auto mygrad = saved[0];
    mygrad.mul_(grad_outputs[0]);
    auto grad_cut=torch::zeros({1});
    return {mygrad,grad_cut};
  }

};

class gaussian : public torch::autograd::Function<gaussian> {
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor &distances,
    torch::Tensor &species_,
    torch::Tensor &rs_t,
    torch::Tensor &inta_t) {
    distances=distances.view({-1,1});
    torch::Tensor delta_r=distances-rs_t;
  #if MODE_CUDA
    torch::Tensor radial=torch::exp(torch::einsum("ij,ij->ij",{inta_t,torch::square(delta_r)})).to(distances.dtype()).to(distances.device());
    torch::Tensor g_radial=torch::einsum("ij,ij->ij",{radial,2*inta_t*delta_r}).to(distances.dtype()).to(distances.device());
  #else
    torch::Tensor radial=torch::exp(torch::einsum("ij,ij->ij",{inta_t,torch::square(delta_r)})).to(distances.dtype());
    torch::Tensor g_radial=torch::einsum("ij,ij->ij",{radial,2*inta_t*delta_r}).to(distances.dtype());
  #endif
    torch::Tensor grad_zeros2=torch::zeros_like(rs_t,distances.device());
    ctx->save_for_backward({g_radial,grad_zeros2});
    return radial;

  }

  static torch::autograd::tensor_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto g_radial = saved[0];
    auto grad_zeros2=saved[1];
    g_radial=torch::einsum("ij,ij->i",{grad_outputs[0], g_radial});
    auto grad_zeros=torch::zeros({1});
    //return {g_gaussian,grad_zeros,grad_zeros2,grad_zeros2};
    return {g_radial,grad_zeros,grad_zeros2,grad_zeros2};
  }

};

torch::Tensor GetDensityImpl::gaussian(
    torch::Tensor &distances,
    torch::Tensor &species_/*,
    torch::Tensor &radial*/) {
  distances=distances.view({-1,1});
  #if MODE_CUDA
  torch::Tensor radial=torch::empty({distances.sizes()[0],this->rs.sizes()[1]},distances.dtype()).to(distances.device());
  #else
  torch::Tensor radial=torch::empty({distances.sizes()[0],this->rs.sizes()[1]},distances.dtype());
  #endif

  for (int itype=0; itype < this->rs.sizes()[0]; ++itype) {
    /*torch::Tensor*/ mask = (species_ == itype);
    /*torch::Tensor*/ ele_index = torch::nonzero(mask).view({-1});
    if (ele_index.sizes()[0]>0) {
      /*torch::Tensor*/ part_radial=torch::exp(this->inta.index({Slice(itype,itype+1)})*torch::square \
        (distances.index_select(0,ele_index)-this->rs.index({Slice(itype,itype+1)})));
      radial.masked_scatter_(mask.view({-1,1}),part_radial);
    }
  }

  return radial;
}

torch::Tensor GetDensityImpl::cutoff_cosine(torch::Tensor &distances) {
  return torch::square(0.5 * torch::cos(distances * this->pi_cut/*(M_PI / this->cutoff)*/) + 0.5);
}

torch::Tensor GetDensityImpl::angular(
    torch::Tensor &dist_vec,
    torch::Tensor /*&*/f_cut/*,
    torch::Tensor &angular_p*/) {
  int totneighbour=dist_vec.sizes()[0];
  dist_vec=dist_vec.permute({1,0}).contiguous();
  torch::Tensor orbital=f_cut.view({1,-1});
  torch::Tensor angular_p=torch::empty({this->index_para.sizes()[0], totneighbour},f_cut.dtype()).to(f_cut.device());
  angular_p.index_put_({0},f_cut);
  int num=1;

  int nipsin_int=nipsin[0].item<int>();
  for (int ipsin=1; ipsin < nipsin_int; ++ipsin) {
#if MODE_EINSUM==true
    orbital=torch::einsum("ji,ki -> jki",{orbital,dist_vec}).view({-1,totneighbour});
#else
orbital=((orbital.index({Slice(),None,Slice()}))*dist_vec).view({-1,totneighbour});
#endif
    angular_p.index_put_({Slice(num,num+orbital.sizes()[0])}, orbital);
    num+=orbital.sizes()[0];
  }

  return angular_p;
}


GetDensityImpl::GetDensityImpl(
    torch::Tensor         rs,
    torch::Tensor         inta,
    double                cutoff,
    int                   nipsin,
    int                   norbit,
    torch::nn::Sequential &ocmod_list) {
  
  this->rs=register_parameter("rs",rs);
  this->inta=register_parameter("inta",inta);
  this->cutoff=torch::tensor({cutoff},torch::kDouble).to(torch::kFloat);
  this->pi_cut=M_PI/this->cutoff;
  this->nipsin=torch::tensor({nipsin},torch::kInt);
  std::vector<int> npara={1};
  torch::Tensor index_para=torch::tensor({0},torch::kLong).to(torch::kLong);

  for (int i=1; i < nipsin; ++i) {
    npara.push_back(int(pow(3,i)));
    index_para=torch::cat({index_para,torch::ones({npara[i]},torch::kLong)*i});
  }

  this->index_para=index_para;

  this->params=register_parameter("params",torch::ones_like(this->rs));
  int size_tmp=int(ocmod_list->size())+1;
  this->hyper=register_parameter("hyper",torch::nn::init::orthogonal_(torch::ones({this->rs.sizes()[1],norbit})).unsqueeze(0).unsqueeze(0).repeat({size_tmp,nipsin,1,1}));

  int i=0;
  std::string f_oc;
  torch::OrderedDict<std::string, std::shared_ptr<torch::nn::Module>> ocmod;
  for (const auto &m:*ocmod_list) {
    f_oc="memssage_"+std::to_string(i);
    ocmod.insert(f_oc,m.ptr());
    i++;
  }

  torch::nn::ModuleDict ocmod_dict(ocmod);
  this->ocmod=register_module("ocmod",ocmod_dict);

}

void GetDensityImpl::forward(
    torch::Tensor &cart,
    torch::Tensor &atom_index,
    torch::Tensor &local_species,
    torch::Tensor &neigh_species,
    torch::Tensor &neigh_list,
    std::vector<torch::Tensor> &density_list,
    std::vector<torch::Tensor> &orb_coeff_list,
    std::vector<torch::Tensor> &orb_coeff_m_list,
    std::vector<torch::Tensor> &orbital_list
) {


  this->pi_cut=this->pi_cut.to(torch::dtype(cart.dtype())).to(cart.device());
  auto tag = lmp->atom->tag; //
  int nlocal_lmp = lmp->atom->nlocal;

  int nlocal=local_species.sizes()[0];

  torch::Tensor selected_cart = cart.index_select(0, atom_index.view({-1})).view({2, -1, 3});
  torch::Tensor dist_vec = selected_cart.index({0}) - selected_cart.index({1});

  torch::Tensor distances = dist_vec.norm(2,-1);

#if true 
  auto rs_t=this->rs.index({neigh_species});
  auto inta_t=this->inta.index({neigh_species});

  torch::Tensor orbital = torch::einsum("ji,ik -> ijk",{this->angular(dist_vec,cutoff_cosine::apply(distances,cutoff)),gaussian::apply(distances,neigh_species,rs_t,inta_t)});
#else
torch::Tensor f_cut;
this->cutoff_cosine(distances,f_cut);
torch::Tensor angular_p;
this->angular(dist_vec,f_cut,angular_p);
torch::Tensor aa=(angular_p).t().contiguous();
torch::Tensor bb;
this->gaussian(distances,neigh_species,bb);
torch::Tensor orbital=(aa.index({Slice(),Slice(),None}))*(bb.index({Slice(),None,Slice()}));
#endif


  torch::Tensor orb_coeff=this->params.index_select(0,local_species);

  torch::Tensor atom_index_0=atom_index.index({0});

  torch::Tensor density;

  this->obtain_orb_coeff(0,nlocal,orbital,atom_index_0,neigh_list,orb_coeff,density);

  density_list.push_back(density);
  orbital_list.push_back(orbital);
  auto local_species_lmp=local_species.index({Slice(0,nlocal_lmp)});

  int ioc_loop=0;
  auto ocmod=this->ocmod;
  for (const auto &item : *ocmod) {
    auto m=item.value()->as<NNMod>();

    auto density_lmp=density.index({Slice(0,nlocal_lmp)});

    auto orb_coeff_m = m->forward(&density_lmp,&local_species_lmp);

    auto arange_lmp=torch::arange(nlocal_lmp,orb_coeff_m.device());
    orb_coeff=orb_coeff.index_add(0,arange_lmp,orb_coeff_m);

  lmp->comm->forward_comm_tensor(1,atom_map, orb_coeff, 1);
    orb_coeff_list.push_back(orb_coeff); // not use clone here

    auto orbital_c=orbital.clone();
    orbital_list.push_back(orbital_c);
    
    this->obtain_orb_coeff(ioc_loop+1,nlocal,orbital_c,atom_index_0,neigh_list,orb_coeff,density);

    density_list.push_back(density);

    ioc_loop++;
  }
}

void GetDensityImpl::obtain_orb_coeff(
    int &&iteration,
    int &numatom,
    torch::Tensor &orbital,
    torch::Tensor &/*&*/center_list,
    torch::Tensor &neigh_list,
    torch::Tensor &orb_coeff,
    torch::Tensor &density_p) {
  torch::Tensor expandpara=orb_coeff.index_select(0,neigh_list);

#if MODE_EINSUM==true
  torch::Tensor worbital=torch::einsum("ijk,ik ->ijk", {orbital,expandpara}).to(orb_coeff.dtype());
#else
  torch::Tensor worbital=(orbital*expandpara.index({Slice(),None,Slice()})).to(orb_coeff.dtype());
#endif

  torch::Tensor sum_worbital=torch::zeros({numatom,orbital.sizes()[1],this->rs.sizes()[1]},orb_coeff.dtype()).to(orb_coeff.device());

  sum_worbital.index_add_(0,center_list,worbital);

  expandpara=this->hyper.index({iteration}).index_select(0,this->index_para);
#if MODE_EINSUM==true
  torch::Tensor hyper_worbital=torch::einsum("ijk,jkm ->ijm",{sum_worbital,expandpara});
#else
torch::Tensor hyper_worbital=(sum_worbital.index({Slice(),Slice(),Slice(),None})*expandpara.index({None,Slice(),Slice(),Slice()})).sum(2);
#endif

  density_p= torch::sum(torch::square(hyper_worbital),{1});
}
