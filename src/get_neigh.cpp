#include "get_neigh.h"
#include <iomanip>
#include <iostream>

using namespace torch::indexing;

inline std::vector<int> arange(int &&start,int &&end) {
  std::vector<int> arr;
  for (int i=start;i<end;++i) {
    arr.push_back(i);
  }
  return arr;
}

Neigh_ListImpl::Neigh_ListImpl(
    double cutoff,
    int nlinked) {
  /// nliked used for the periodic boundary condition in cell linked list

  this->cutoff=cutoff;
  this->cell_list=this->cutoff/nlinked;
  torch::Tensor r1 = torch::arange(-nlinked, nlinked+1);
  this->linked=torch::cartesian_prod({r1, r1, r1}).view({1,-1,3});
}

//torch::Tensor Neigh_ListImpl::forward(
void Neigh_ListImpl::forward(
    torch::Tensor &period_table,
    torch::Tensor &coordinates,
    torch::Tensor &cell,
    torch::Tensor &mass,
    torch::Tensor &neigh_list_p, 
    torch::Tensor &shifts_p) {

  int numatom=coordinates.sizes()[0];
  torch::Tensor inv_cell=torch::inverse(cell);
torch::Tensor inv_coor=torch::matmul(coordinates, inv_cell);
  torch::Tensor deviation_coor=torch::round(inv_coor-inv_coor.index({0}));
  inv_coor=inv_coor-deviation_coor;
coordinates.index_put_({Slice(),Slice()}, torch::matmul(inv_coor,cell));
  torch::Tensor totmass=torch::sum(mass);
torch::Tensor com=torch::matmul(mass.view({1,-1}),coordinates)/totmass;

  coordinates.index_put_({Slice(),Slice()},coordinates-com.index({None,Slice()}));
  auto t_min=std::get<0>(torch::min(this->cutoff/torch::abs(cell),0));
  torch::Tensor num_repeats = torch::ceil(t_min).to(torch::kInt);

  /// the number of periodic image in each direction
  num_repeats = period_table*num_repeats;
  torch::Tensor num_repeats_up = (num_repeats+1).detach();
  torch::Tensor num_repeats_down = (-num_repeats).detach();
  std::vector<int> arr_r1=arange(num_repeats_down.index({0}).item<int>(), num_repeats_up.index({0}).item<int>());
  torch::Tensor r1=torch::from_blob(arr_r1.data(),arr_r1.size(),torch::kInt).to(coordinates.device());
  std::vector<int> arr_r2=arange(num_repeats_down.index({1}).item<int>(), num_repeats_up.index({1}).item<int>());
  torch::Tensor r2=torch::from_blob(arr_r2.data(),arr_r2.size(),torch::kInt).to(coordinates.device());
  std::vector<int> arr_r3=arange(num_repeats_down.index({2}).item<int>(), num_repeats_up.index({2}).item<int>());
  torch::Tensor r3=torch::from_blob(arr_r3.data(),arr_r3.size(),torch::kInt).to(coordinates.device());

  torch::Tensor shifts=torch::cartesian_prod({r1, r2, r3}).to(coordinates.dtype());

shifts=torch::matmul(shifts,cell);
  int num_shifts = shifts.sizes()[0];
  torch::Tensor all_shifts = torch::arange(num_shifts, coordinates.device());
  torch::Tensor all_atoms = torch::arange(numatom, coordinates.device());
  torch::Tensor prod = torch::cartesian_prod({all_shifts,all_atoms}).t().contiguous();
  torch::Tensor mincoor=std::get<0>(torch::min(coordinates,0))-this->cutoff-1e-6;
  coordinates=coordinates-mincoor;
  torch::Tensor maxcoor=std::get<0>(torch::max(coordinates,0))+this->cutoff;
  torch::Tensor image=(coordinates.index({None,Slice(),Slice()})+shifts.index({Slice(),None,Slice()})).view({-1,3});
  /// get the index in the range (ori_cell-rc,ori_cell+rs) in  each direction;
  torch::Tensor mask=torch::nonzero(((image<maxcoor)*(image>0)).all(1)).view({-1});
  torch::Tensor image_mask=image.index_select(0,mask);
  /// save the index(shifts, atoms) for each atoms in the modified cell ;
  prod=prod.index({Slice(),mask});
  torch::Tensor ori_image_index=torch::floor(coordinates/this->cell_list);

  /// the central atoms with its linked cell index;
  torch::Tensor cell_linked=this->linked.expand({numatom,-1,3}).to(coordinates.device());
  torch::Tensor neigh_cell=ori_image_index.index({Slice(),None,Slice()})+cell_linked;
  /// all the index for each atoms in the modified cell;
  torch::Tensor image_index=torch::floor(image_mask/this->cell_list);
  torch::Tensor max_cell_index=torch::ceil(maxcoor/this->cell_list);
  torch::Tensor neigh_cell_index=(neigh_cell.index({Slice(),Slice(),2})*max_cell_index.index({1})*max_cell_index.index({0})+\
  neigh_cell.index({Slice(),Slice(),1})*max_cell_index.index({0})+neigh_cell.index({Slice(),Slice(),0})).to(torch::kUInt8);
  torch::Tensor nimage_index=(image_index.index({Slice(),2})*max_cell_index.index({1})*max_cell_index.index({0})+\
  image_index.index({Slice(),1})*max_cell_index.index({0})+image_index.index({Slice(),0})).to(torch::kUInt8);
torch::Tensor t1=neigh_cell_index.index({Slice(),None,Slice()}).to(torch::kUInt8);
torch::Tensor t2=nimage_index.index({None,Slice(),None}).to(torch::kUInt8);
torch::Tensor m=t1==t2;
torch::Tensor mask_neigh=torch::nonzero(m);
  torch::Tensor atom_index=mask_neigh.index({Slice(),Slice(0,2)});
  atom_index=atom_index.t().contiguous();
  torch::Tensor selected_coordinate1 = coordinates.index_select(0, atom_index.index({0})).contiguous();
  torch::Tensor selected_coordinate2 = image_mask.index_select(0, atom_index.index({1})).contiguous();
  torch::Tensor distances = (selected_coordinate1 - selected_coordinate2).norm(2, -1);

  torch::Tensor pair_index = torch::nonzero((distances < this->cutoff)*(distances>0.001)).reshape(-1);
  torch::Tensor neigh_index = atom_index.index({Slice(),pair_index});
  torch::Tensor tmp=prod.index({Slice(),neigh_index.index({1})});
  torch::Tensor neigh_list=torch::vstack({neigh_index.index({0}), tmp.index({1})});
  shifts = shifts.index_select(0, tmp.index({0}));

  neigh_list_p=neigh_list;
  shifts_p=shifts;
}
