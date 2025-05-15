// Copyright 2018 Andreas Singraber (University of Vienna)
// //
// // This Source Code Form is subject to the terms of the Mozilla Public
// // License, v. 2.0. If a copy of the MPL was not distributed with this
// // file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include <mpi.h>
//#include <stdlib.h>
#include <pair_reann_mpi.h>
#include <string>
#include <numeric>
#include <vector>
#include <iostream> // added for cout to file
#include <fstream>
#include <sstream>
#include <algorithm>
#include "atom.h"
//#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "utils.h"
#include "utils_x.h"
#include <unistd.h>

#include <unordered_map> //for map

#include "update.h"

using namespace LAMMPS_NS;
using namespace std;

PairREANN_MPI::PairREANN_MPI(LAMMPS *lmp) : Pair(lmp) {
}

PairREANN_MPI::~PairREANN_MPI() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
  // delete the map from the global index to local index
  atom->map_delete();
  atom->map_style = Atom::MAP_NONE;
}

void PairREANN_MPI::allocate() {
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 0;
    }
  }
}

void PairREANN_MPI::init_style() {

int me;
me=lmp->comm->me;

if (me==0) {
std::cout << "Compiled on: " << __DATE__ << "-" << __TIME__;
#if MODE_FLOAT
std::cout<<" with float type";
#else
std::cout<<" with double type";
#endif

#if MODE_CUDA
std::cout<<" on GPU\n";
#else
std::cout<<" on CPU\n";
#endif
}

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  try {
    #if MODE_FLOAT
      tensor_type=torch::kFloat;
    #else
      tensor_type = torch::kDouble;
    #endif
    torch::DeviceType tensor_device_type=torch::kCPU;

    this->module=PES(1);
    module->to(tensor_device_type);
    module->to(tensor_type);
    
    this->module->lmp=lmp;
    this->module->density->lmp=lmp;

    #if MODE_FLOAT
      torch::load(module,"REANN_LAMMPS_FLOAT.pt");
    #else
      torch::load(module,"REANN_LAMMPS_DOUBLE.pt");
    #endif

    if (torch::cuda::is_available()) {
      // used for assign the CUDA_VISIBLE_DEVICES= id
      MPI_Barrier(MPI_COMM_WORLD);
      // return the GPU id for the process
      //int id;
      int id=select_gpu();
      cout << "Using select_gpu "<< endl;
      //int id=me; // a naive way to use GPU; Process i <-> GPU i
      tensor_device_type=torch::kCUDA;
      auto device=torch::Device(tensor_device_type,id);
      tensor_device=tensor_device.to(device);
      module->to(device);
      this->module->density->cutoff=this->module->density->cutoff.to(tensor_device.device(),true);
      this->module->density->nipsin=this->module->density->nipsin.to(tensor_device.device(),true);
      this->module->density->index_para=this->module->density->index_para.to(tensor_device.device(),true);
      //tensor_option=torch::TensorOptions().dtype(tensor_type);
      cout << "The simulations are performed on the GPU "<< id << endl;
      option1=option1.pinned_memory(true);
      option2=option2.pinned_memory(true);

    }
    module->eval();

  }
  catch (const c10::Error& e) {
    std::cerr << "error of loading the pytorch model\n";
  }
//  std::cout << " success of loading the pytorch model\n";

  // create the map from global to local
  if (atom->map_style == Atom::MAP_NONE) {
    atom->nghost=0;
    atom->map_init(1);
    atom->map_set();
  }
  
}

void PairREANN_MPI::coeff(int narg, char **arg) {
  if (!allocated) {
    allocate();
  }

  int n = atom->ntypes;
  int ilo,ihi,jlo,jhi;
  ilo = 0;
  jlo = 0;
  ihi = n;
  jhi = n;
  //utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  //utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
  cutoff = utils::numeric(FLERR, arg[2], false,lmp);
  datatype=arg[3];
  cutoffsq=cutoff*cutoff;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }
  //if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


void PairREANN_MPI::settings(int narg, char **arg)
{
}


double PairREANN_MPI::init_one(int i, int j) {
  return cutoff;
}
/*=========================copy from the integrate.ev_set===========================
   set eflag,vflag for current iteration
   invoke matchstep() on all timestep-dependent computes to clear their arrays
   eflag/vflag based on computes that need info on this ntimestep
   eflag = 0 = no energy computation
   eflag = 1 = global energy only
   eflag = 2 = per-atom energy only
   eflag = 3 = both global and per-atom energy
   vflag = 0 = no virial computation (pressure)
   vflag = 1 = global virial with pair portion via sum of pairwise interactions
   vflag = 2 = global virial with pair portion via F dot r including ghosts
   vflag = 4 = per-atom virial only
   vflag = 5 or 6 = both global and per-atom virial
=========================================================================================*/
//#pragma GCC push_options
//#pragma GCC optimize (0)

void PairREANN_MPI::compute(int eflag, int vflag) {

  if(eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
  double **x=atom->x;
  double **f=atom->f;
  int *type = atom->type; 
  tagint *tag = atom->tag; //
  int nlocal = atom->nlocal,nghost=atom->nghost;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int i,ii,inum,j,jj,jnum,maxneigh;
  int totneigh=0,nall=nghost+nlocal;
  int totdim=nall*3;

  torch::Tensor cart_, atom_index_, neigh_list_, local_species_,neigh_species_;
  torch::Tensor energy_p, force_p, output_p; 
  torch::Tensor tensor_etot, tensor_force, tensor_atom_ene;
  double *etot, *force, *atom_ene;
   
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  int numneigh_atom = accumulate(numneigh, numneigh + inum , 0);

  vector<double> cart(totdim);
  vector<long> atom_index(numneigh_atom*2);
  vector<long> neigh_species(numneigh_atom);
  vector<long> neigh_list(numneigh_atom);
  int len_ls=numneigh_atom>nall?numneigh_atom:nall;
  vector<long> local_species(len_ls);
  double dx,dy,dz,d2;
  double xtmp,ytmp,ztmp;
  unsigned countnum=0;
for(size_t i=0;i<numneigh_atom;i++){
  neigh_list[i]=-1;
}

  for (ii=0; ii<nall; ++ii) {
    for (jj=0; jj<3; ++jj) {
      cart[countnum]=x[ii][jj];
      ++countnum;
    }
  }

std::unordered_map<int, int>atom_map;

for(size_t ii=0;ii<inum;ii++){
  int i=ilist[ii];
  atom_map[tag[i]]=i;
  local_species[i]=type[i]-1;
}

int count_j=inum;

  for (jj=inum; jj<nall; ++jj) {
    int tag_j=tag[jj];
    j=atom->map(tag_j);
    if(atom_map.find(tag_j)==atom_map.end()){
      atom_map[tag_j]=count_j;//j;
      local_species[count_j]=type[j]-1;
      count_j+=1;
    }
  }

    lmp->comm->atom_map=atom_map;
    this->module->atom_map=atom_map;
    this->module->density->atom_map=atom_map;
//    this->module->mytag=mytag;
    this->module->count_j=count_j;
//    this->module->density->mytag=mytag;
    this->module->density->count_j=count_j;

  for (ii=0; ii<inum; ++ii) {
    i=ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jnum=numneigh[i];
    jlist=firstneigh[i];
    for (jj=0; jj<jnum; ++jj) {
      j=jlist[jj];
      dx = xtmp - x[j][0];
      dy = ytmp - x[j][1];
      dz = ztmp - x[j][2];
      d2 = dx * dx + dy * dy + dz * dz;
      if (d2<cutoffsq) {
        atom_index[totneigh*2]=i;
        atom_index[totneigh*2+1]=j;
        neigh_list[totneigh]=atom_map[tag[j]]; /*j;//type[j]-1;tag[j];*/ //atom->map(tag[j]);
        neigh_species[totneigh]= type[j]-1;
        ++totneigh;
      }
    }
  }


  #if MODE_CUDA
    #if MODE_FLOAT
  /*auto*/ cart_=torch::from_blob(cart.data(),{nall,3},option1).to(tensor_device.device(),true).to(torch::kFloat);
    #else
  /*auto*/ cart_=torch::from_blob(cart.data(),{nall,3},option1).to(tensor_device.device(),true);
    #endif
  /*auto*/ atom_index_=torch::from_blob(atom_index.data(),{totneigh,2},option2).to(tensor_device.device(),true);
  /*auto*/ neigh_list_=torch::from_blob(neigh_list.data(),{totneigh},option2).to(tensor_device.device(),true);
  /*auto*/ neigh_species_=torch::from_blob(neigh_species.data(),{totneigh},option2).to(tensor_device.device(),true);
  /*auto*/ local_species_=torch::from_blob(local_species.data(),{/*inum*/count_j},option2).to(tensor_device.device(),true);
  #else
    #if MODE_FLOAT
  /*auto*/ cart_=torch::from_blob(cart.data(),{nall,3},option1).to(torch::kFloat);
    #else
  /*auto*/ cart_=torch::from_blob(cart.data(),{nall,3},option1);
    #endif
  /*auto*/ atom_index_=torch::from_blob(atom_index.data(),{totneigh,2},option2);
  /*auto*/ neigh_list_=torch::from_blob(neigh_list.data(),{totneigh},option2);
  /*auto*/ neigh_species_=torch::from_blob(neigh_species.data(),{totneigh},option2);
  /*auto*/ local_species_=torch::from_blob(local_species.data(),{/*inum*/count_j},option2);
  #endif


  module->forward(cart_,atom_index_,local_species_,neigh_species_,neigh_list_,energy_p,force_p,output_p);

  #if MODE_FLOAT
    #if MODE_CUDA
  /*auto*/ tensor_etot  =energy_p.to(torch::kDouble).cpu();
  /*auto*/ tensor_force =force_p.to(torch::kDouble).cpu();
  /*auto*/ tensor_atom_ene=output_p.to(torch::kDouble).cpu();
    #else
  /*auto*/ tensor_etot  =energy_p.to(torch::kDouble);
  /*auto*/ tensor_force =force_p.to(torch::kDouble);
  /*auto*/ tensor_atom_ene=output_p.to(torch::kDouble);
    #endif
  #else
    #if MODE_CUDA
  /*auto*/ tensor_etot  =energy_p.cpu();
  /*auto*/ tensor_force =force_p.cpu();
  /*auto*/ tensor_atom_ene=output_p.cpu();
    #else
  /*auto*/ tensor_etot  =energy_p;
  /*auto*/ tensor_force =force_p;
  /*auto*/ tensor_atom_ene=output_p;
    #endif
  #endif
  /*auto*/ etot  = tensor_etot.data_ptr<double>();
  /*auto*/ force = tensor_force.data_ptr<double>();
  /*auto*/ atom_ene = tensor_atom_ene.data_ptr<double>();

  for (i=0; i<nall; ++i) {
    //for (j=0; j<3; ++j) {
    //  f[i][j] += *force++; 
    //  //f[i][j] += force[i*3+j]; //slow
    //}
    f[i][0] += *force++; 
    f[i][1] += *force++; 
    f[i][2] += *force++; 
  }


  if (eflag_global) {
    ev_tally(0,0,nlocal,1,etot[0],0.0,0.0,0.0,0.0,0.0);
  }

  if (eflag_atom) {
    for ( ii = 0; ii < nlocal; ++ii) {
      i=ilist[ii];
      eatom[ii] = atom_ene[i];
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

}
//#pragma GCC pop_options
//
int PairREANN_MPI::select_gpu() {
  int numprocesses, id_process;
  int trap_key = 0;
  MPI_Status status;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numprocesses);
  MPI_Comm_rank(MPI_COMM_WORLD, &id_process);

  //const char* tmpstr_1=("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >gpu_info-"+to_string(id_process)).data();
  const char* tmpstr_1=("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used|sed -e 's/[a-zA-Z:]//g' >gpu_info-"+to_string(id_process)).data();
  string tmpstr_2=("gpu_info-"+to_string(id_process)).data();
  const char* tmpstr_3=("rm gpu_info-"+to_string(id_process)).data();

  if (id_process != trap_key)  //this will allow only process 0 to skip this stmt
    MPI_Recv(&trap_key, 1, MPI_INT, id_process-1, 0, MPI_COMM_WORLD, &status);
  
  system(tmpstr_1);
  ifstream gpu_sel(tmpstr_2);
  string texts;
  vector<double> memgpu_arr;
  while (getline(gpu_sel,texts))
  {
    memgpu_arr.push_back(std::stod(texts));
    std::cout<<"MB myidprocess "<<id_process<<"|"<< std::stod(texts)<<std::endl;
  }
  gpu_sel.close();

  //auto smallest=min_element(std::begin(memgpu_arr),std::end(memgpu_arr));//minval
  auto smallest=min_element(memgpu_arr.begin(),memgpu_arr.end());//minval
  auto id=distance(memgpu_arr.begin(), smallest);//index of minval
  if (id_gpu_using<0){
    id_gpu_using=id;
  }else{
    id=id_gpu_using;
  }
  std::cout<<"idprocess "<<id_process<<" id_gpu_using| "<<id_gpu_using<<std::endl;

  torch::Tensor tensor_device=torch::empty(1000,torch::Device(torch::kCUDA,id));

  if(id_process != numprocesses-1)  // this will allow only the last process to skip this
    MPI_Send(&trap_key, 1, MPI_INT, id_process+1, 0, MPI_COMM_WORLD);
  system(tmpstr_3);
  return id;
}
