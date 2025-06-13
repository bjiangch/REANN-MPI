path_1=/public/home/xjf/proj-22-0117/lammps/test_mpi/lmp/lammps-libtorch/src/
path_2=./
echo diff $path_1 vs $path_2

files=(
activate.cpp
activate.h
get_neigh.cpp
get_neigh.h
log_time.cpp
log_time.h
log_xjf.h
model.cpp
model.h
readinput.cpp
readinput.h
ResBlock.cpp
ResBlock.h
set_parameters.cpp
set_parameters.h
settings.cpp
settings.h
utils_x.cpp
utils_x.h
density.cpp
density.h
pair_reann_cpp.cpp
pair_reann_cpp.h
PES.cpp
PES.h
comm_brick_reann.h
comm_brick_reann.cpp
comm.cpp
comm.h    
comm_brick.cpp  
comm_brick.h
comm_tiled.cpp  
comm_tiled.h    
input.h
input.cpp
readme
)

for f in ${files[@]};do
  echo ----------------------------------
  echo "            "$f
  echo "            "
  ###cp $path_1/$f $path_2
  diff $path_1/$f $path_2/$i
  echo -e "\n"
done

# a new version
#for i in `ls`;do a=`diff $i ../../../src/$i|wc -l`;if [ $a -gt 0 ];then echo $i $a;fi done
