path_lmp=$1

cp -r build-reann-mpi/ $path_lmp
cp cmake/CMakeLists.txt $path_lmp/cmake/
cp src/* $path_lmp/src/
