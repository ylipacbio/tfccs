set -vex

include_dir=/home/UNIXHOME/yli/venv/tensorflow2.0/include
lib_dir=/home/UNIXHOME/yli/venv/tensorflow2.0/lib

echo "Sanity check headers under ~/venv/tensorflow2.0/include"
ls ${include_dir}/google/protobuf && echo protobuf library installed to include dir
ls ${include_dir}/tensorflow/c && echo tensorflow c library installed to include dir
ls ${include_dir}/tensorflow/cc && echo tensorflow C++ library installed to include dir
ls ${include_dir}/tensorflow/core && echo tensorflow core library installed to include dir
ls ${lib_dir}/lib/libtensorflow.so && echo libtensorflow c lib installed
ls ${lib_dir}/lib/libtensorflow_cc.so && echo libtensorflow_cc C++ lib installed
ls ${lib_dir}/lib/libtensorflow_framework.so && echo libtensorflow_framework lib installed

cp /home/UNIXHOME/yli/repo/tensorflow/tensorflow/cc/tutorials/example_trainer.cc .
echo "Compile an example trainer which calls C++ API"
gcc example_trainer.cc \
    -I/home/UNIXHOME/yli/venv/tensorflow2.0/include \
    -L/home/UNIXHOME/yli/venv/tensorflow2.0/lib \
    -lrt -lpthread -lstdc++ \
    -ltensorflow_cc -ltensorflow_framework 

echo "Run compiled executable"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lib_dir} ./a.out
