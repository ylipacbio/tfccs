set -vex -o pipefail
. /mnt/software/Modules/current/init/sh
module load python/3.7.3
branch=r2.1
root_dir=/home/UNIXHOME/yli/venv/tf-${branch}-cpu
python3 -m venv ${root_dir}
source ${root_dir}/bin/activate
pip install numpy six
pip install --upgrade pip

module load bazel/0.26.1 gcc/8.3.0
git clone https://github.com/tensorflow/tensorflow.git
cd /tensorflow
git checkout origin/${branch}
rm -rf ~/.cache/bazel
cat > .tf_configure.bazelrc << EOF
build --action_env PYTHON_BIN_PATH="${root_dir}/bin/python"
build --action_env PYTHON_LIB_PATH="${root_dir}/lib/python3.7/site-packages"
build --python_path="${root_dir}/bin/python"
build:xla --define with_xla_support=true
build --config=xla
build --config=monolithic
build:opt --copt=-march=nehalem
build:opt --copt=-Wno-sign-compare
build:opt --copt=-static-libstdc++
build:opt --copt=-static-libgcc
build:opt --cxxopt=-static-libstdc++
build:opt --cxxopt=-static-libgcc
build:opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
build:opt --linkopts=-static-libstdc++
build:opt --linkopts=-static-libgcc
build:opt --host_copt=-march=nehalem
build:opt --define PREFIX=${root_dir}
build:opt --define LIBDIR=${root_dir}/lib
build:opt --define INCLUDEDIR=${root_dir}/include
build:opt --define with_default_optimizations=true
build:v2 --define=tf_api_version=2
test --flaky_test_attempts=3
test --test_size_filters=small,medium
test --test_tag_filters=-benchmark-test,-no_oss,-oss_serial
test --build_tag_filters=-benchmark-test,-no_oss
test --test_tag_filters=-gpu
test --build_tag_filters=-gpu
build --action_env TF_CONFIGURE_IOS="0"
EOF
bazel fetch //tensorflow:grpc
cat ../patch-00 | (cd ~/.cache/bazel/*/*/external/.. && patch -p0)
bazel build //tensorflow/tools/lib_package:libtensorflow
bazel build //tensorflow:libtensorflow_cc.so
bazel build //tensorflow:libtensorflow_framework.so
bazel fetch @com_google_absl//absl:all # doesn't seem needed
tar xvf bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz -C ${root_dir} 
rsync -aP bazel-bin/tensorflow/lib* ${root_dir}/lib/
find \
tensorflow/{cc,core} \
-name '*.h' -o -name '*.hpp' | rsync --timeout=400 -avx --files-from - ./ ${root_dir}/include
cp -a \
third_party \
bazel-genfiles/* \
  bazel-tensorflow/external/eigen_archive/unsupported/ \
  bazel-tensorflow/external/eigen_archive/Eigen/ \
  ~/.cache/bazel/*/*/external/com_google_protobuf/src/google \
  ~/.cache/bazel/*/*/external/com_google_absl/absl \
  ${root_dir}/include/
g++ tensorflow/cc/tutorials/example_trainer.cc \
  -I${root_dir}/include \
  -ltensorflow_cc -ltensorflow_framework -lrt \
  -L${root_dir}/lib \
  -o a.out
./a.out
