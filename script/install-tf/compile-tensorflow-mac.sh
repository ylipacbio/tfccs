set -vex -o pipefail

# MUST SPEICFY root_dir where you want to install libtensorflow!
root_dir=${HOME}/venv/tf
# MUST SPEICFY which git tensorflow version to install
branch=r2.1

python3 -m venv ${root_dir}
source ${root_dir}/bin/activate
pip install numpy six
pip install --upgrade pip

# NOTE: If you have homebrew/protobuf installed, must 
# uninstall Homebrew/protobuf first! Otherwise, I saw bazel 
# failed to compile tensorflow due to conflicting protobuf headers
brew uninstall protobuf

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout origin/${branch}
rm -rf ~/.cache/bazel

# NOTE: If you have failed to bazel build tensorflow before, 
# a deep clean may be needed
# bazel clean --expunge

# NOTE: the following is equivalent to run `./configure` manually
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

echo "It will take bazel several hours to compile tensorflow"
bazel fetch //tensorflow:grpc
bazel build //tensorflow/tools/lib_package:libtensorflow
bazel build //tensorflow:libtensorflow_cc.so
bazel build //tensorflow:libtensorflow_framework.so
bazel fetch @com_google_absl//absl:all

echo "After bazel successfully built tensorflow, create tensorflow library"
tar xvf bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz -C ${root_dir} 
rsync -aP bazel-bin/tensorflow/lib* ${root_dir}/lib/
find \
tensorflow/{cc,core} \
-name '*.h' -o -name '*.hpp' | rsync --timeout=400 -avx --files-from - ./ ${root_dir}/include
cp -a third_party  ${root_dir}/include/
cp -a bazel-genfiles/*  ${root_dir}/include/
mkdir 
cp -a bazel-tensorflow/external/eigen_archive/unsupported  ${root_dir}/include/
cp -a bazel-tensorflow/external/eigen_archive/Eigen ${root_dir}/include/

echo "At this point, you have to find bazel cache directory to find headers for protobuf and absl!"
echo "bazel document said cache directory should be under ~/.cache/, however, I found it under /private/var/tmp"
#bazel_cache_dir=/private/var/tmp/_bazel_yli/17eff0af0c8a8ce054a5583ed1c05cca/
cp -a `ls -d /private/var/tmp/_bazel*/*/external/com_google_protobuf/src/google/` ${root_dir}/include/ 
cp -a `ls -d /private/var/tmp/_bazel*/*/external/com_google_absl/absl/` ${root_dir}/include/ 

# Test with tensorflow example_trainer.cc
clang tensorflow/cc/tutorials/example_trainer.cc \
  -I${root_dir}/include \
  -ltensorflow_cc \
  -ltensorflow_framework \
  -L${root_dir}/lib \
  -lstdc++ \
  -o a.out
# Run compiled executable
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${root_dir}/lib ./a.out
