set -vex

# Define where to install headers and shared libraries
include_dir=/home/UNIXHOME/yli/venv/tensorflow2.0/include/
lib_dir=/home/UNIXHOME/yli/venv/tensorflow2.0/lib/

hostname # host was mp-ml00

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

./configure
# HERE configure tensorflow, and I have set NO to all add-ons.

# Test, which should be done in the first time, and can be skipped later.
# bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test

echo "Build libtensorflow, libtensorflow_cc and libtensorflow_framework"
echo "Note that I used --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2, which may not apply to all nodes"
echo "Note the following bazel commands may take ~20 - 30 m to complete"
bazel build --config opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow/tools/lib_package:libtensorflow
bazel build --config opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
bazel build --config opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow:libtensorflow_framework.so

cwd=`pwd`
echo "Install shared library to ${lib_dir}"
mkdir -p ${lib_dir}
cp bazel-bin/tensorflow/lib* ${lib_dir}/

echo "Install headers to ${include_dir}"
mkdir -p ${include_dir}
cp bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz ${include_dir}/../
cd ${include_dir}/../ && tar xvf libtensorflow.tar.gz && cd ${cwd}
for f in `find tensorflow/cc/ -name '*.h'`; do (mkdir -p `dirname ${include_dir}/$f` && cp $f ${include_dir}/$f); done
for f in `find tensorflow/core/ -name '*.h'`; do (mkdir -p `dirname ${include_dir}/$f` && cp $f ${include_dir}/$f); done
cp -r bazel-genfiles/* ${include_dir}
cp -r third_party ${include_dir}/
cp -r bazel-tensorflow/external/eigen_archive/unsupported/  ${include_dir}
cp -r bazel-tensorflow/external/eigen_archive/Eigen/  ${include_dir}
cp -r tensorflow/lite/tools/make/downloads/absl/absl ${include_dir}

echo "To test if libtensorflow, libtensorflow_cc and libtensorflow_framework are installed correctly, call test_tensorflow.sh"
