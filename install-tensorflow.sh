function install_tf_on_pbcluster() {
echo "Install python 3.7.3 and tensorflow v2.0 to virtual env"

. /mnt/software/Modules/current/init/sh
module load python/3.7.3
python3 -m venv ~/venv/tf
source ~/venv/tf/bin/activate
which python3
python --version
# 3.7.3

echo "upgrade pip"
pip install --upgrade pip

echo "install tensorflow"
pip install tensorflow==2.0
python -c 'import tensorflow as tf; print(tf.__version__)'
# 2.0.0

echo "install useful py packages"
pip install pylint autopep8 pytest ipython
}

function install_tf_on_mac() {
echo "Brew install python3"
python3 --version
# python 3.7.4

echo "Create virtual env for tensorflow"
python3 -m venv ~/venv/tf

echo "Activate tensor flow virtual env"
source ~/venv/tf/bin/activate
which python3
# ~/venv/tf/bin/python

echo "upgrade pip"
pip install --upgrade pip

echo "install tensorflow"
pip install tensorflow==2.0
python -c 'import tensorflow as tf; print(tf.__version__)'
# 2.0.0

echo "install useful py packages"
pip install pylint autopep8 pytest ipython jupyterlab

echo "Launch jupyter notebook"
juypter notebook&
}
