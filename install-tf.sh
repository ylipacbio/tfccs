echo "Install python 3.7.3 and tensorflow v2.0 to virtual env"

. /mnt/software/Modules/current/init/sh
module load python/3.7.3
python3 -m venv ~/venv/tf
source ~/venv/tf/bin/activate
which python3
# ~/venv/py374/bin/python3
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
