#!/bin/bash
set -xe

# Install pip
if ! [ `command -v pip` ]
then
  sudo easy_install pip
fi

# Create virtual env
VENV=.venv-pem
sudo pip2.7 install virtualenv
sudo rm -rf $VENV
virtualenv $VENV -p /usr/bin/python2.7
source $VENV/bin/activate

# Install dependencies
pip2.7 install --download-cache=cache -r muvr.pip

# Insteall neon latest
git clone --branch v1.1.0 https://github.com/NervanaSystems/neon.git $VENV/neon
cd $VENV/neon
make sysinstall
cd -

# Install pem
cd pem
python setup.py install
cd -

cat >> $VENV/bin/activate << EOF
export SPARK_HOME=$SPARK_HOME
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH
export PATH=$SPARK_HOME/bin:$PATH
EOF
source $VENV/bin/activate
