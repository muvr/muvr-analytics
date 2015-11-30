#!/bin/bash
set -xe

# Install pip
if ! [ `command -v pip` ]
then
  sudo easy_install pip
fi

# Create virtual env
VENV=.venv-dcos
sudo pip2.7 install virtualenv
sudo rm -rf $VENV
virtualenv $VENV -p /usr/bin/python2.7
source $VENV/bin/activate

# Install dependencies
pip2.7 install --download-cache=cache -r muvr.pip

cd $VENV
curl -O https://downloads.mesosphere.io/dcos-cli/install.sh
bash install.sh . http://ec2-54-154-13-232.eu-west-1.compute.amazonaws.com
