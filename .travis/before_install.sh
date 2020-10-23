#!/bin/bash
#
# Travis CI install script
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
# All rights reserved
#
# Contributed by Rodrigo Tobar
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307  USA
#

# Eigen 3.3.90
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout fb0c6868ad8d43e052c9e027b41b3dfe660bb57d
mkdir build && cd build
cmake ../ && sudo make install
cd ../../
export EIGEN3_DIR=/usr/local/include/

# Test Data
wget "https://cloudstor.aarnet.edu.au/plus/s/NJRXKpU30ax77uO/download" -O ./testdata/1197638568-split.tar.gz
tar -C ./testdata/ -xvf ./testdata/1197638568-32.tar.gz

wget "https://cloudstor.aarnet.edu.au/plus/s/Eb65Nqy66hUE2tO/download" -O ./testdata/1197638568-split.tar.gz
tar -C ./testdata/ -xvf ./testdata/1197638568-split.tar.gz

