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
wget https://aus01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fcloudstor.aarnet.edu.au%2Fplus%2Fs%2FNJRXKpU30ax77uO%2Fdownload&amp;data=02%7C01%7Ccallan.gray%40uwa.edu.au%7Cbccd03a140084ca8067d08d858912c75%7C05894af0cb2846d8871674cdb46e2226%7C1%7C0%7C637356727822343686&amp;sdata=GNb8DDTqJU785Mgf5dJJmKn1I2L4tT6vXVo9EdMwbUU%3D&amp;reserved=0 -O ./testdata/1197638568-split.tar.gz
tar xvzf 1197638568-split.tar.gz
