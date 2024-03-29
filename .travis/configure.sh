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

fail() {
	echo $1 1>&2
	exit 1
}

# Set and use system default compilers
sudo update-alternatives --set gcc $CC
sudo update-alternatives --set g++ $CXX

cd ${TRAVIS_BUILD_DIR}
mkdir build
cd ${TRAVIS_BUILD_DIR}/build

#Debug Build Configuration
CMAKE_OPTIONS="$CMAKE_OPTIONS -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DGSL_ROOT_DIR=${GSL_ROOT_DIR}"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG} -O1"
CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_BUILD_TYPE=RelWithDebInfo"
cmake .. ${CMAKE_OPTIONS} || fail "cmake failed"
cd ..
