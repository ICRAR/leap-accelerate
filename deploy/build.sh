#!/bin/bash
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2019
# Copyright by UWA (in the framework of the ICRAR)
# All rights reserved
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#


print_usage() {
	echo "Usage: $0 [options]"
	echo
	echo "Options:"
	echo " -b <branch>    git branch name or release tag, defaults to master"
	echo " -c <cudahome>  CUDA base directory, defaults to CUDA_HOME"
	echo " -d             build debug version"
	echo " -D <opts>      additional cmake options; e.g. -D -DBUILD_INFO=ON; include 2nd -D"
	echo " -h             print this usage text"
	echo " -p <prefix>    prefix for installation, if not set, binaries remain in build tree"
	echo " -s <system>    target system, local(default) or hyades"
}

banner() {
	msg="** $@ **"
	echo "$msg" | sed -n '{h; x; s/./*/gp; x; h; p; x; s/./*/gp}';
}

try() {
	"$@"
	status=$?
	if [ $status -ne 0 ]; then
		echo "Command exited with status $status, aborting build now: $@" 1>&2
		exit 1
	fi
}

check_supported_values() {
	val_name=$1
	given_val=$2
	shift; shift
	for supported in "$@"; do
		if [ "$given_val" == "$supported" ]; then
			return
		fi
	done
	echo "Unsupported $val_name: $given_val" 1>&2
	echo "Supported $val_name values are: $@" 1>&2
	exit 1
}

branch=
cudahome=$CUDA_HOME
buildtype="Release"
cmake_opts=
prefix=$HOME/leap
system="local"

make_jobs=1
make_targets="LeapAccelerate LeapAccelerateCLI"

while getopts "h?b:c:dD:p:s:" opt
do
	case "$opt" in
		[h?])
			print_usage
			exit 0
			;;
		b)
			branch="$OPTARG"
			;;
		c)
			cudahome="$OPTARG"
			;;
		d)
			buildtype="Debug"
			;;
		D)
			cmake_opts="$cmake_opts $OPTARG"
			;;
		p)
			prefix="$OPTARG"
			;;
		s)
			system="$OPTARG"
			;;
		*)
			print_usage 1>&2
			exit 1
			;;
	esac
done

check_supported_values system $system local hyades

# set branch to master if unspecified
if [ -z "$branch" ]; then
	branch="master"
fi

# no CUDA, no fun
if [ -z "$cudahome" ]; then
	cudahome=/usr/local/cuda
fi
if [ ! -d "$cudahome" ]; then
	echo "CUDA not found in folder <${cudahome}>. Terminating." 1>&2
	exit 1
fi

# check access to installation directory
if [ ! -z "$prefix" ]; then
	if [ ! -d "$prefix" ]; then
		try mkdir -p "$prefix"
	fi
fi

if [ $system == "hyades" ]; then
	module load cmake/3.15.1 gcc/6.3.0 boost/1.66.0 casacore/3.1.2
	module unload gfortran/default
	module load isl/default
	
	make_jobs=8
else 
	# TODO when needed
	echo "Installation on system type <$system> is not supported. Terminating." 1>&2
	exit 1
fi

# assemble cmake options
##cmake_opts="$cmake_opts -DCUDA_HOST_COMPILER=g++"
if [ ! -z "$BLDR_CASACORE_BASE_PATH" ]; then
	cmake_opts="$cmake_opts -DCASACORE_ROOT_DIR=${BLDR_CASACORE_BASE_PATH}"
fi
cmake_opts="$cmake_opts -DCMAKE_BUILD_TYPE=${buildtype}"
if [ ! -z "$cudahome" ]; then
	cmake_opts="$cmake_opts -DCUDA_TOOLKIT_ROOT_DIR=${cudahome} -DCUDA_SDK_ROOT_DIR=${cudahome}"
fi
if [ ! -z "$prefix" ]; then
	cmake_opts="$cmake_opts -DCMAKE_INSTALL_PREFIX=$prefix"
fi

# make targets
if [ ! -z "$prefix" ]; then
	make_targets="$make_targets install"
fi


repo2dir() {
	d=`basename $1`
	echo ${d%%.git}
}

_build() {
	banner Running make -j${make_jobs}
	try make -j${make_jobs}
	banner Running make -j${make_jobs} ${make_targets}
	try make -j${make_jobs} ${make_targets}
}

_prebuild_cmake() {
	test -d build || try mkdir build
	cd build
	banner Running cmake .. "$@"
	try cmake .. "$@"
}

# Build macro:
#  arg1 git repo;
#  arg2 git branch/tag;
#  arg3 cmake options
build_and_install() {
	banner Cloning $1
	try git clone $1 --branch $2
	cd `repo2dir $1`
	# git reset --hard $2
	shift; shift

	# dependencies
	git submodule update --init --recursive

	# Build
	banner Building $srcdir
	if [ -e CMakeLists.txt ]; then
		_prebuild_cmake "$@"
		_build
		cd ../..
	else
		echo "No CMakeLists.txt in folder <$srcdir> to run cmake on. Terminating." 1>&2
		exit 1
	fi
}



# ******** 
# * main *
# ********


original_dir="$PWD"

# setup environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:${cudahome}/bin
if [ ! -z "$prefix" ]; then
	export PATH=$PATH:$prefix/bin
fi

# build
banner Installing leap-accelerate $branch
build_and_install https://gitlab.com/ska-telescope/icrar-leap-accelerate.git $branch $cmake_opts

cd $original_dir

# run binary from installation folder
if [ ! -z "$prefix" ]; then
	banner Running LeapAccelerateCLI -h
	try LeapAccelerateCLI -h
fi
