#!/bin/bash
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia
# Copyright by UWA(in the framework of the ICRAR)
# All rights reserved
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA  02110-1301  USA
#

#
# Usage: sbatch submit.sh [LEAP_ARGS]
#   for a list of LEAP_ARGS see the output of command LeapAccelerateCLI -h 
#
# examples:
#  sbatch ./bin/submit.sh --help
#  sbatch ./bin/submit.sh -f ~/leap-accelerate/testdata/mwa/1197638568-split.ms -s 126 -i cuda -d 
#[[-0.4606549305661674,-0.29719233792392513],[-0.753231018062671,-0.44387635324622354],[-0.6207547100721282,-0.2539086572881469],[-0.41958660604621867,-0.03677626900108552],[-0.41108685258900596,-0.08638012622791202],[-0.7782459495668798,-0.4887860989684432],[-0.17001324965728973,-0.28595644149463484],[-0.7129444556035118,-0.365286407171852],[-0.1512764129166089,-0.21161026349648748]]
#  

#SBATCH --job-name=leap
#SBATCH --gres=gpu
#SBATCH --nodes=1
#SBATCH --time=05:00

# customize and uncomment to generate additional outputs
##SBATCH --mail-type=ALL
##SBATCH --mail-user=first.last@icrar.org
##SBATCH --output=/scratch/myuser/leap%A.log
##SBATCH --error=/scratch/myuser/leap%A.err

echo "HOSTNAME: " ${HOSTNAME}
echo "SLURM_JOBID: " $SLURM_JOBID

module load cmake/3.15.1 boost/1.66.0 casacore/3.1.2
module unload gfortran/default
module load isl/default
export CUDA_HOME=/usr/local/cuda

# set $LEAP_HOME to the local installation folder; corresponds to CMAKE_INSTALL_PREFIX
export LEAP_HOME=$HOME/leap

# to run the unit tests set $LEAP_TEST to the local build folder 
export LEAP_BUILD=$HOME/leap/build

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin:$LEAP_HOME/bin

# run unit tests
#cd $LEAP_BUILD
#ctest --verbose

# run leap command and pass through command line arguments
$LEAP_HOME/bin/LeapAccelerateCLI $*


