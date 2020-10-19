#!/bin/bash
#
# Markus Dolensky, ICRAR, 2020-10-01
#
# Usage: sbatch leapjob.sh [OPTIONS]
#  Options:
#  -h,--help                   Print this help message and exit
#  -s,--stations INT           Override number of stations to use in the measurement set
#  -f,--filepath TEXT          MeasurementSet file path
#  -d,--directions TEXT        Direction calibrations
#  -i,--implementation TEXT    Compute implementation type
#  -c,--config TEXT            Config filepath
#
# examples:
#  sbatch ./bin/leapjob.sh --help
#  sbatch ./bin/leapjob.sh -f ~/leap-accelerate/testdata/1197638568-32.ms -s 126 -i cuda -d 
#[[-0.4606549305661674,-0.29719233792392513],[-0.753231018062671,-0.44387635324622354],[-0.6207547100721282,-0.2539086572881469],[-0.41958660604621867,-0.03677626900108552],[-0.41108685258900596,-0.08638012622791202],[-0.7782459495668798,-0.4887860989684432],[-0.17001324965728973,-0.28595644149463484],[-0.7129444556035118,-0.365286407171852],[-0.1512764129166089,-0.21161026349648748]]
#

#SBATCH --job-name=leap
#SBATCH --partition=mlgpu
#SBATCH --nodes=1
#SBATCH --time=05:00
##SBATCH --mail-type=ALL
##SBATCH --mail-user=markus.dolensky@uwa.edu.au
##SBATCH --output=/scratch/mdolensky/leap%A.log
##SBATCH --error=/scratch/mdolensky/leap%A.err

echo "HOSTNAME: " ${HOSTNAME}
echo "SLURM_JOBID: " $SLURM_JOBID

module load cmake/3.15.1 boost/1.66.0 casacore/3.1.2
module unload gfortran/default
module load isl/default
export CUDA_HOME=/usr/local/cuda

# set $LEAP_HOME to the local installation folder; corresponds to CMAKE_INSTALL_PREFIX
export LEAP_HOME=$HOME/leap

# to run the unit tests set $LEAP_TEST to the local build folder 
export LEAP_BUILD=$HOME/leap-accelerate/build

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin:$LEAP_HOME/bin

# run unit tests
#cd $LEAP_BUILD
#ctest --verbose

# run leap command; pass through all arguments
LeapAccelerateCLI $*


