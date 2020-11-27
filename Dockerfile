# We need the base CUDA image, just avoiding the CUDA toolkit installation
# by starting from an image containing that already
FROM cuda11.0:base

# Should not need that anymore with the new build
RUN apt update && apt install -y clang-tidy libboost1.71-all-dev libblas-dev liblapack-dev
RUN groupadd -r leap && useradd --no-log-init -r -g leap leap

COPY / /leap-accelerate
RUN cd /leap-accelerate && git submodule update --init --recursive &&\
    export CUDA_HOME=/usr/local/cuda &&\
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64 &&\
    export PATH=$PATH:$CUDA_HOME/bin &&\
    mkdir -p /leap-accelerate/build/linux/Debug &&\
    cd /leap-accelerate/build/linux/Debug &&\
    cmake ../../.. -DCMAKE_CXX_FLAGS_DEBUG=-O1 -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=OFF &&\
    make && make install

# Second stage to cleanup the mess to a certain extend
# Note: This may require quite some disk space on the host temporarily.
# If it fails try increasing the disk space allocated to docker (MacOSX)
FROM ubuntu:20.04
COPY --from=0 /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=0 /usr/local /usr/local
COPY --from=0 /bin/bash /bin/bash
RUN apt update && apt install -y liblapack3
CMD ["/usr/local/bin/LeapAccelerateCLI"]

# After this run the brilliant tool from https://github.com/mvanholsteijn/strip-docker-image.git
# 
# strip-docker-image -i icrar/20.04leap:clean -t icrar/20.04leap:stripped -f /usr/local/bin/LeapAccelerateCLI
#
# which will produce a ~50MB docker image
#
# as a basic test run the image against the testdata:
# cd <leap_source_dir>/testdata
# ./install.sh    # first install the testdata (just once)
# docker run -v "$(pwd)":/testdata icrar/leap_cli:initial LeapAccelerateCLI \
#           -f /testdata/1197638568-split.ms -s 126 -i eigen \
#           -d "[[-0.4606549305661674,-0.29719233792392513],[-0.753231018062671,-0.44387635324622354]]"

# AWS run command on instance command gives error presumably because of CUDA version mismatch.
# docker run --gpus all --rm --mount src=`pwd`,target=/testdata,type=bind icrar/leap_cli:initial 
# sh -c 'LeapAccelerateCLI -f /testdata/1197638568-split.ms -s 126 -i cuda 
# -d "[[-0.4606549305661674,-0.29719233792392513]]" > /testdata/leap.out'