# Docker image build and usage

The following procedure will generate a small docker image containing just the bare minimum binary and libraries to run LeapAccelerateCLI. Start with cloning out the repository:

    git clone https://gitlab.com/ska-telescope/icrar-leap-accelerate.git

and then change into the directory:

    cd icrar-leap-accelerate

All the following commands assume that you are in the root directory of the repository.

## Docker image build

The Dockerfile builds the image from scratch, but that takes pretty long. Depending on the network connection this build can take a long time. It is downloading the CUDA tool chain which is about 2.7 GB. After the download the unpacking and installation takes significant time in addition.

    docker build . --tag icrar/leap_cli:big

Typically, after the first build, subsequent builds are much faster.

### Stripping the image

Due to the size of the CUDA tool chain the initial image created by the initial build is very large (˜ 6GB). In order to strip this down to a reasonable size another step is recommended, which reduces the docker image size by more than a factor of 100.

In order to clean this up, it is highly recommended to run the tool from <https://github.com/mvanholsteijn/strip-docker-image.git>

    cd .. ; git clone https://github.com/mvanholsteijn/strip-docker-image.git; cd icrar-leap-accelerate

and then

    ../strip-docker-image/bin/strip-docker-image -i icrar/leap_cli:big -t icrar/leap_cli:`cat version.txt` -f /usr/local/bin/LeapAccelerateCLI -f /bin/bash -f /usr/bin/cat -f /usr/bin/ls -f /etc/passwd -f /home/ray

The resulting image is less than 50MB and contains just the required binary.

### Testing the image

From the main directory of the leap_accelarate checkout run install.sh in the testdata directory:

    cd testdata; ./install.sh; cd ..

and then in the main directory of leap_accelarate:

    docker run -w /testdata --user ray -v "$(pwd)"/testdata:/testdata icrar/leap_cli:`cat version.txt` LeapAccelerateCLI -f /testdata/mwa/1197638568-split.ms -i cpu -d "[[-0.4606549305661674,-0.29719233792392513]]"

The output should be a JSON data structure.

You can also use a configuration file to run the same test run:

    docker run -w /testdata --user ray -v "$(pwd)"/testdata:/testdata icrar/leap_cli:`cat version.txt` LeapAccelerateCLI --config /testdata/mwa_test.json

In this case the output will be generated in a file called testdata/mwa_cal.out.
NOTE: The tests above are using the CPU implementation of the algorithm.

## Pulling and testing the docker image

NOTE: This part of the guide still assumes that you have cloned the repository and are located in the root directory of the repo, but that is only required if you want to run the test, else the docker pull is sufficient.

The leap-accelerate docker image is also available on dockerhub. Using that is very straight forward:

    docker pull icrar/leap_cli:`cat version.txt`

The testing procedure is still the same. Download and unpack the testdata

    cd testdata; ./install.sh; cd ..

and then execute the actual test:

    docker run -w /testdata --user ray -v "$(pwd)"/testdata:/testdata icrar/leap_cli:`cat version.txt` LeapAccelerateCLI -f /testdata/mwa/1197638568-split.ms -i cpu -d "[[-0.4606549305661674,-0.29719233792392513]]"

