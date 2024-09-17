# This Dockerfile installs everything from scratch into a Ubuntu 20.04 based container
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as buildenv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    gnupg2 git wget gcc g++ gdb doxygen cmake pipx pybind11-dev python3-pybind11 python3-numpy libopenblas-dev \
    casacore-dev libboost1.74-all-dev software-properties-common
RUN pipx run poetry==1.6.1 config virtualenvs.in-project true

# Get LEAP native sources and build
COPY CMakeLists.txt version.txt .clang-tidy /leap-accelerate/
COPY src/ /leap-accelerate/src/
COPY cmake/ /leap-accelerate/cmake/
COPY external/ /leap-accelerate/external/
COPY python/pyproject.toml python/build.py python/poetry.lock python/README.md /leap-accelerate/python/
COPY python/leap/__init__.py /leap-accelerate/python/leap/

# Python build args
ENV CUDA_ENABLED=TRUE
RUN cd /leap-accelerate/python &&\
    pipx run poetry==1.6.1 build

# Get LEAP python sources and bundle into the build
COPY ./ /leap-accelerate/
RUN cd /leap-accelerate/python &&\
    pipx run poetry==1.6.1 build &&\
    . .venv/bin/activate &&\
    pip install .

# Final stage is a fresh layer with only installed files
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 as runtime
RUN apt-get update && apt-get install -y python3
COPY --from=buildenv /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=buildenv /etc/alternatives /etc/alternatives
COPY --from=buildenv /usr/local/cuda/lib64/libnvJitLink.so.12 /usr/local/cuda/lib64/
COPY --from=buildenv /usr/local/cuda/lib64/libcublas.so.12 /usr/local/cuda/lib64/
COPY --from=buildenv /usr/local/cuda/lib64/libcusolver.so.11 /usr/local/cuda/lib64/
COPY --from=buildenv /usr/local/cuda/lib64/libcublasLt.so.12 /usr/local/cuda/lib64/
COPY --from=buildenv /usr/local/cuda/lib64/libcusparse.so.12 /usr/local/cuda/lib64/
COPY --from=buildenv /leap-accelerate/python/.venv /usr/local/.venv/
RUN ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1

ENV PATH="/usr/local/.venv/bin:${PATH}"

# add a user to run this container rather than using root
RUN useradd leap
USER leap
WORKDIR /home/leap
CMD ["python3", "-m", "pip", "show", "leap"]
