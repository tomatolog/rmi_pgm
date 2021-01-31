FROM rikorose/gcc-cmake:gcc-9
RUN apt update
RUN apt install -y zstd mpi
