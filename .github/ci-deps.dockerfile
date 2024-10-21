FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -q update \
&& apt-get -qy install --no-install-recommends --no-install-suggests \
gcc \
g++ \
binutils \
make \
cmake \
git \
python3 \
python3-pip \
libboost-all-dev

RUN apt-get -qy install --no-install-recommends --no-install-suggests \
wget \
lsb-release \
software-properties-common \
gnupg

RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16
RUN apt-get -qy install libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev lld-16

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp
RUN mkdir AdaptiveCpp/build

RUN cmake -B AdaptiveCpp/build -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp AdaptiveCpp
RUN cd AdaptiveCpp/build && make install -j