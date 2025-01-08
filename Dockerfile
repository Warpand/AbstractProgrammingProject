FROM ubuntu:22.04
LABEL Desription="Abstract Programming Project Build"

ENV HOME /root

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    g++ \
    cmake \
    gdb \
    wget

RUN cd ${HOME} && \
    wget --no-check-certificate --quiet \
        https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz && \
        tar xzf ./boost_1_86_0.tar.gz && \
        cd ./boost_1_86_0 && \
        ./bootstrap.sh && \
        ./b2 install && \
        cd .. && \
        rm -rf ./boost_1_86_0 && \
        rm boost_1_86_0.tar.gz
