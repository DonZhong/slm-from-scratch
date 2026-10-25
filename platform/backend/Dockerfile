
# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# 允许 APT 下载缓存跨构建复用
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
       > /etc/apt/apt.conf.d/keep-cache

# 先使用 Ubuntu 原有的软件源安装 CA 根证书
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        ca-certificates

# 证书安装完成后，再切换到 HTTPS 软件源
RUN sed -i \
    's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
    /etc/apt/sources.list.d/ubuntu.sources

# 安装 C++ 编译和 Drogon 开发依赖
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Retries=5 -o APT::Update::Error-Mode=any update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential \
        cmake \
        libdrogon-dev \
        libjsoncpp-dev \
        uuid-dev \
        zlib1g-dev \
        libpq-dev \
        libsqlite3-dev \
        libmariadb-dev \
        libbrotli-dev \
        libhiredis-dev \
        libyaml-cpp-dev

WORKDIR /src

COPY CMakeLists.txt ./
COPY src/ ./src/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel 2


FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        libdrogon1t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build \
    /src/build/slm_platform_backend \
    ./slm_platform_backend

USER 10001:10001

EXPOSE 8080

CMD ["./slm_platform_backend"]
