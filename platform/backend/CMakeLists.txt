cmake_minimum_required(VERSION 3.20)

project(slm_platform_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Drogon CONFIG REQUIRED)

add_executable(
    slm_platform_backend
    src/main.cpp
)

target_link_libraries(
    slm_platform_backend
    PRIVATE
    Drogon::Drogon
)