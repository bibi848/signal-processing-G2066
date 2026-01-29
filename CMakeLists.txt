cmake_minimum_required(VERSION 3.22)

project(signal_processing_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY LAST)
set(Python_USE_STATIC_LIBS OFF)

find_package(Python 3.13 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(CPP)