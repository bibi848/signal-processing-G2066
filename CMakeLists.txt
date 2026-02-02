cmake_minimum_required(VERSION 3.22)

project(signal_processing_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY LAST)
set(Python_USE_STATIC_LIBS OFF)

find_package(Python 3.13 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(CPP/TFM)

# LOL not apple 
if(UNIX AND NOT APPLE)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    elseif(CMAKE_PREFIX_PATH)
        list(GET CMAKE_PREFIX_PATH 0 ROCM_PATH)
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
    
    # Check if hipcc exists
    find_program(HIPCC hipcc PATHS ${ROCM_PATH}/bin)
    
    if(HIPCC)
        message(STATUS "Found hipcc: ${HIPCC}")
        message(STATUS "Building GPU version")
        add_subdirectory(CPP/TFM_GPU)
    else()
        message(STATUS "hipcc not found - skipping GPU version")
    endif()
endif()