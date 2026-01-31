if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
elseif(CMAKE_PREFIX_PATH)
    list(GET CMAKE_PREFIX_PATH 0 ROCM_PATH)
else()
    set(ROCM_PATH "/opt/rocm")
endif()

message(STATUS "Using ROCm path: ${ROCM_PATH}")
set(HIPCC ${ROCM_PATH}/bin/hipcc)

# Manually compile the HIP kernel to an object file using hipcc
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tfm_gpu_kernel.o
    COMMAND ${HIPCC}
        -fPIC
        -c ${CMAKE_CURRENT_SOURCE_DIR}/tfm_gpu.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/tfm_gpu_kernel.o
        -I${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tfm_gpu.cpp
    COMMENT "Compiling HIP kernel with hipcc"
)

# Create the Python module
pybind11_add_module(tfm_gpu
    bindings_gpu.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/tfm_gpu_kernel.o
)

# Link HIP runtime library
target_link_libraries(tfm_gpu PRIVATE
    ${ROCM_PATH}/lib/libamdhip64.so
    OpenMP::OpenMP_CXX
)

target_include_directories(tfm_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "Building TFM GPU version")