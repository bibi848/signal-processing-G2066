if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
elseif(CMAKE_PREFIX_PATH)
    list(GET CMAKE_PREFIX_PATH 0 ROCM_PATH)
else()
    set(ROCM_PATH "/opt/rocm")
endif()

message(STATUS "[TFM_U] Using ROCm path: ${ROCM_PATH}")
set(HIPCC ${ROCM_PATH}/bin/hipcc)

# Compile the HIP kernel
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tfm_ultra_kernel.o
    COMMAND ${HIPCC}
        -fPIC
        -O3
        -c ${CMAKE_CURRENT_SOURCE_DIR}/tfm_ultra.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/tfm_ultra_kernel.o
        -I${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tfm_ultra.cpp
    COMMENT "[TFM_U] Compiling HIP batch kernel with hipcc"
)

# Pybind11 Module
pybind11_add_module(tfm_ultra
    bindings_ultra.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/tfm_ultra_kernel.o
)

target_link_libraries(tfm_ultra PRIVATE
    ${ROCM_PATH}/lib/libamdhip64.so
    OpenMP::OpenMP_CXX
)

target_include_directories(tfm_ultra PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "[TFM_U] Building TFM ULTRA GPU batch module")
