# Author: OD

pybind11_add_module(tfm_cpp
    tfm.cpp
    bindings.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tfm_cpp PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found, multi-threading enabled")
else()
    if(APPLE)
        find_path(OMP_INCLUDE_DIR omp.h
            PATHS
                /opt/homebrew/opt/libomp/include
                /usr/local/opt/libomp/include
        )
        find_library(OMP_LIBRARY omp
            PATHS
                /opt/homebrew/opt/libomp/lib
                /usr/local/opt/libomp/lib
        )
        if(OMP_INCLUDE_DIR AND OMP_LIBRARY)
            message(STATUS "Homebrew libomp found -- multi-threading enabled")
            target_include_directories(tfm_cpp PRIVATE ${OMP_INCLUDE_DIR})
            target_link_libraries(tfm_cpp PRIVATE ${OMP_LIBRARY})
            target_compile_options(tfm_cpp PRIVATE -Xpreprocessor -fopenmp)
        else()
            message(WARNING "OpenMP not found.")
        endif()
    else()
        message(WARNING "OpenMP not found, building single-threaded.")
    endif()
endif()

target_include_directories(tfm_cpp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "Building TFM CPU version")