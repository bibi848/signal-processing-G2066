pybind11_add_module(tfm_cpp
    tfm.cpp
    bindings.cpp
)

target_include_directories(tfm_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)