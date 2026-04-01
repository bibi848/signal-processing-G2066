add_subdirectory(TFM)

if(HIPCC)
    add_subdirectory(TFM_GPU)
    add_subdirectory(TFM_U)
endif()