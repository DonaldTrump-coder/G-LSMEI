cmake_minimum_required(VERSION 3.12)
project(lsmatching)

set(CMAKE_CXX_STANDARD 14)
add_compile_definitions(HAS_CUDA)

find_package(CUDAToolkit REQUIRED)
set(OpenCV_DIR "../build/opencv_install")
find_package(OpenCV REQUIRED)
add_subdirectory("../third_party/pybind11" pybind11_build)
add_compile_options(/wd4996)

set(CUDA_ARCHS "-gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_86,code=sm_86")
set(CUDA_INC "-I${CMAKE_SOURCE_DIR}/../core/include -I${CMAKE_SOURCE_DIR}/../core/cuda")
set(CUDA_INC "${CUDA_INC} -I${CUDAToolkit_INCLUDE_DIRS} -I${OpenCV_INCLUDE_DIRS}")

foreach(CU_FILE extract_cuda correlation_cuda matching_cuda)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CU_FILE}.obj
        COMMAND ${CUDAToolkit_BIN_DIR}/nvcc.exe
            -c
            ${CMAKE_SOURCE_DIR}/../core/cuda/${CU_FILE}.cu
            -o ${CMAKE_CURRENT_BINARY_DIR}/${CU_FILE}.obj
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_86,code=sm_86
            -DHAS_CUDA
            -std=c++14
            -O2
            -I${CMAKE_SOURCE_DIR}/../core/include
            -I${CMAKE_SOURCE_DIR}/../core/cuda
            -I${CUDAToolkit_INCLUDE_DIRS}
            -I${OpenCV_INCLUDE_DIRS}
            --compiler-options /wd4996,/EHsc,/MD
        DEPENDS ${CMAKE_SOURCE_DIR}/../core/cuda/${CU_FILE}.cu
    )
    list(APPEND CUDA_OBJS ${CMAKE_CURRENT_BINARY_DIR}/${CU_FILE}.obj)
endforeach()

pybind11_add_module(_lsmatching
    bindings.cpp
    ../core/src/matching.cpp
    ../core/src/matrix.cpp
    ../core/src/tools.cpp
    ../core/src/Correlation.cpp
    ../core/src/Extract.cpp
    ${CUDA_OBJS}
)

target_include_directories(_lsmatching PRIVATE
    ../core/include
    ../core/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(_lsmatching PRIVATE
                        ${OpenCV_LIBS}
                        ${CUDAToolkit_LIBRARY_DIR}/cudart.lib
                        )

install(TARGETS _lsmatching
        LIBRARY DESTINATION "lsmatching"
        RUNTIME DESTINATION "lsmatching")
install(FILES ./package/__init__.py
        DESTINATION "lsmatching")
install(FILES ./package/version.py
        DESTINATION "lsmatching")
install(FILES ./package/README_PyPI.md
        DESTINATION "lsmatching")

if(WIN32)
    file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/vc*/bin/*.dll")
    if(OPENCV_DLLS)
        install(FILES ${OPENCV_DLLS} DESTINATION "lsmatching")
    endif()
    file(GLOB CUDA_DLL "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
    if(CUDA_DLL)
        install(FILES ${CUDA_DLL} DESTINATION "lsmatching")
    endif()
endif()