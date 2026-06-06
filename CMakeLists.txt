cmake_minimum_required(VERSION 3.12)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(LEASTSQUARES_MATCHING)
set(CMAKE_CXX_STANDARD 14)
set(OPENCV_INSTALL_DIR "${CMAKE_BINARY_DIR}/opencv_install")
set(OpenCV_DIR "${OPENCV_INSTALL_DIR}")
if(NOT EXISTS "${OPENCV_INSTALL_DIR}/OpenCVConfig.cmake")
    message(STATUS "OpenCV not found, building from source...")
    set(OPENCV_BUILD_DIR "${CMAKE_BINARY_DIR}/opencv_build")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_SOURCE_DIR}/third_party/src/opencv
                -B ${OPENCV_BUILD_DIR}
                -G "Visual Studio 17 2022"
                -D CMAKE_CXX_FLAGS="/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
                -D CMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
                -D BUILD_EXAMPLES=OFF
                -D BUILD_TESTS=OFF
                -D BUILD_DOCS=OFF
                -D BUILD_opencv_videoio=OFF
                -D BUILD_opencv_video=OFF
                -D BUILD_opencv_calib3d=OFF
                -D BUILD_opencv_features2d=OFF
                -D BUILD_opencv_flann=OFF
                -D BUILD_opencv_objdetect=OFF
                -D BUILD_opencv_photo=OFF
                -D BUILD_opencv_stitching=OFF
                -D BUILD_opencv_dnn=OFF
                -D BUILD_opencv_ml=OFF
                -D BUILD_opencv_gapi=OFF
                -D WITH_FFMPEG=OFF
                -D WITH_TIFF=OFF
                -D WITH_WEBP=OFF
                -D WITH_JASPER=OFF
                -D WITH_OPENEXR=OFF
                -D WITH_IPP=OFF
                -D WITH_ADE=OFF
                -D WITH_QUIRC=OFF
                -D WITH_1394=OFF
                -D WITH_PROTOBUF=OFF
                -D CPU_DISPATCH=
                -D BUILD_JAVA=OFF
                -D BUILD_opencv_python3=OFF
                -D OPENCV_ENABLE_NONFREE=OFF
                -D BUILD_ZLIB=ON
                -D BUILD_libpng=ON
                -D BUILD_JPEG=ON
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to configure OpenCV")
    endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${OPENCV_BUILD_DIR} --target install --config Release
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to build OpenCV")
    endif()
endif()
find_package(OpenCV REQUIRED)
add_executable(leastsquares_matching
    core/src/matching.cpp
    core/src/test.cpp
    core/src/matrix.cpp
    core/src/tools.cpp
    core/src/Correlation.cpp
    core/src/Extract.cpp
)
target_link_libraries(leastsquares_matching PRIVATE ${OpenCV_LIBS})
target_include_directories(leastsquares_matching PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)

if(WIN32)
    file(GLOB OPENCV_DLLS "${OpenCV_DIR}/x64/vc*/bin/opencv_core*.dll"
                         "${OpenCV_DIR}/x64/vc*/bin/opencv_imgproc*.dll"
                         "${OpenCV_DIR}/x64/vc*/bin/opencv_imgcodecs*.dll"
                         "${OpenCV_DIR}/x64/vc*/bin/opencv_highgui*.dll")
    add_custom_command(TARGET leastsquares_matching POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DLLS} $<TARGET_FILE_DIR:leastsquares_matching>
    )
endif()