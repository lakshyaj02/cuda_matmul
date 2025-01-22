cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_mm)
set(CMAKE_CXX_STANDARD 17)

set(Caffe2_DIR "/home/lj9979/anaconda3/envs/blocksparse/lib/python3.11/site-packages/torch/include/caffe2")

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(pybind11)

# pybind11_add_module(custom_mm custom_mm.cpp)

add_executable(custom_mm custom_mm.cpp)
target_link_libraries(custom_mm "${PYTHON_LIBRARIES}")
target_link_libraries(custom_mm "${TORCH_LIBRARIES}")
target_link_libraries(custom_mm CUDA::cusparse)
set_property(TARGET custom_mm PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET custom_mm
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:custom_mm>)
endif (MSVC)
