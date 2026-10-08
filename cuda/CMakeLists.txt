# CUDA port (float only). Configure with -DBT_CUDA=ON; the architecture
# defaults to the GPU in this machine (CMake >= 3.24), override with
# -DCMAKE_CUDA_ARCHITECTURES=86 etc.
find_package(CUDAToolkit REQUIRED)

add_library(bt_cuda STATIC gpu_render.cu)
target_link_libraries(bt_cuda PUBLIC bt_f CUDA::cudart)
target_include_directories(bt_cuda PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)
set_target_properties(bt_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(bt_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

add_executable(bt_render_cuda ${CMAKE_SOURCE_DIR}/apps/bt_render.cpp)
target_compile_definitions(bt_render_cuda PRIVATE BT_WITH_CUDA)
target_link_libraries(bt_render_cuda bt_cuda)

if(BT_TESTS)
  add_executable(bt_tests_cuda ${CMAKE_SOURCE_DIR}/tests/test_main.cpp test_cuda.cpp)
  target_link_libraries(bt_tests_cuda bt_cuda)
  add_test(NAME tests_cuda COMMAND bt_tests_cuda)
endif()
