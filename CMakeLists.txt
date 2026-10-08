cmake_minimum_required(VERSION 3.18)
project(beamtrace2007 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BT_NATIVE "Compile with -march=native" ON)
option(BT_SIMD "Use SSE (float) / AVX2 (double) intrinsics for the 4 corner rays" ON)
option(BT_CUDA "Build the CUDA port" OFF)
option(BT_TESTS "Build tests" ON)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-maybe-uninitialized>)
# No implicit fused multiply-adds: otherwise the compiler fuses differently
# depending on inlining, and the depth-first tracer, its stackless variant and
# the wavefront reference engine (which share their arithmetic) would differ
# in the last bits. The cost is within noise.
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>)
if(BT_NATIVE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
find_package(Threads REQUIRED)

set(BT_SOURCES
  src/util/image.cpp
  src/util/misc.cpp
  src/scene/scene.cpp
  src/scene/loaders.cpp
  src/scene/procedural.cpp
  src/accel/kdtree.cpp
  src/beam/beam_tracer.cpp
  src/render/render.cpp
)

# The library is built twice: float (the paper's SSE configuration) and
# double (the reference used for correctness checks).
foreach(prec f d)
  add_library(bt_${prec} STATIC ${BT_SOURCES})
  target_include_directories(bt_${prec} PUBLIC src)
  target_link_libraries(bt_${prec} PUBLIC Threads::Threads)
  if(prec STREQUAL "d")
    target_compile_definitions(bt_${prec} PUBLIC BT_REAL_DOUBLE)
  endif()
  if(NOT BT_SIMD)
    target_compile_definitions(bt_${prec} PUBLIC BT_NO_SIMD)
  endif()
endforeach()

add_executable(bt_render apps/bt_render.cpp)
target_link_libraries(bt_render bt_f)
add_executable(bt_render_d apps/bt_render.cpp)
target_link_libraries(bt_render_d bt_d)
add_executable(bt_compare apps/bt_compare.cpp)
foreach(prec f d)
  add_executable(bt_golden_${prec} apps/bt_golden.cpp)
  target_link_libraries(bt_golden_${prec} bt_${prec})
endforeach()

if(BT_TESTS)
  enable_testing()
  foreach(prec f d)
    add_executable(bt_tests_${prec} tests/test_main.cpp tests/test_geometry.cpp tests/test_tracing.cpp
                                  tests/test_continuation.cpp)
    target_link_libraries(bt_tests_${prec} bt_${prec})
    target_include_directories(bt_tests_${prec} PRIVATE ${CMAKE_SOURCE_DIR})
    add_test(NAME tests_${prec} COMMAND bt_tests_${prec})
  endforeach()
endif()

if(BT_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
  endif()
  enable_language(CUDA)
  add_subdirectory(cuda)
endif()
