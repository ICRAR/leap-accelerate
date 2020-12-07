
set(TEST_TARGET_NAME LeapAccelerate.Tests)

set(sources
  main.cc
  
  test_helper.cc

  math/eigen_helper.cc
  math/CasacoreMatrixTests.cc
  math/MatrixTests.cc
  math/VectorTests.cc
  math/MatrixTests.cc
  math/MathConversionTests.cc

  ms/MSUtilsTests.cc

  model/MetaDataTests.cc
  model/IntegrationTests.cc
  
  algorithm/PhaseRotateTests.cc
)

set(cuda_sources
  math/cuda/CudaVectorTests.cu
  math/cuda/CudaMatrixTests.cu
  math/cuda/CudaMatrixEigenTests.cu
)

# Only CUDA 10+ is capable of compiling gtest
if(CUDA_VERSION_MAJOR GREATER 9)
  list(APPEND sources ${cuda_sources})
endif()

add_executable(${TEST_TARGET_NAME}
  ${sources}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
# TODO(calgray): clang-tidy tries to analyze pch systems headers
#  target_precompile_headers(${TEST_TARGET_NAME}
#    PRIVATE
#    [[pch.h]]
#  )
endif()

# CUDA Options
target_compile_options(${TEST_TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)
configure_cuda_warnings(${TEST_TARGET_NAME})

target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET_NAME}
  COMMAND ${TEST_TARGET_NAME}
)

#gtest_add_tests(
#  TARGET ${TEST_TARGET_NAME}
#  TEST_PREFIX "*"
#  EXTRA_ARGS --gtest_stack_trace_depth=10
#)