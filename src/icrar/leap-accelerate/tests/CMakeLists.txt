
set(TEST_TARGET LeapAccelerate.Tests)

set(sources
  main.cc
  
  test_helper.cc

  math/casacore_eigen_tests.cc
  math/eigen_tests.cc
  math/vector_tests.cc
  math/matrix_tests.cc
  math/linear_math_tests.cc

  ms/MSUtilsTests.cc

  model/MetaDataTests.cc
  model/IntegrationTests.cc
  
  algorithm/PhaseRotateTests.cc
)

set(cuda_sources
  math/cuda/cuda_vector_tests.cu
  math/cuda/cuda_matrix_tests.cu
  math/cuda/cuda_vector_eigen_tests.cu
)

# Only CUDA 10+ is capable of compiling gtest
if(CUDA_VERSION_MAJOR GREATER 9)
  list(APPEND sources ${cuda_sources})
endif()

#configure_file("testdata")

add_executable(${TEST_TARGET}
  ${sources}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
  target_precompile_headers(${TEST_TARGET}
    PRIVATE
      [[pch.h]]
  )
endif()

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

gtest_add_tests(
  TARGET ${TEST_TARGET}
  TEST_PREFIX "*"
  EXTRA_ARGS --gtest_stack_trace_depth=10
)