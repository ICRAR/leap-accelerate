
set(TEST_TARGET LeapAccelerate.Tests)

set(sources
  main.cc
  
  math/casacore_eigen_tests.cc
  math/eigen_tests.cc
  math/vector_tests.cc
  math/matrix_tests.cc
  
  model/MetaDataTests.cc
  
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

#configure_file("test_data")

add_executable(${TEST_TARGET}
  ${sources}
)

target_precompile_headers(${TEST_TARGET}
  PRIVATE
    [[pch.h]]
)

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
