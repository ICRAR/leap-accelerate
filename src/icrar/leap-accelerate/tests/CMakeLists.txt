
set(TEST_TARGET LeapAccelerate.Tests)

set(sources
  main.cc
  casacore_tests.cc
  eigen_tests.cc
  PhaseRotateTests.cc

  vector_tests.cc
  matrix_tests.cc
)

set(cuda_sources
  cuda_tests.cu
)

# Only CUDA 10+ is capable of compiling gtest
if(CUDA_VERSION_MAJOR GREATER 9)
  list(APPEND sources ${cuda_sources})
endif()

add_executable(${TEST_TARGET}    
  ${sources}
)

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
