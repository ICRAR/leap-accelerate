
set(TARGET_NAME LeapAccelerate.Tests)

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

add_executable(${TARGET_NAME}
  ${sources}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
  target_precompile_headers(${TARGET_NAME}
    PRIVATE
      [[pch.h]]
  )
endif()

# CUDA Options
target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)
target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)

if(CUDA_VERSION_STRING VERSION_GREATER 10)
  # 2829 annotation on a defaulted function is ignored
  # 3057 annotation is ignored on a function that is explicitly defaulted
  # 2929 annotation is ignored on a function that is explicitly defaulted
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=3057,2929">)
else()
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=2829">)
endif()

target_link_libraries(${TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TARGET_NAME}
  COMMAND ${TARGET_NAME}
)

#gtest_add_tests(
#  TARGET ${TARGET_NAME}
#  TEST_PREFIX "*"
#  EXTRA_ARGS --gtest_stack_trace_depth=10
#)