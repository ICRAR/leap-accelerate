
set(TEST_TARGET LeapAccelerate.Tests)

add_executable(${TEST_TARGET}
    main.cc
    casacore_tests.cc
    eigen_tests.cc
    PhaseRotateTests.cc

    cuda/cuda_tests.cu
)

message(${gtest_SOURCE_DIR})

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
