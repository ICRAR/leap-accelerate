
set(TEST_TARGET_NAME LeapAccelerateClient.Tests)

add_executable(${TEST_TARGET_NAME}
    main.cc
    cuda_tests.cu
    casacore_tests.cc
    PhaseRotateTests.cc
)

configure_cuda_warnings(${TEST_TARGET_NAME})

target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
