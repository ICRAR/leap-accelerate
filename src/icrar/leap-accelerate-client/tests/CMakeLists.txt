
set(TEST_TARGET LeapAccelerateClient.Tests)

add_executable(${TEST_TARGET}
    main.cc
    cuda_tests.cu
    casacore_tests.cc
    PhaseRotateTests.cc
)

message(${gtest_SOURCE_DIR})

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
