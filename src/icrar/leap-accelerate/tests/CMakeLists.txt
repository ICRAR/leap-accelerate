set(TEST_TARGET LeapAccelerateTests)
add_executable(${TEST_TARGET} tests.cc)
target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)
add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)