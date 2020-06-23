set(TEST_TARGET chgcentre.tests)
add_executable(${TEST_TARGET} tests.cc)
target_link_libraries(${TEST_TARGET} LeapAccelerateLib gtest)
add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)