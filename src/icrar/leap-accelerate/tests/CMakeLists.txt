
set(TEST_TARGET LeapAccelerate.Tests)

add_executable(${TEST_TARGET}
    main.cc
    #tests.cc
    #tests.cu
)

message(${gtest_SOURCE_DIR})

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest gtest_main)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
