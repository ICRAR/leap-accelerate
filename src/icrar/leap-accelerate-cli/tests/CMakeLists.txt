
set(TEST_TARGET_NAME LeapAccelerateCLI.Tests)

set(sources
  main.cc

  CommandLineTests.cc
  JSONHelperTests.cc    
  E2EPerformanceTests.cc
)

add_executable(${TEST_TARGET_NAME}
  ${sources}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
# TODO(calgray): clang-tidy tries to analyze pch systems headers
#  target_precompile_headers(${TEST_TARGET_NAME}
#    PRIVATE
#    [[pch.h]]
#  )
endif()

configure_cuda_warnings(${TEST_TARGET_NAME})

target_link_libraries(${TEST_TARGET_NAME} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET_NAME}
  COMMAND ${TEST_TARGET_NAME}
)
