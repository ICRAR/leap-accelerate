
set(TEST_TARGET LeapAccelerateCLI.Tests)

set(sources
  main.cc

  CommandLineTests.cc
  JSONHelperTests.cc    
  E2EPerformanceTests.cc
)

add_executable(${TEST_TARGET}
  ${sources}
)

# if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
#   target_precompile_headers(${TEST_TARGET}
#     PRIVATE
#       [[pch.h]]
#   )
# endif()

configure_cuda_warnings(${TARGET_NAME})

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
