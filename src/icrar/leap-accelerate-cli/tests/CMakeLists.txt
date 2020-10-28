
set(TEST_TARGET LeapAccelerateCLI.Tests)

set(sources
  main.cc

  CommandLineTests.cc
  JSONHelperTests.cc    
  E2EPerformanceTests.cc
)

#configure_file("testdata")

add_executable(${TEST_TARGET}
  ${sources}
)

# if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
#   target_precompile_headers(${TEST_TARGET}
#     PRIVATE
#       [[pch.h]]
#   )
# endif()

# CUDA Options
if(CUDA_VERSION_STRING VERSION_GREATER 10)
  # 2829 annotation on a defaulted function is ignored
  # 3057 annotation is ignored on a function that is explicitly defaulted
  # 2929 annotation is ignored on a function that is explicitly defaulted
  target_compile_options(${TEST_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=3057,2929">)
else()
  target_compile_options(${TEST_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=2829">)
endif()

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
