
set(TEST_TARGET LeapAccelerateTests)

add_executable(${TEST_TARGET} tests.cc tests.cu)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( ${TEST_TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(${TEST_TARGET} gtest)
target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)