
set(TEST_TARGET LeapAccelerate.Tests)

add_executable(${TEST_TARGET}
    main.cc
    cuda_tests.cu
    casacore_tests.cc
    eigen_tests.cc
    PhaseRotateTests.cu
)

message(${gtest_SOURCE_DIR})

if(USE_CUDA)
    # We need to explicitly state that we need all CUDA files in the target
    # library to be built with -dc as the member functions could be called by
    # other libraries and executables
    set_property(TARGET ${TARGET_NAME} CUDA_STANDARD 11)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    set_target_properties(${THE_TARGET} PROPERTIES COMPILE_FLAGS -lcuda -lcudart)

    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
    
    link_directories(
      /usr/local/cuda/targets/x86_64-linu/lib/
    )
    
    target_link_libraries(${TARGET_NAME} PUBLIC CUDA)
endif()

target_link_libraries(${TEST_TARGET} LeapAccelerate gtest)

add_test(
  NAME ${TEST_TARGET}
  COMMAND ${TEST_TARGET}
)
