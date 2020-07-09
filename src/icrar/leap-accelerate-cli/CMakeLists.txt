set(TARGET_NAME LeapAccelerateCLI)
set(TARGET_EXPORT leapaccelerate-targest)
set(TARGET_TEST LeapAccelerateCLIExec)

set(sources
  main.cc
)

# TODO: use solution directory macro
include_directories(../../../external/CLI11/include)

# add the executable
add_executable (${TARGET_NAME} main.cc)
target_link_libraries (${TARGET_NAME} LeapAccelerate)

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

install (
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${TARGET_EXPORT}
  DESTINATION
    bin
)

#add_test(NAME ${TARGET_TEST} COMMAND ${TARGET_NAME})
