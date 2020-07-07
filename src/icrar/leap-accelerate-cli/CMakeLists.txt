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

install (
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${TARGET_EXPORT}
  DESTINATION
    bin
)

#add_test(NAME ${TARGET_TEST} COMMAND ${TARGET_NAME})