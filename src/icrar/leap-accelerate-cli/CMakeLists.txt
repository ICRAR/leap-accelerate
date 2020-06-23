set(TARGET_NAME LeapAccelerateCLI)
set(TARGET_EXPORT leapaccelerate-targest)
set(TARGET_TEST LeapAccelerateCLIExec)

set(sources
  main.cc
)

# add the executable
add_executable (${TARGET_NAME} main.cc)
target_link_libraries (${TARGET_NAME} LeapAccelerateLib)

install (
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${TARGET_EXPORT}
  DESTINATION
    bin
)

add_test(NAME ${TARGET_TEST} COMMAND ${TARGET_NAME})
