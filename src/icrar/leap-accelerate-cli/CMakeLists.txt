set(TARGET_NAME LeapAccelerateCLI)
set(TARGET_EXPORT leapaccelerate-targest)

set(sources
  main.cc
)

include_directories(${CMAKE_SOURCE_DIR}/external/CLI11/include)

# add the executable
add_executable (${TARGET_NAME} main.cc)
target_link_libraries (${TARGET_NAME} LeapAccelerate)
set_target_properties(${TARGET_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (BUILD_TESTING)
  add_subdirectory(tests)
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
