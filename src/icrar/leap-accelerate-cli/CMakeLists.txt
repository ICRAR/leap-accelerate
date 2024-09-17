set(TARGET_NAME LeapAccelerateCLI)
set(TARGET_EXPORT leapaccelerate-target)

set(sources
  main.cc
)

# add the executable
add_executable(
  ${TARGET_NAME}
  ${sources}
)

# Legacy TODO: build support for xenial
#include_directories(${CMAKE_SOURCE_DIR}/external/rapidjson-1.1.0/include)

target_link_libraries(${TARGET_NAME} LeapAccelerate)
target_link_libraries(${TARGET_NAME} RapidJSON::RapidJSON)

set_target_properties(${TARGET_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

install (
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${TARGET_EXPORT}
  DESTINATION
    bin
)

message(STATUS "Configured ${TARGET_NAME}")

if (BUILD_TESTING)
  add_subdirectory(tests)
endif()