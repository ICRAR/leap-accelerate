# Configure a header file to pass some of the CMake settings
# to the source code
configure_file(config.h.in config.h)

set(TARGET_NAME LeapCalLib)

set(public_headers
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
  main.h
)
set(private_headers
  helper.h
)
set(sources
  main.cc
)

add_library(
  ${TARGET_NAME}
    ${sources}
    ${private_headers}
    ${public_headers}
)
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  $<INSTALL_INTERFACE:include>
)
add_library(${SOLUTION_NAME_LOWER}::${TARGET_NAME} ALIAS ${TARGET_NAME})

if (BUILD_TESTING)
  add_subdirectory(tests)
endif()

install(
  TARGETS
    ${TARGET_NAME}
  EXPORT
    ${SOLUTION_NAME_LOWER}-targets
  LIBRARY DESTINATION
    lib
  ARCHIVE DESTINATION
    lib
)

install(
  FILES
    ${public_headers}
  DESTINATION
    include/icrar/${SOLUTION_NAME_LOWER}
)
