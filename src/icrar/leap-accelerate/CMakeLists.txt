# Configure a header file to pass some of the CMake settings
# to the source code
configure_file(config.h.in config.h)

set(TARGET_NAME LeapAccelerate)

set(public_headers
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
    main.h
    chgcentre.h
    visibility.h
)
set(private_headers
    helper.h
)
set(sources
    main.cc
)

set(cuda_sources
    math/vector.cu
)

# Libraries
option(USE_CUDA "Use Cuda" ON)
option(USE_CASA "Use Casa" OFF)
option(USE_BOOST "Use Boost" OFF)

#if(USE_CUDA)
#    if(CUDA_FOUND)
#        message(STATUS "CUDA found")
#        enable_language("CUDA")
#        set(sources ${source} ${cuda_source})
#    else()
#        message(STATUS "CUDA not found")
#    endif()
#endif()

add_library(
  ${TARGET_NAME}
    ${sources}
    ${private_headers}
    ${public_headers}
)

if(USE_CUDA)
    target_link_libraries(${TARGET_NAME} CUDA)
endif()
if(USE_CASA)
    target_link_libraries(${TARGET_NAME} Casacore)
endif()
if(USE_BOOST)
target_link_libraries(${TARGET_NAME} Boost)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)


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

unset(USE_CUDA CACHE)
unset(USE_CASA CACHE)