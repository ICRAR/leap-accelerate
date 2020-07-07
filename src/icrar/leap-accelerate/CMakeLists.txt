# Configure a header file to pass some of the CMake settings
# to the source code
configure_file(config.h.in config.h)

set(TARGET_NAME LeapAccelerate)

set(public_headers
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
    visibility.h
    band_data.h
    MetaData.h

    algorithm/PhaseRotate.h
    algorithm/Invert.h
)
set(private_headers
    math/casacore_helper.h
    math/eigen_helper.h
    math/math.h

    wsclean/banddata.h
    wsclean/chgcentre.h
    wsclean/imagecoordinates.h
    wsclean/multibanddata.h
    wsclean/progressbar.h
    wsclean/radeccoord.h
)
set(sources
    utils.cc
    MetaData.cc

    algorithm/PhaseRotate.cc
    algorithm/Invert.cc

    wsclean/chgcentre.cc
    wsclean/multibanddata.cpp
    wsclean/progressbar.cpp
)

set(cuda_headers
    math/vector.cuh
    algorithm/PhaseRotate.cuh
)

set(cuda_sources
    math/vector.cu
)

# Libraries
option(USE_CUDA "Use Cuda" CUDA_FOUND)
option(USE_CASA "Use Casa" TRUE)
option(USE_BOOST "Use Boost" OFF)

if(USE_CUDA)
  list(APPEND public_headers ${cuda_headers})
  list(APPEND sources ${cuda_sources})
endif()

add_library(
  ${TARGET_NAME} STATIC
    ${sources}
    ${private_headers}
    ${public_headers}
)

# Request that the target be built with -std=c++11
# As this is a public compile feature anything that links to the target
# will also build with -std=c++11
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_11)

if(USE_CUDA)
    # We need to explicitly state that we need all CUDA files in the target
    # library to be built with -dc as the member functions could be called by
    # other libraries and executables
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_50,code=sm_50>)
    target_link_libraries(${TARGET_NAME} CUDA)
endif()
if(USE_CASA)
    message("Using casa")
    target_link_libraries(${TARGET_NAME} ${CASACORE_LIBRARIES})
endif()
if(USE_BOOST)
    target_link_libraries(${TARGET_NAME} Boost)
endif()
target_link_libraries(${TARGET_NAME} Eigen3::Eigen)

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
unset(USE_BOOST CACHE)