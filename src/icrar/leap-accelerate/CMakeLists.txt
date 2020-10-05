# Configure a header file to pass some of the CMake settings
# to the source code
configure_file(config.h.in config.h)

set(TARGET_NAME LeapAccelerate)

set(public_headers
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
    
    model/visibility.h
    model/band_data.h
    model/casa/MetaData.h

    algorithm/casa/PhaseRotate.h
)
set(private_headers

    math/casacore_helper.h
    math/math_conversion.h
    math/math.h
)
set(sources
    exception/exception.cc

    ms/utils.cc
    ms/MeasurementSet.cc

    model/casa/Integration.cc
    model/casa/MetaData.cc

    model/cpu/Integration.cc
    model/cpu/MetaData.cc
    model/cpu/CalibrateResult.cc
    
    math/math_conversion.cc
    math/casa/vector.cc
    math/casa/matrix.cc
    math/casa/matrix_invert.cc
    math/cpu/vector.cc

    core/compute_implementation.cc
    core/logging.cc

    json/json_helper.cc
    
    algorithm/casa/PhaseRotate.cc
    algorithm/cpu/PhaseRotate.cc
)

set(cuda_headers
    cuda/cuda_info.h
    cuda/helper_cuda.cuh
    cuda/helper_string.cuh

    math/cuda/vector.h
    math/cuda/vector.cuh
    math/cuda/matrix.h
    math/cuda/matrix.cuh
    math/cuda/matrix_multiply.cuh
    math/cuda/matrix_transpose.cuh

    algorithm/cuda/PhaseRotate.h
)

set(cuda_sources
    cuda/cuda_info.cu
    model/cuda/DeviceMetaData.cu
    model/cuda/DeviceIntegration.cu

    math/cuda/vector.cu
    math/cuda/matrix.cu

    algorithm/cuda/PhaseRotate.cu
)

# Libraries
option(CASA_ENABLED "Casa Enabled" TRUE)
option(GSL_ENABLED "GSL Enabled" OFF)

add_library(
  ${TARGET_NAME} STATIC
    ${sources}
    ${cuda_sources}
    ${private_headers}
    ${public_headers}
    ${cuda_headers}
)

if(${CMAKE_VERSION} VERSION_GREATER "3.16.0")
  target_precompile_headers(${TARGET_NAME}
    PRIVATE
      [[pch.h]]
  )
endif()


# Request that the target be built with -std=c++11
# As this is a public compile feature anything that links to the target
# will also build with -std=c++11
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_14)

# We need to explicitly state that we need all CUDA files in the target
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_STANDARD 11)
#set(CUDA_PROPAGATE_HOST_FLAGS ON)
#set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_60,code=sm_60>)
target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --display_error_number>)
if(CUDA_VERSION VERSION_GREATER 10)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=3057">)
endif()
#target_link_libraries(${TARGET_NAME} CUDA)

target_include_directories(${TARGET_NAME} PUBLIC "/usr/local/cuda/include")
target_include_directories(${TARGET_NAME} PUBLIC "/usr/local/cuda/include/crt")

target_link_libraries(${TARGET_NAME} ${Boost_LIBRARIES})
target_link_libraries(${TARGET_NAME} Eigen3::Eigen)
target_link_libraries(${TARGET_NAME} RapidJSON::RapidJSON)

if(CASA_ENABLED)
  target_link_libraries(${TARGET_NAME} ${CASACORE_LIBRARIES})
  target_include_directories(${TARGET_NAME} PUBLIC ${CASACORE_INCLUDES})
endif()
if(GSL_ENABLED)
  target_link_libraries(${TARGET_NAME} GSL::gsl)
endif()

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

unset(CASA_ENABLED CACHE)
unset(GSL_ENABLED CACHE)