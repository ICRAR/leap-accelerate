cmake_minimum_required(VERSION 3.15)

set(SOLUTION_NAME LEAP-Accelerate)
set(SOLUTION_NAME_LOWER leap-accelerate)

file(STRINGS version.txt SOLUTION_VERSION)
message(STATUS "Building ${SOLUTION_NAME} version ${SOLUTION_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)


#==================
# Build Options
#==================

option(CUDA_ENABLED "Build with cuda support" OFF)
option(PYTHON_ENABLED "Build with python support" OFF)
set(CMAKE_CUDA_ARCHITECTURES "60;61;70;72;75;80;86" CACHE STRING "Cuda Compute and Streaming Multiprocessor level")
option(CMAKE_RUN_CLANG_TIDY "Run clang-tidy with the compiler." OFF)
option(WERROR "Warnings as Errors" OFF)
option(WCONVERSION "Conversion warnings" OFF)
option(TRACE "Trace Data Values" OFF)
option(USE_PCH "Use pre-compile headers internally when possible" ON)
option(BUILD_DOCS "Generate documentation build target" ON)

# Override config via environment variables
if(DEFINED ENV{CUDA_ENABLED})
  set(CUDA_ENABLED $ENV{CUDA_ENABLED})
  message(STATUS "Environment variable override: CUDA_ENABLED=${CUDA_ENABLED}")
endif()
if(DEFINED ENV{CMAKE_C_COMPILER})
  set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
  message(STATUS "Environment variable override: CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
endif()
if(DEFINED ENV{CMAKE_CXX_COMPILER})
  set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
  message(STATUS "Environment variable override: CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()
if(DEFINED ENV{CMAKE_CUDA_COMPILER})
  set(CMAKE_CUDA_COMPILER $ENV{CMAKE_CUDA_COMPILER})
  message(STATUS "Environment variable override: CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
endif()
if(DEFINED ENV{PYTHON_ENABLED})
  set(PYTHON_ENABLED $ENV{PYTHON_ENABLED})
  message(STATUS "Environment variable override: PYTHON_ENABLED=${PYTHON_ENABLED}")
endif()
if(DEFINED ENV{WERROR})
  set(WERROR $ENV{WERROR})
  message(STATUS "Environment variable override: WERROR=${WERROR}")
endif()
if(DEFINED ENV{TRACE})
  set(TRACE $ENV{TRACE})
  message(STATUS "Environment variable override: TRACE=${TRACE}")
endif()
if(DEFINED ENV{CMAKE_RUN_CLANG_TIDY})
  set(CMAKE_RUN_CLANG_TIDY $ENV{CMAKE_RUN_CLANG_TIDY})
  message(STATUS "Environment variable override: CMAKE_RUN_CLANG_TIDY=${CMAKE_RUN_CLANG_TIDY}")
endif()
if(DEFINED ENV{BUILD_DOCS})
  set(BUILD_DOCS $ENV{BUILD_DOCS})
  message(STATUS "Environment variable override: BUILD_DOCS=${BUILD_DOCS}")
endif()


#==================
# Solution Config
#==================

if(TRACE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTRACE")
endif()

# Pre-compile headers are only supported in cmake >= 3.16
if(NOT ${CMAKE_VERSION} VERSION_GREATER "3.16.0")
  message(STATUS "Disabling pre-compile headers (cmake version < 3.16)")
  set(USE_PCH OFF)
endif()

# Enable Linting with build
if(CMAKE_RUN_CLANG_TIDY)
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "CMake_RUN_CLANG_TIDY requires an out-of-source build!")
  endif()
  find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
  if(NOT CLANG_TIDY_COMMAND)
    message(WARNING "CMake_RUN_CLANG_TIDY is ON but clang-tidy is not found!")
    set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE)
  else()
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
  endif()
  file(SHA1 ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy clang_tidy_sha1)
  set(CLANG_TIDY_DEFINITIONS "CLANG_TIDY_SHA1=${clang_tidy_sha1}")
  unset(clang_tidy_sha1)

  # Seperate lint target
  include(lint)
  configure_file(.clang-tidy .clang-tidy COPYONLY)
endif()

# Project configuration, specifying version, languages,
# and the C++ standard to use for the whole project
project(${SOLUTION_NAME} LANGUAGES C CXX VERSION ${SOLUTION_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  if(WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
  if(WCONVERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion") # -Wsign-conversion
  endif()
endif()

# Load CMake Modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Testing
include(CTest)

# Gather External Projects
find_package(Git)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed, see https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
  option(GIT_SUBMODULE "Check submodules during build" OFF)
  if(GIT_SUBMODULE)
      message(STATUS "Submodule update")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if(NOT GIT_SUBMOD_RESULT EQUAL "0")
          message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
      endif()
  endif()
else()
  message(WARNING "git not found, skipping submodule checks")
endif()


if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/eigen-3.4.90/CMakeLists.txt")
  message(FATAL_ERROR "Submodule eigen-3.4.90 was not downloaded! Please update submodules and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/gtest-1.11.0/CMakeLists.txt")
  message(FATAL_ERROR "Submodule gtest-1.11.0 was not downloaded! Please update submodules and try again.")
endif()

# Global Libraries
if(HIPSYCL_ENABLED)
  find_package(hipSYCL CONFIG REQUIRED)
endif()

if(CUDA_ENABLED)
  find_package(CUDAToolkit 9.0)
  enable_language(CUDA)
endif()

#modify exported compile commands for linting cuda
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  # HACK: CMakeDetermineCUDACompiler.cmake does not appear to identifying clang as the
  # cuda compiler and does not run Clang-CUDA.cmake yet so this instead manually overrides
  # NVIDIA-CUDA.cmake output variables. Alternatively it may be possible to use nvcc with 
  # clang using compile command 'nvcc -ccbin clang++ ...'

  set(CMAKE_CUDA_COMPILER ${CMAKE_CXX_COMPILER})
  #message("CMAKE_CUDA_COMPILE_PTX_COMPILATION: ${CMAKE_CUDA_COMPILE_PTX_COMPILATION}")
  #set(CMAKE_CUDA_COMPILE_PTX_COMPILATION "")
  #message("CMAKE_CUDA_COMPILE_SEPERABLE_COMPILATION: ${CMAKE_CUDA_COMPILE_SEPERABLE_COMPILATION}")
  #set(CMAKE_CUDA_COMPILE_SEPERABLE_COMPILATION "")
  #message("CMAKE_CUDA_COMPILE_WHOLE_COMPILATION: ${CMAKE_CUDA_COMPILE_WHOLE_COMPILATION}")
  set(CMAKE_CUDA_COMPILE_WHOLE_COMPILATION "<CMAKE_CUDA_COMPILER> --no-cuda-version-check -nocudalib <DEFINES> <INCLUDES> <FLAGS> -x cu -c <SOURCE> -o <OBJECT>")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${PROJECT_SOURCE_DIR}/external/eigen-3.4.90")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${PROJECT_SOURCE_DIR}/external/eigen-3.4.90")

  # Clang IWYU Configuration
  # Clang will ignore explicit includes of its preconfigured include directory
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem /usr/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem /usr/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/include")
endif()

add_definitions(-DBOOST_ALL_DYN_LINK)
find_package(Boost REQUIRED
  COMPONENTS log log_setup filesystem program_options
)

set(CASACORE_FIND_QUIETLY ON)
find_package(Casacore REQUIRED
  COMPONENTS ms
)

if(PYTHON_ENABLED)
  find_package(pybind11 CONFIG REQUIRED)
  find_package(Python3 REQUIRED
    COMPONENTS Interpreter Development # NumPy
  )
endif()

# Eigen Acceleration
# find_package(BLAS REQUIRED)
# add_definitions(-DEIGEN_USE_BLAS)
# find_package(LAPACK REQUIRED)
# add_definitions(-DEIGEN_USE_LAPACKE)

# Local Libraries
#add_subdirectory(external/eigen-3.4.90) # builds eigen3 tests
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/eigen-3.4.90)
find_package(Eigen3 REQUIRED)

# Use system rapidjson if available
find_package(RapidJSON)
if(RapidJSON_FOUND)
  message(STATUS "RapidJSON: ${RapidJSON_INCLUDE_DIRS}")
else()
  message(STATUS "Loading submodule rapidjson")
  set(PC_RapidJSON_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/rapidjson-1.1.0/include)
  find_package(RapidJSON REQUIRED)
  message(STATUS "RapidJSON: ${RapidJSON_INCLUDE_DIRS}")
endif()

# Local Targets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src/icrar/leap-accelerate)
add_subdirectory(src/icrar/leap-accelerate-cli)

if (BUILD_DOCS)
add_subdirectory(docs)
endif()

# Testing Libraries
if (BUILD_TESTING)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # required for Win32
  add_subdirectory(external/gtest-1.11.0 googletest EXCLUDE_FROM_ALL)
  enable_testing()
  
  include(GoogleTest)
  mark_as_advanced(
      BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
      gmock_build_tests gtest_build_samples gtest_build_tests
      gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )
  set_target_properties(gtest PROPERTIES FOLDER external)
  set_target_properties(gtest_main PROPERTIES FOLDER external)
  set_target_properties(gmock PROPERTIES FOLDER external)
  set_target_properties(gmock_main PROPERTIES FOLDER external)
endif()

# Code Coverage
if(BUILD_TESTING)
  add_custom_target(coverage
              COMMAND ${CMAKE_CTEST_COMMAND} -T test --no-compress-output --verbose
              COMMAND mkdir -p coverage
              COMMAND gcovr -r ${PROJECT_SOURCE_DIR} -e '.*/external/.*' -e '.*/CompilerIdCXX/.*' -e '.*/tests/.*' --html --html-details -o coverage/index.html
              WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_dependencies(coverage LeapAccelerate.Tests)
endif()

# Install cmake config + version + target files
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/${SOLUTION_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Config.cmake"
  INSTALL_DESTINATION
    share/${SOLUTION_NAME_LOWER}/cmake
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Version.cmake"
  COMPATIBILITY
    AnyNewerVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${SOLUTION_NAME}Version.cmake
  DESTINATION
    share/${SOLUTION_NAME_LOWER}/cmake
  COMPONENT
    dev
)
install(EXPORT ${SOLUTION_NAME_LOWER}-targets
  FILE
  ${SOLUTION_NAME}Targets.cmake
  DESTINATION
    share/${SOLUTION_NAME_LOWER}/cmake
  COMPONENT
    dev
  NAMESPACE
  ${SOLUTION_NAME_LOWER}::
)
