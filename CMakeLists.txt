cmake_minimum_required(VERSION 3.8)

set(SOLUTION_NAME LEAP-Accelerate)
set(SOLUTION_NAME_LOWER leap-accelerate)

file(STRINGS version.txt SOLUTION_VERSION)
message(STATUS "Building ${SOLUTION_NAME} version ${SOLUTION_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(WERROR "Warnings as Errors" ON)
option(TRACE "Trace Data Values" OFF)

# Override config via environment variables
if(DEFINED ENV{WERROR})
  set(WERROR $ENV{WERROR})
endif()
if(DEFINED ENV{TRACE})
  set(TRACE $ENV{TRACE})
endif()

# Project configuration, specifying version, languages,
# and the C++ standard to use for the whole project
project(${SOLUTION_NAME} LANGUAGES CXX VERSION ${SOLUTION_VERSION})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  if(WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif()

if(TRACE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DTRACE")
endif()

add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/testdata")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/cmake-modules)

include(CTest)

# Gather External Projects
find_package(Git)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed, see https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
  option(GIT_SUBMODULE "Check submodules during build" ON)
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
  message(WARNING "git not found!")
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/cmake-modules/dependencies.cmake")
  message(FATAL_ERROR "Submodule cmake-modules was not downloaded! Please update submodules and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/CLI11/CMakeLists.txt")
  message(FATAL_ERROR "Submodule CLI11 was not downloaded! Please update submodules and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/eigen-3.3.90/CMakeLists.txt")
  message(FATAL_ERROR "Submodule eigen-3.3.90 was not downloaded! Please update submodules and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/gtest-1.8.1/CMakeLists.txt")
  message(FATAL_ERROR "Submodule gtest-1.8.1 was not downloaded! Please update submodules and try again.")
endif()
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/rapidjson-1.1.0/CMakeLists.txt")
  message(FATAL_ERROR "Submodule rapidjson-1.1.0 was not downloaded! Please update submodules and try again.")
endif()


if (BUILD_TESTING)
  #find_package(GTest)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # required for Win32
  add_subdirectory(external/gtest-1.8.1 googletest EXCLUDE_FROM_ALL)
  enable_testing()
  
  include(GoogleTest)
  mark_as_advanced(
      BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
      gmock_build_tests gtest_build_samples gtest_build_tests
      gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )
  set_target_properties(gtest PROPERTIES FOLDER external)
  #set_target_properties(gtest_main PROPERTIES FOLDER external)
  set_target_properties(gmock PROPERTIES FOLDER external)
  #set_target_properties(gmock_main PROPERTIES FOLDER external)
endif()

# Global Libraries

find_package(Doxygen)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/config-file.doxygen)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target(doxygen
        #ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

find_package(CUDA REQUIRED)
enable_language(CUDA)

add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(Boost REQUIRED COMPONENTS log log_setup)

find_package(Casacore REQUIRED)

find_package(LAPACK)

find_package(GSL)

# Local Libraries
#add_subdirectory(external/eigen-3.3.90) # builds eigen3 tests
set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/eigen-3.3.90)
find_package(Eigen3 REQUIRED)

set(PC_RapidJSON_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external/rapidjson-1.1.0/include)
find_package(RapidJSON REQUIRED)


# Local Targets
include(lint)
add_subdirectory(src/icrar/leap-accelerate)
add_subdirectory(src/icrar/leap-accelerate-client)
add_subdirectory(src/icrar/leap-accelerate-server)
add_subdirectory(src/icrar/leap-accelerate-cli)

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
