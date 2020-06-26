cmake_minimum_required(VERSION 3.8)

set(SOLUTION_NAME LEAP-Accelerate)
set(SOLUTION_NAME_LOWER leap-accelerate)

file(STRINGS version.txt SOLUTION_NAME_VERSION)
message(STATUS "Building ${SOLUTION_NAME} version ${SOLUTION_NAME_VERSION}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Project configuration, specifying version, languages,
# and the C++ standard to use for the whole project
project(${SOLUTION_NAME} LANGUAGES CXX CUDA VERSION ${SOLUTION_NAME_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/cmake-modules)

include(dependencies)
include(lint)
include(CTest)

# External projects
if (BUILD_TESTING)
    #add_subdirectory(external/gtest-1.8.1 googletest EXCLUDE_FROM_ALL)
    # installed as git submodule - if this is your first clone you need to
    # git submodule init
    # git submodule update
    # This is a cmake module and needs no further input from you

    #${PROJECT_SOURCE_DIR}
    
	#find_package(GTest)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(external/gtest-1.8.1)
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

find_dependencies()

if(CUDA_FOUND)
	enable_language(CUDA)
endif()

add_subdirectory(src/icrar/leap-accelerate)
add_subdirectory(src/icrar/leap-accelerate-cli)
add_subdirectory(src/icrar/leap-accelerate-client)

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
