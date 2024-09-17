# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/
find_package(Doxygen)
find_package(Sphinx)
if(Sphinx_FOUND AND Doxygen_FOUND)

  # Find all the public headers
  get_target_property(PUBLIC_HEADER_DIR LeapAccelerate INTERFACE_INCLUDE_DIRECTORIES)
  file(GLOB_RECURSE PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/src/*.h")

  set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
  set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  #This will be the main output of our command
  set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

  #Replace variables inside @@ with the current values
  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us

  add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                    DEPENDS ${PUBLIC_HEADERS}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                    COMMENT "Generating doxygen")

  # Nice named target so we can run the job easily
  add_custom_target(doxygen DEPENDS ${DOXYGEN_INDEX_FILE})

  set(SPHINX_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
  set(SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
  set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

  file(GLOB_RECURSE DOC_FILES "*.rst")

  if(WERROR)
    set(WARGS -W --keep-going)
  endif()

  # Only regenerate Sphinx when:
  # - Doxygen has rerun
  # - Our doc files have been updated
  # - The Sphinx config has been updated
  add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                    COMMAND 
                    ${SPHINX_EXECUTABLE} -b html -T
                    # Tell Breathe where to find the Doxygen output
                    -Dbreathe_projects.LeapAccelerate=${DOXYGEN_OUTPUT_DIR}/xml
                    # -Ddoxygen_xml=${DOXYGEN_OUTPUT_DIR}/xml
                    ${WARGS}
                    ${SPHINX_SOURCE_DIR} ${SPHINX_BUILD_DIR}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    DEPENDS
                    # Other docs files you want to track should go here (or in some variable)
                    ${DOC_FILES}
                    ${DOXYGEN_INDEX_FILE}
                    MAIN_DEPENDENCY ${SPHINX_SOURCE_DIR}/conf.py
                    COMMENT "Generating documentation with sphinx")

  # TODO: clean needs to perform:
  # rm -rf doxygen &&
  # rm -rf sphinx &&
  # rm -rf /icrar-leap-accelerate/docs/src/api
  # TODO: Sphinx needs Doxyfile.in touched to rebuild

  # Nice named target so we can run the job easily
  add_custom_target(sphinx DEPENDS ${SPHINX_INDEX_FILE})
  add_dependencies(sphinx doxygen)

  # Add an install target to install the docs
  include(GNUInstallDirs)
  install(DIRECTORY ${SPHINX_BUILD_DIR}
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
else()
  message(STATUS "Sphinx or Doxygen not found - skipping docs targets")
endif()