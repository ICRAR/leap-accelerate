if(NOT TARGET lint)
    find_program(RUN_CLANG_TIDY
        NAMES 
            run-clang-tidy
        PATHS
            /usr/local/opt/llvm/bin
    )

    if (NOT RUN_CLANG_TIDY)
        message(WARNING "run-clang-tidy not found - no lint target")
    else()
        message(STATUS "run-clang-tidy found adding lint target")
        
        file(GLOB_RECURSE ALL_SOURCE_FILES
            ${PROJECT_SOURCE_DIR}/src/*.cc
            ${PROJECT_SOURCE_DIR}/src/*.h
        )

        add_custom_target(lint
            COMMAND ${RUN_CLANG_TIDY} -j 4 -quiet
            ${ALL_SOURCE_FILES}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            ${INCLUDE_DIRECTORIES}
        )
    endif()
endif()

