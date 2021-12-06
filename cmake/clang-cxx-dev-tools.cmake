# Additional targets to perform clang-format/clang-tidy
# Get all project files
file(GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        *.[ch] *.[chi]pp *.[chi]xx *.cc *.hh *.ii *.[CHI]
        )

find_program(
        CLANG_FORMAT
        NAMES "clang-format"
        DOC "Path to clang-format executable"
)
if (NOT CLANG_FORMAT)
    message(STATUS "clang-format not found.")
else ()
    message(STATUS "clang-format found: ${CLANG_FORMAT}")
    add_custom_target(
            clang-format
            COMMAND ${CLANG_FORMAT}
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )
endif ()
