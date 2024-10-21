# Define tests target
add_executable(tests
        EXCLUDE_FROM_ALL
        test_main.cpp
        test_algorithm.cpp
        test_bigrade_ordering.cpp
        test_graded_matrix.cpp
        test_minimal_presentation.cpp
        test_minimal_presentation_merges.cpp
        test_linkage_hierarchy.cpp
)
target_include_directories(tests PRIVATE .)
target_link_libraries(tests PRIVATE api)
