# Setup cmake
cmake_minimum_required(VERSION 3.12)
project(biperscan CXX)

# Setup c++
if (MSVC)
    add_compile_options(/std:c++latest)
else ()
    set(CMAKE_CXX_STANDARD 23)
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0076 NEW)

# Set GCC release optimization to 02
string(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# Configure options
set(BUILD_TESTS ON CACHE BOOL "Enable building tests.")
set(BUILD_TRIALS ON CACHE BOOL "Enable building trial scripts.")

# Add sources
add_library(api STATIC
        biperscan.h
        lib/concepts.h
        lib/algorithm.h
        lib/base_types.h
        lib/bigrade_ordering.h
        lib/graded_matrix.h
        lib/minimal_presentation.h
        lib/minimal_presentation_merges.h
        lib/minimal_presentation_graph.h
        lib/minimal_presentation_iteration_state.h
        lib/linkage_hierarchy.h
        lib/ostream.h
        lib/ostream.cpp
)
target_include_directories(api INTERFACE .)

add_library(py INTERFACE py_biperscan.h)
target_link_libraries(py INTERFACE api)
set_target_properties(py PROPERTIES LINKER_LANGUAGE CXX)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()

if (BUILD_TRIALS)
    add_subdirectory(trial_scripts)
endif ()