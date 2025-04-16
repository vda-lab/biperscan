add_executable(try_py_bifiltrate EXCLUDE_FROM_ALL try_py_bifiltrate.cpp)
target_link_libraries(try_py_bifiltrate PRIVATE py)

add_executable(try_small_bifiltrate EXCLUDE_FROM_ALL try_small_bifiltrate.cpp)
target_link_libraries(try_small_bifiltrate PRIVATE api)

add_executable(try_horse EXCLUDE_FROM_ALL try_horse.cpp)
target_link_libraries(try_horse PRIVATE api)

add_executable(try_flareable EXCLUDE_FROM_ALL try_flareable.cpp)
target_link_libraries(try_flareable PRIVATE api)

add_executable(try_demo EXCLUDE_FROM_ALL try_demo.cpp)
target_link_libraries(try_demo PRIVATE api)
