cmake_minimum_required(VERSION 3.22)

project(project.exe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

string(APPEND CMAKE_CXX_FLAGS " -Wall")
string(APPEND CMAKE_CXX_FLAGS " -Wbuiltin-macro-redefined")
string(APPEND CMAKE_CXX_FLAGS " -pedantic")
string(APPEND CMAKE_CXX_FLAGS " -Werror")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/autograd/*.h)
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/autograd/*.cpp)
file(GLOB TESTS ${CMAKE_SOURCE_DIR}/test/*.cpp)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(
    ${PROJECT_NAME}
    main.cpp
    ${SOURCES}
    ${HEADERS}
)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost REQUIRED COMPONENTS container)

target_link_libraries(
    ${PROJECT_NAME}
    Boost::container
)


enable_testing()

add_executable(
    test.exe
    ${TESTS}
)

target_link_libraries(
    test.exe
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test.exe)

add_test(NAME all COMMAND test.exe)
