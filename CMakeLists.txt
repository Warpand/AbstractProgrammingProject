cmake_minimum_required(VERSION 3.22)

project(project.exe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

string(APPEND CMAKE_CXX_FLAGS " -Wall")
string(APPEND CMAKE_CXX_FLAGS " -Wbuiltin-macro-redefined")
string(APPEND CMAKE_CXX_FLAGS " -pedantic")
string(APPEND CMAKE_CXX_FLAGS " -Werror")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/autograd/*.h)
file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/autograd/*.cpp")

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
