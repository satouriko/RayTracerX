cmake_minimum_required(VERSION 3.19)
project(RayTracerX)

set(CMAKE_CXX_STANDARD 17)
include(cmake/clang-cxx-dev-tools.cmake)

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

add_executable(RayTracerX main.cpp)
target_link_libraries(RayTracerX OpenGL::GL glfw GLEW::GLEW)
