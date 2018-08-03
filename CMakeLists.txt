cmake_minimum_required(VERSION 3.8)
project(MLP)

set(CMAKE_C_STANDARD 11)

set(SOURCE_FILES main.c main2.c)
add_executable(MLP ${SOURCE_FILES})