## Disclaimer: it's my first time with CMake.
# I'm overcommenting the file since I want to keep track of the thing I'm doing,
# hoping it will help next time I use CMake.

cmake_minimum_required(VERSION 3.15)
project(Colloquio:RBM_Training
  LANGUAGES CXX
)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wfatal-errors -Wshadow -Wformat=2 -pedantic -Wfloat-equal -Wconversion -Wshift-overflow -Wcast-qual -Wcast-align -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fstack-protector -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -funroll-loops")

if (CMAKE_COMPILER_IS_GNUCXX)
  message("Specializing on g++")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wlogical-op -Wduplicated-cond")
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

if (CLANG)
  message("Specializing on Clang")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wduplicate-enum -Wno-undefined-var-template")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -mllvm -inline-treshold=10000")
endif()

## Static Linking does not work also in Ubuntu...
# if (NOT APPLE)
#   set(CMAKE_CXX_FLAGS_RELEASE "-static ${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(rbm-library)
add_subdirectory(simulations)
add_subdirectory(classification)