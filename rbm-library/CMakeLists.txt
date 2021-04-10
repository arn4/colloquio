
## Disclaimer: it's my first time with CMake.
# I'm overcommenting the file since I want to keep track of the thing I'm doing,
# hoping it will help next time I use CMake.

cmake_minimum_required(VERSION 3.15)
project(Colloquio:RBM_Trainig
  LANGUAGES CXX
)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wshadow -Wformat=2 -pedantic -Wfloat-equal -Wconversion -Wshift-overflow -Wcast-qual -Wcast-align -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fstack-protector -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -funroll-loops")

if (CMAKE_COMPILER_IS_GNUCXX)
  message("Specializing on g++")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wlogical-op -Wduplicated-cond")
endif()

if (CLANG)
  message("Specializing on Clang")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wduplicate-enum -Wno-undefined-var-template")
endif()

# if (NOT APPLE)
#   set(CMAKE_CXX_FLAGS_RELEASE "-static ${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

## Compile the library
add_library(rbm
  ./libsrc/BinaryRBM.cpp
  ./libsrc/TrainingSet.cpp
  ./libsrc/MarcovChain.cpp
  ./libsrc/TrainingAlgorithm.cpp
  ./libsrc/ContrastiveDivergence.cpp
  ./libsrc/PersistentContrastiveDivergence.cpp
)
target_include_directories(rbm PUBLIC #PUBLIC will set this include path also for object
  ./include    # stardard function that can be precompiled
  ./libsrc/tpp # inline and template function that must be treated when an executable is produced
)
target_compile_features(rbm PUBLIC
  cxx_std_17
)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rbm PUBLIC OpenMP::OpenMP_CXX)
endif()



## Compile the tests
add_executable(test-cd test/cd.cpp)
target_link_libraries(test-cd PRIVATE
  rbm
)
target_compile_features(test-cd PUBLIC
  cxx_std_17 # not really needed since the library has already the same PUBLIC feature 
)

add_executable(test-learn-generated test/learn-generated.cpp)
target_link_libraries(test-learn-generated PRIVATE
  rbm
)

add_executable(test-rbm test/rbm.cpp)
target_link_libraries(test-rbm PRIVATE
  rbm
)
