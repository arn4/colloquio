
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
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wduplicate-enum")
endif()

if (NOT APPLE)
  set(CMAKE_CXX_FLAGS_RELEASE "-static ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

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

## Compile the tests
add_executable(test_cd test/cd.cpp)
target_link_libraries(test_cd PRIVATE
  rbm
)
target_compile_features(test_cd PUBLIC
  cxx_std_17 # not really needed since the library has already the same PUBLIC feature 
)

add_executable(generate-and-train-cd test/learn-generated.cpp)
target_link_libraries(generate-and-train-cd PRIVATE
  rbm
)

add_executable(rbm-test test/rbm.cpp)
target_link_libraries(rbm-test PRIVATE
  rbm
)

# ## Compile the executables
# add_executable(test test_exec.cpp)
# target_link_libraries(test PRIVATE
#   rbm
# )
# target_compile_features(test PUBLIC
#   cxx_std_17 # not really needed since the library has already the same PUBLIC feature 
# )