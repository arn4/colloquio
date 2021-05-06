## Compile the library
add_library(rbm
  src/BernoulliRBM.cpp
  src/TrainingSet.cpp
  src/MarkovChain.cpp
  src/TrainingAlgorithm.cpp
  src/ContrastiveDivergence.cpp
  src/PersistentContrastiveDivergence.cpp
  src/ExtendedMeanField.cpp
)
target_include_directories(rbm PUBLIC #PUBLIC will set this include path also for code that rely on this library
  include/   # stardard functions that can be precompiled
  src/tpp # inline and template functions that must be compiled when an executable is produced
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
set_target_properties(test-cd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/rbm-library-tests")

add_executable(test-learn-generated test/learn-generated.cpp)
target_link_libraries(test-learn-generated PRIVATE rbm)
target_compile_features(test-learn-generated  PUBLIC cxx_std_17)
set_target_properties(test-learn-generated PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/rbm-library-tests")

add_executable(test-rbm test/rbm.cpp)
target_link_libraries(test-rbm PRIVATE rbm)
target_compile_features(test-rbm PUBLIC cxx_std_17)
set_target_properties(test-rbm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/rbm-library-tests")
