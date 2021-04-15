
## Compile MNIST hidden calculator
add_executable(mnist-hidden mnist-hidden-probabilities.cpp)
target_link_libraries(mnist-hidden PRIVATE rbm)
target_compile_features(mnist-hidden PRIVATE cxx_std_17)
