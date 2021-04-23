
## Compile MNIST
add_executable(mnist-learning mnist.cpp)
target_link_libraries(mnist-learning PRIVATE rbm)
target_compile_features(mnist-learning PRIVATE cxx_std_17)

## Compile MNIST generator
add_executable(mnist-generator mnist-generator.cpp)
target_link_libraries(mnist-generator PRIVATE rbm)
target_compile_features(mnist-generator PRIVATE cxx_std_17)

## Compile MNIST stability
add_executable(mnist-stability mnist-stability-test.cpp)
target_link_libraries(mnist-stability PRIVATE rbm)
target_compile_features(mnist-stability PRIVATE cxx_std_17)
