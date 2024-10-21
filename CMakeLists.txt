cmake_minimum_required(VERSION 3.20)

project(jmyml VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### FLAGS
set(WARNING_FLAGS -Wall -Werror -Wpedantic -Wno-unused-function -fsanitize=address)
set(WARNING_LINK_FLAGS -fsanitize=address)
set(PERFORMANCE_FLAGS -Ofast -march=native -flto -fomit-frame-pointer)
set(DEBUG_FLAGS ${WARNING_FLAGS} -O0 -g)
set(RELEASE_FLAGS ${WARNING_FLAGS} ${PERFORMANCE_FLAGS})

add_library(flags INTERFACE)
target_compile_options(flags INTERFACE
  $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)
target_link_options(flags INTERFACE $<$<CONFIG:Debug>:${WARNING_LINK_FLAGS}>)

### INCLUDE
include_directories(.)

add_subdirectory(fmt)
add_subdirectory(googletest)
add_subdirectory(jmyml)
add_subdirectory(raylib)

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/data")

### EXECUTABLES
add_executable(main main.cpp)
target_link_libraries(main PRIVATE flags fmt::fmt)

### TESTS
enable_testing()
include(GoogleTest)

add_executable(LinearLayer.test jmyml/layers/LinearLayer.test.cpp)
target_link_libraries(LinearLayer.test PRIVATE GTest::gtest_main flags)
gtest_discover_tests(LinearLayer.test)

add_executable(ActivationLayer.test jmyml/layers/ActivationLayer.test.cpp)
target_link_libraries(ActivationLayer.test PRIVATE GTest::gtest_main flags)
gtest_discover_tests(ActivationLayer.test)

add_executable(IdxFile.test jmyml/data/IdxFile.test.cpp)
target_link_libraries(IdxFile.test PRIVATE GTest::gtest_main flags fmt::fmt)
gtest_discover_tests(IdxFile.test)
