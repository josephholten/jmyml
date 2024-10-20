cmake_minimum_required(VERSION 3.20)

project(jmyml VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(WARNING_FLAGS -Wall -Werror -Wpedantic -Wno-unused-function -fsanitize=address)
set(WARNING_LINK_FLAGS -fsanitize=address)
set(PERFORMANCE_FLAGS -Ofast -march=native -flto -fomit-frame-pointer)
set(DEBUG_FLAGS ${WARNING_FLAGS} -O0 -g)
set(RELEASE_FLAGS ${WARNING_FLAGS} ${PERFORMANCE_FLAGS})

add_subdirectory(fmt)
add_subdirectory(googletest)

#add_library(jmyml
    #add further files here
#)

include_directories(.)

enable_testing()

add_executable(
  LinearLayer.test
  jmyml/LinearLayer.test.cpp
)

target_link_libraries(
  LinearLayer.test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(LinearLayer.test)


add_executable(main main.cpp)

# target_link_libraries(main PRIVATE jmyml)
target_link_libraries(main PRIVATE fmt::fmt)
target_compile_options(main PRIVATE
  $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)
target_link_options(main PRIVATE $<$<CONFIG:Debug>:${WARNING_LINK_FLAGS}>)
