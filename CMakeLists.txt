cmake_minimum_required(VERSION 3.20)

project(jmyml VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(JMYML_BUILD_IDXVIEW "build idxview" OFF)

### FLAGS
set(WARNING_FLAGS -Wall -Werror -Wpedantic -Wno-unused-function)
set(DEBUG_FLAGS ${WARNING_FLAGS} -O0 -g)
set(RELEASE_FLAGS ${WARNING_FLAGS} -O3)

add_library(flags INTERFACE)
target_compile_options(flags INTERFACE
  $<$<CONFIG:Debug>:${DEBUG_FLAGS}>
  $<$<CONFIG:Release>:${RELEASE_FLAGS}>
)

### INCLUDE
include_directories(.)

add_subdirectory(fmt)
add_subdirectory(googletest)

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/data")

### EXECUTABLES
add_executable(scratch scratch.cpp)
target_link_libraries(scratch PRIVATE flags)
target_link_libraries(scratch PRIVATE fmt::fmt)

if(JMYML_BUILD_IDXVIEW)
  add_subdirectory(raylib)

  add_executable(idxview idxview.cpp)
  target_link_libraries(idxview PRIVATE flags)
  target_link_libraries(idxview PRIVATE fmt::fmt)
  target_link_libraries(idxview PRIVATE raylib)
endif()

### TESTS
enable_testing()
include(GoogleTest)

set(tests
  jmyml/layers/LinearLayer.test.cpp
  jmyml/layers/ActivationLayer.test.cpp
  jmyml/data/IdxFile.test.cpp
  jmyml/layers/Sequential.test.cpp
)
foreach(testf ${tests})
  cmake_path(GET testf STEM LAST_ONLY test)
  add_executable(${test} ${testf})
  target_link_libraries(${test} PRIVATE GTest::gtest_main)
  target_link_libraries(${test} PRIVATE fmt::fmt)
  target_link_libraries(${test} PRIVATE flags)
  gtest_discover_tests(${test})
endforeach(testf ${testfiles})
