cmake_minimum_required(VERSION 3.20)

project(fpstudy
        VERSION 0.1.0
        DESCRIPTION "Floating point precision evaluation suite"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FPSTUDY_BUILD_TESTS "Build unit tests" ON)

include(FetchContent)

# Stillwater Universal library
FetchContent_Declare(
    universal
    GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
    GIT_TAG main
)
FetchContent_GetProperties(universal)
if(NOT universal_POPULATED)
    FetchContent_Populate(universal)
endif()

set(UNIVERSAL_EXTERN_DIR ${universal_SOURCE_DIR} CACHE PATH "" FORCE)

if (TARGET universal::universal)
    set(UNIVERSAL_TARGET universal::universal)
elseif (TARGET universal)
    set(UNIVERSAL_TARGET universal)
else()
    add_library(universal INTERFACE)
    target_include_directories(universal INTERFACE
        ${UNIVERSAL_EXTERN_DIR}/include
        ${UNIVERSAL_EXTERN_DIR}/include/sw
    )
    set(UNIVERSAL_TARGET universal)
endif()

add_library(fpstudy_formats
    src/formats/precision.cpp
    src/core/io.cpp
)

target_include_directories(fpstudy_formats
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fpstudy_formats
    PUBLIC
        ${UNIVERSAL_TARGET}
)

add_library(fpstudy_algorithms INTERFACE)
target_include_directories(fpstudy_algorithms INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(fpstudy src/main.cpp)
target_link_libraries(fpstudy PRIVATE fpstudy_formats fpstudy_algorithms ${UNIVERSAL_TARGET})

if(FPSTUDY_BUILD_TESTS)
    enable_testing()
    add_executable(fpstudy_tests
        tests/matmul_tests.cpp
        tests/iterative_tests.cpp
        tests/io_tests.cpp
        tests/fir_tests.cpp
    )
    target_link_libraries(fpstudy_tests PRIVATE fpstudy_formats fpstudy_algorithms ${UNIVERSAL_TARGET})
    add_test(NAME MatMulTests COMMAND fpstudy_tests MatMul)
    add_test(NAME IterativeTests COMMAND fpstudy_tests Iterative)
    add_test(NAME IOTests COMMAND fpstudy_tests IO)
    add_test(NAME FIRTests COMMAND fpstudy_tests FIR)
endif()

