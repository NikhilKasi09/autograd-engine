cmake_minimum_required(VERSION 3.20)
project(autograd_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release. An -O0 build's benchmark numbers are meaningless, and
# forgetting -DCMAKE_BUILD_TYPE is the easiest way to produce one by accident.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Matches the C Makefile exactly. -march=native is deliberately absent: the
# baseline in ~/gemm-c-baseline.txt was built without it, so adding it here
# would make the port and the tuning indistinguishable in the numbers.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Off during the mechanical C->C++ port so the whole warning list can be read
# and triaged at once, rather than stopping at the first one. Flipped on as the
# final commit of that step.
option(GEMM_WERROR "Treat warnings as errors" OFF)

set(GEMM_SANITIZE "off" CACHE STRING "Sanitizer: off, address, or thread")
set_property(CACHE GEMM_SANITIZE PROPERTY STRINGS off address thread)

# Applied to every target we build. -mavx2/-mfma are here rather than in the
# Release flags because the sanitizer builds compile the same intrinsics and
# need them just as much.
function(gemm_apply_flags target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -pedantic -mavx2 -mfma)

    if(GEMM_WERROR)
        target_compile_options(${target} PRIVATE -Werror)
    endif()

    # -O1 keeps naive down to a couple of seconds under instrumentation, and
    # -fno-sanitize-recover makes UBSan fail rather than print and carry on.
    # The sanitizers are needed at link time as well as compile time.
    if(NOT GEMM_SANITIZE STREQUAL "off")
        target_compile_options(${target} PRIVATE -g -O1 -fno-omit-frame-pointer -fno-sanitize-recover=all)
    endif()

    if(GEMM_SANITIZE STREQUAL "address")
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    elseif(GEMM_SANITIZE STREQUAL "thread")
        target_compile_options(${target} PRIVATE -fsanitize=thread)
        target_link_options(${target} PRIVATE -fsanitize=thread)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Python / pybind11
# ---------------------------------------------------------------------------

# Prefer the project venv unless the caller names an interpreter explicitly,
# so a bare `cmake -B build` picks up the pybind11 installed there.
if(NOT DEFINED Python3_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 ships its own CMake config inside the wheel; ask the interpreter
# where rather than making the caller set CMAKE_PREFIX_PATH by hand.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_QUERY_RESULT
)
if(NOT pybind11_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not locate pybind11 via ${Python3_EXECUTABLE}. Is the venv installed?")
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core bindings/bindings.cpp)
gemm_apply_flags(_core)

# Drop the module next to the Python sources so pytest imports it without an
# install step or a PYTHONPATH dance.
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python")

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

include(CTest)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

# Placeholder until the shape harness is ported. It exists so that the whole
# chain -- fetch Catch2, compile against it, register with ctest, run under the
# setarch wrapper -- is proven now rather than at the step that depends on it.
add_executable(gemm_tests tests/cpp/test_smoke.cpp)
target_link_libraries(gemm_tests PRIVATE Catch2::Catch2WithMain)
gemm_apply_flags(gemm_tests)

# TSan only supports vm.mmap_rnd_bits up to 28 and Ubuntu 24.04 defaults to 32,
# so without ASLR off the binary dies intermittently with "unexpected memory
# mapping". CTest has no RUNNER equivalent, so the wrapper goes on the command.
if(GEMM_SANITIZE STREQUAL "thread")
    add_test(NAME gemm_tests COMMAND setarch ${CMAKE_SYSTEM_PROCESSOR} -R $<TARGET_FILE:gemm_tests>)
else()
    add_test(NAME gemm_tests COMMAND $<TARGET_FILE:gemm_tests>)
endif()
