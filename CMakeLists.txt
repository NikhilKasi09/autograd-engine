cmake_minimum_required(VERSION 3.20)
project(autograd_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Writes build/compile_commands.json, which is how an editor's language server
# learns the real flags. Without -mavx2 it reports every _mm256_* intrinsic as
# undefined, and without -std=c++20 it rejects the class.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release. An -O0 build's benchmark numbers are meaningless, and
# forgetting -DCMAKE_BUILD_TYPE is the easiest way to produce one by accident.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Matches the C Makefile exactly. -march=native is deliberately absent: the
# baseline in ~/gemm-c-baseline.txt was built without it, so adding it here
# would make the port and the tuning indistinguishable in the numbers.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# On now that the sources compile clean as C++. It was off for the mechanical
# port so the whole warning list could be read and triaged in one pass rather
# than stopping at the first one.
option(GEMM_WERROR "Treat warnings as errors" ON)

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

# Every tree used to write the module to one shared path, so a sanitizer build
# clobbered the Release module with an instrumented one that cannot be imported
# ("ASan runtime does not come first in initial library list") - which looks
# like a broken binding rather than a clobbered file. The old fix was to build
# the module only in unsanitized trees; this is the better one, and it lets the
# sanitizer and Debug trees have a module at all.
#
# Only the canonical Release tree writes next to the sources, so a bare
# `pytest tests/python` from the repo root still works with no PYTHONPATH.
set(GEMM_PYTHON_ROOT "" CACHE PATH "Directory placed on sys.path; holds the autograd package")
if(NOT GEMM_PYTHON_ROOT)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" AND GEMM_SANITIZE STREQUAL "off")
        set(GEMM_PYTHON_ROOT "${CMAKE_SOURCE_DIR}/python")
    else()
        set(GEMM_PYTHON_ROOT "${CMAKE_BINARY_DIR}/python")
    endif()
endif()
set(GEMM_PYTHON_MODULE_DIR "${GEMM_PYTHON_ROOT}/autograd")

pybind11_add_module(_core bindings/bindings.cpp)
gemm_apply_flags(_core)

# The bindings call the kernels, so they need the library and, through its
# PUBLIC include directory, the headers.
target_link_libraries(_core PRIVATE gemm_core)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GEMM_PYTHON_MODULE_DIR}")

# Every tree gets its own copy of the pure-Python package beside its own _core,
# so each tree is a complete importable package and PYTHONPATH alone decides
# which one is used. Without this the source-tree package would be the only one
# that imports, and pointing PYTHONPATH at a build tree would find a _core with
# no package around it - or worse, fall back to the stale Release module.
#
# CONFIGURE_DEPENDS re-runs the glob at build time, so a new module added to the
# package reaches the other trees without a manual reconfigure. Phase 5 turned
# this from a single hardcoded __init__.py into a glob for exactly that reason.
file(GLOB GEMM_PYTHON_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/python/autograd/*.py")

# Globbing *.py rather than copying the directory is deliberate: the source
# package also holds the Release build's _core.so, and copying that into another
# tree is the two-trees-one-module trap the per-tree layout exists to prevent.
#
# A separate ALL target, NOT a POST_BUILD on _core. POST_BUILD runs only when
# _core itself relinks, so editing a .py and rebuilding left the tree holding
# the previous copy - stale, silent, and looking exactly like broken code when
# the suite was run against it. This target's command runs every build;
# copy_if_different compares before writing, so the cost is four stats.
if(NOT GEMM_PYTHON_MODULE_DIR STREQUAL "${CMAKE_SOURCE_DIR}/python/autograd")
    add_custom_target(autograd_package ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GEMM_PYTHON_MODULE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GEMM_PYTHON_SOURCES}
                "${GEMM_PYTHON_MODULE_DIR}/"
        COMMENT "Copying the autograd package next to _core")
    add_dependencies(autograd_package _core)
endif()

# ---------------------------------------------------------------------------
# GEMM library
# ---------------------------------------------------------------------------

# Listed explicitly rather than globbed, so adding a kernel is a deliberate
# edit here and the ladder stays visible in one place.
add_library(gemm_core STATIC
    src/benchmark.cpp
    src/gemm_common.cpp
    src/gemm_ikj.cpp
    src/gemm_naive.cpp
    src/gemm_threaded.cpp
    src/gemm_simd.cpp
    src/gemm_tiled.cpp
    src/gemm_tiled_simd.cpp
    src/tensor.cpp
    src/tensor_ops.cpp
    src/validate.cpp
)
target_include_directories(gemm_core PUBLIC include)
gemm_apply_flags(gemm_core)

# _core is a shared module, and a shared object cannot link a static archive
# built without position-independent code.
#
# Measured rather than assumed: an interleaved A/B of five runs each, PIC
# against not, overlaps on all 36 shape/kernel pairs. The kernels take
# everything through registers and touch globals only on error paths, so the
# GOT indirection never lands in a hot loop.
set_target_properties(gemm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Replaces the Makefile's -pthread. Threads::Threads carries the flag to every
# consumer, so the linker gets it too.
find_package(Threads REQUIRED)
target_link_libraries(gemm_core PUBLIC Threads::Threads)

add_executable(gemm_benchmark apps/benchmark_main.cpp)
target_link_libraries(gemm_benchmark PRIVATE gemm_core)
gemm_apply_flags(gemm_benchmark)

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

# One Catch2 binary for everything. Tags select: "[gemm]" for the shape table,
# "[tensor]" for the Tensor units, "[view]" for the view operations, "[ops]"
# for the elementwise layer, and "[.fuzz]" for the randomised run, which is
# hidden by default because it costs a second.
add_executable(gemm_tests
    tests/cpp/test_gemm.cpp
    tests/cpp/test_tensor.cpp
    tests/cpp/test_tensor_ops.cpp
)
target_link_libraries(gemm_tests PRIVATE gemm_core Catch2::Catch2WithMain)
gemm_apply_flags(gemm_tests)

# TSan only supports vm.mmap_rnd_bits up to 28 and Ubuntu 24.04 defaults to 32,
# so without ASLR off the binary dies intermittently with "unexpected memory
# mapping". CTest has no RUNNER equivalent, so the wrapper goes on the command.
if(GEMM_SANITIZE STREQUAL "thread")
    add_test(NAME gemm_tests COMMAND setarch ${CMAKE_SYSTEM_PROCESSOR} -R $<TARGET_FILE:gemm_tests>)
else()
    add_test(NAME gemm_tests COMMAND $<TARGET_FILE:gemm_tests>)
endif()

# The Python suite is part of the test surface from phase 4 on, so `ctest` is
# the one command. Note ctest does not build: run `cmake --build build` first.
#
# Registered only for unsanitized trees. A sanitized _core still gets *built*
# above, but importing one into a stock CPython needs LD_PRELOAD plus
# ASAN_OPTIONS plus PYTHONMALLOC, which is a documented manual one-off rather
# than something ctest should attempt.
#
# GEMM_EXPECTED_MODULE_DIR turns a wrong-tree import into a hard failure in
# conftest.py rather than a quietly wrong set of numbers.
#
# GEMM_REQUIRE_TORCH decides what a missing torch means for the cross-check:
#   AUTO (default) - probe now, require it later only if it is here now. Catches
#                    a torch that breaks after configure; does NOT catch one that
#                    was never installed, which configures clean and skips.
#   ON             - configure fails without it. For a tree where the
#                    cross-check must actually run.
#   OFF            - never required.
set(GEMM_REQUIRE_TORCH "AUTO" CACHE STRING "Require torch for the gradient cross-check")
set_property(CACHE GEMM_REQUIRE_TORCH PROPERTY STRINGS AUTO ON OFF)

if(GEMM_SANITIZE STREQUAL "off")
    set(GEMM_TORCH_ENV "")
    if(NOT GEMM_REQUIRE_TORCH STREQUAL "OFF")
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import torch"
                        RESULT_VARIABLE GEMM_TORCH_MISSING
                        OUTPUT_QUIET ERROR_QUIET)
        if(GEMM_TORCH_MISSING EQUAL 0)
            set(GEMM_TORCH_ENV ";GEMM_REQUIRE_TORCH=1")
            message(STATUS "torch found: the gradient cross-check is required")
        elseif(GEMM_REQUIRE_TORCH STREQUAL "ON")
            message(FATAL_ERROR
                    "GEMM_REQUIRE_TORCH=ON but torch will not import. Install it:\n"
                    "  ${Python3_EXECUTABLE} -m pip install torch "
                    "--index-url https://download.pytorch.org/whl/cpu")
        else()
            message(STATUS "torch not found: the gradient cross-check will SKIP")
        endif()
    endif()

    add_test(NAME python_bindings
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_SOURCE_DIR}/tests/python" -q)
    set_tests_properties(python_bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=${GEMM_PYTHON_ROOT};GEMM_EXPECTED_MODULE_DIR=${GEMM_PYTHON_MODULE_DIR}${GEMM_TORCH_ENV}")
endif()
