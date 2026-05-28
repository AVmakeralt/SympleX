cmake_minimum_required(VERSION 3.20)
project(SympleX
    VERSION 1.0.0
    DESCRIPTION "Polyhedral Tensor Superoptimizer for AI Training Acceleration"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Compiler flags ──────────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wno-unused-parameter -Wno-array-bounds -Wno-stringop-overread
    -march=native -O3
    -ffast-math -fno-math-errno
    -Wno-error
)

# ── Rust polyhedral engine (static library) ────────────────────────────
# Build the Rust engine as a static library and link it into the C++ project.
# The Rust FFI bridge (src/polyhedral/rust_bridge.h) declares the C-linkage
# functions that are implemented in rust-engine/src/ffi.rs.
set(RUST_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust-engine")
set(RUST_STATIC_LIB "${CMAKE_BINARY_DIR}/rust-engine/release/libsymplex_poly.a")

find_program(CARGO_EXECUTABLE cargo REQUIRED)

add_custom_command(
    OUTPUT ${RUST_STATIC_LIB}
    COMMAND ${CARGO_EXECUTABLE} build --release --manifest-path "${RUST_ENGINE_DIR}/Cargo.toml"
            --target-dir "${CMAKE_BINARY_DIR}/rust-engine"
    DEPENDS
        "${RUST_ENGINE_DIR}/src/lib.rs"
        "${RUST_ENGINE_DIR}/src/types.rs"
        "${RUST_ENGINE_DIR}/src/polyhedral.rs"
        "${RUST_ENGINE_DIR}/src/ffi.rs"
        "${RUST_ENGINE_DIR}/Cargo.toml"
    COMMENT "Building SympleX Rust polyhedral engine (staticlib)"
)

add_custom_target(symplex_poly_rust DEPENDS ${RUST_STATIC_LIB})

# ── Find dependencies ───────────────────────────────────────────────────
# ISL (Integer Set Library) for polyhedral operations
find_package(PkgConfig REQUIRED)
pkg_check_modules(ISL IMPORTED_TARGET isl)

# nlohmann_json for configuration serialization
find_package(nlohmann_json 3.9 QUIET)

# Threads
find_package(Threads REQUIRED)

# CUDA (optional – needed for empirical profiling and PTX emission)
include(CMakeDependentOption)
cmake_dependent_option(SYMPLEX_ENABLE_CUDA "Enable CUDA backend" ON "CUDA_FOUND" OFF)
if(SYMPLEX_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# NCCL (optional – needed for distributed sharding)
cmake_dependent_option(SYMPLEX_ENABLE_NCCL "Enable NCCL distributed backend" ON "SYMPLEX_ENABLE_CUDA" OFF)

# LLVM (optional – needed for ORC JIT compilation)
cmake_dependent_option(SYMPLEX_ENABLE_LLVM "Enable LLVM ORC JIT backend" OFF "LLVM_FOUND" OFF)
if(SYMPLEX_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
    target_compile_definitions(symplex PUBLIC SYMPLEX_HAS_LLVM)
endif()

# pybind11 (required for Python bindings)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fallback: use FetchContent to get pybind11
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── Core library ────────────────────────────────────────────────────────
set(SYMPLEX_SOURCES
    src/polyhedral/integer_polytope.cpp
    src/polyhedral/affine_map.cpp
    src/polyhedral/dependency.cpp
    src/polyhedral/iteration_space.cpp
    src/polyhedral/union_map.cpp
    src/polyhedral/polyhedral_optimizer.cpp
    src/schedule/schedule_tree.cpp
    src/schedule/tiling.cpp
    src/schedule/fusion.cpp
    src/schedule/parallelization.cpp
    src/schedule/schedule_map.cpp
    src/hardware/gpu_topology.cpp
    src/hardware/tensor_core.cpp
    src/hardware/memory_hierarchy.cpp
    src/hardware/hardware_target.cpp
    src/optimizer/search_phase1.cpp
    src/optimizer/search_phase2.cpp
    src/optimizer/search_phase3.cpp
    src/optimizer/egraph.cpp
    src/optimizer/superoptimizer.cpp
    src/optimizer/tile_config.cpp
    src/costmodel/roofline.cpp
    src/costmodel/analytical.cpp
    src/costmodel/empirical.cpp
    src/costmodel/cost_model.cpp
    src/codegen/ptx_emitter.cpp
    src/codegen/wmma.cpp
    src/codegen/swizzle.cpp
    src/codegen/register_allocator.cpp
    src/codegen/code_generator.cpp
    src/distributed/mesh.cpp
    src/distributed/sharding.cpp
    src/distributed/nccl_bridge.cpp
    src/distributed/pipeline_overlap.cpp
    src/fault_tolerance/health_monitor.cpp
    src/fault_tolerance/forward_recovery.cpp
    src/fault_tolerance/communicator_repair.cpp
    src/fault_tolerance/checkpoint.cpp
    src/training/dynamic_batch.cpp
    src/training/memory_watchdog.cpp
    src/training/training_loop.cpp
    src/training/compiler_pipeline.cpp
    src/tracer/tracer.cpp
    src/ir/symplex_ir.cpp
    src/lowering/mlir_lowering.cpp
    src/jit/orc_jit.cpp
    src/fusion/fusion_engine.cpp
)

add_library(symplex STATIC ${SYMPLEX_SOURCES})
# Position-independent code required for linking into pybind11 shared module
set_target_properties(symplex PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(symplex PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Link the Rust polyhedral engine ─────────────────────────────────────
add_dependencies(symplex symplex_poly_rust)
target_link_libraries(symplex PUBLIC
    Threads::Threads
    ${RUST_STATIC_LIB}
    dl
    pthread
    m
)

if(ISL_FOUND)
    target_link_libraries(symplex PUBLIC PkgConfig::ISL)
    target_compile_definitions(symplex PUBLIC SYMPLEX_HAS_ISL)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(symplex PUBLIC nlohmann_json::nlohmann_json)
    target_compile_definitions(symplex PUBLIC SYMPLEX_HAS_NLOHMANN_JSON)
endif()

if(SYMPLEX_ENABLE_CUDA)
    target_link_libraries(symplex PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(symplex PUBLIC SYMPLEX_ENABLE_CUDA)
    if(SYMPLEX_ENABLE_NCCL)
        target_compile_definitions(symplex PUBLIC SYMPLEX_ENABLE_NCCL)
    endif()
endif()

# ── Tests ───────────────────────────────────────────────────────────────
enable_testing()

set(SYMPLEX_TEST_SOURCES
    test_polytope
    test_schedule
    test_optimizer
    test_codegen
    test_distributed
    test_fault_tolerance
    test_rust_ffi
)

foreach(test_name IN LISTS SYMPLEX_TEST_SOURCES)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE symplex)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ── Benchmarks ──────────────────────────────────────────────────────────
add_executable(bench_optimizer benchmarks/bench_optimizer.cpp)
target_link_libraries(bench_optimizer PRIVATE symplex)

add_executable(bench_codegen benchmarks/bench_codegen.cpp)
target_link_libraries(bench_codegen PRIVATE symplex)

# ── Examples ────────────────────────────────────────────────────────────
add_executable(example_matmul
    examples/matmul_optimize.cpp
)
target_link_libraries(example_matmul PRIVATE symplex)

install(TARGETS symplex
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/symplex DESTINATION include)

# ── Python Bindings ─────────────────────────────────────────────────────
pybind11_add_module(_symplex
    python/symplex/_tracer.cpp
)
target_include_directories(_symplex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(_symplex PRIVATE symplex)
# The tracer .cpp is already in the symplex static lib, but we need
# to make sure the pybind11 module can find all symbols.
set_target_properties(_symplex PROPERTIES
    PREFIX "")
# Install the Python extension alongside the package
install(TARGETS _symplex
    LIBRARY DESTINATION symplex
)
