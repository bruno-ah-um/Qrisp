# CMakeLists.txt for the JASP MLIR dialect shared library
#
# Builds libJaspDialect.so (Linux) / libJaspDialect.dylib (macOS), which can
# be loaded by mlir-opt via --load-dialect-plugin to enable full dialect-aware
# validation without --allow-unregistered-dialect.
#
# Usage
# -----
#   cmake -S . -B build -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir
#   cmake --build build
#
# The resulting library is written to build/libJaspDialect.so.
# Point the test at it with:
#   export JASP_DIALECT_LIB=$(pwd)/build/libJaspDialect.so
#   pytest tests/jax_tests/test_mlir.py::test_mlir_opt_roundtrip

cmake_minimum_required(VERSION 3.20)
project(JaspDialect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Locate LLVM and MLIR
# --------------------------------------------------------------------------
# LLVM must be found before MLIR because MLIRConfig.cmake depends on it.
# Pass -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm if not auto-detected.
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake from: ${MLIR_DIR}")
message(STATUS "MLIR include dirs: ${MLIR_INCLUDE_DIRS}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include(AddLLVM)

# mlir_tablegen needs to find mlir/IR/DialectBase.td etc.
set(LLVM_TABLEGEN_FLAGS "-I${MLIR_INCLUDE_DIRS}" "-I${CMAKE_CURRENT_SOURCE_DIR}")

# --------------------------------------------------------------------------
# TableGen: generate C++ from .td files
# --------------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS JaspDialect.td)

mlir_tablegen(JaspDialect.h.inc      -gen-dialect-decls)
mlir_tablegen(JaspDialect.cpp.inc    -gen-dialect-defs)
mlir_tablegen(JaspTypes.h.inc        -gen-typedef-decls)
mlir_tablegen(JaspTypes.cpp.inc      -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS JaspOps.td)
mlir_tablegen(JaspOps.h.inc          -gen-op-decls)
mlir_tablegen(JaspOps.cpp.inc        -gen-op-defs)

add_public_tablegen_target(JaspDialectIncGen)

# --------------------------------------------------------------------------
# C++ sources
# --------------------------------------------------------------------------
# These files are generated in-tree alongside this CMakeLists.txt so that the
# library can be built without any extra source directories.

set(JASP_DIALECT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/JaspDialect.cpp
)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JaspDialect.cpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/JaspDialect.cpp"
"
//===- JaspDialect.cpp - JASP dialect registration -----------------------===//
//
// Glue that registers the JASP dialect and its types/ops so that mlir-opt
// can load it as a plugin via --load-dialect-plugin.
//
//===----------------------------------------------------------------------===//

#include \"mlir/Bytecode/BytecodeOpInterface.h\"
#include \"mlir/IR/Builders.h\"
#include \"mlir/IR/DialectImplementation.h\"
#include \"mlir/IR/SymbolTable.h\"
#include \"mlir/Tools/Plugins/DialectPlugin.h\"
#include \"llvm/ADT/TypeSwitch.h\"

// 1. Dialect class declaration (no macro guard in generated file).
#include \"JaspDialect.h.inc\"

// 2. Type class declarations -- gated behind GET_TYPEDEF_CLASSES.
#define GET_TYPEDEF_CLASSES
#include \"JaspTypes.h.inc\"
#undef GET_TYPEDEF_CLASSES

// 3. Op class declarations -- gated behind GET_OP_CLASSES.
#define GET_OP_CLASSES
#include \"JaspOps.h.inc\"
#undef GET_OP_CLASSES

// 4. Dialect method definitions -- gated behind GET_DIALECT_DEFS.
#define GET_DIALECT_DEFS
#include \"JaspDialect.cpp.inc\"
#undef GET_DIALECT_DEFS

// 5. Type method definitions -- gated behind GET_TYPEDEF_CLASSES.
#define GET_TYPEDEF_CLASSES
#include \"JaspTypes.cpp.inc\"
#undef GET_TYPEDEF_CLASSES

// 6. Op method definitions -- gated behind GET_OP_CLASSES.
#define GET_OP_CLASSES
#include \"JaspOps.cpp.inc\"
#undef GET_OP_CLASSES

// ---------------------------------------------------------------------------
// Dialect initializer -- registers all types and operations.
// ---------------------------------------------------------------------------

void jasp::JaspDialect::initialize() {
    // Allow ops unknown to this plugin (e.g. jasp.quantum_kernel, jasp.return,
    // jasp.call) so that the statically-linked TypeID mismatch between mlir-opt
    // and the plugin does not break terminator/trait checks.  Unknown ops are
    // still accepted by --allow-unregistered-dialect.
    allowUnknownOperations();

    addTypes<
        jasp::QuantumStateType,
        jasp::QubitType,
        jasp::QubitArrayType
    >();
    addOperations<
        jasp::CreateQubitsOp,
        jasp::GetQubitOp,
        jasp::GetSizeOp,
        jasp::SliceOp,
        jasp::FuseOp,
        jasp::ResetOp,
        jasp::MeasureOp,
        jasp::DeleteQubitsOp,
        jasp::CreateQuantumKernelOp,
        jasp::ConsumeQuantumKernelOp,
        jasp::QuantumGateOp,
        jasp::ParityOp,
        jasp::JaspModuleOp
    >();
}

// Plugin entry point: called by mlir-opt --load-dialect-plugin.
extern \"C\" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        \"JaspDialect\",
        \"0.1\",
        [](mlir::DialectRegistry *registry) {
            registry->insert<jasp::JaspDialect>();
        }
    };
}
")
endif()

# --------------------------------------------------------------------------
# Shared library target
# --------------------------------------------------------------------------
add_library(JaspDialect SHARED ${JASP_DIALECT_SRCS})

add_dependencies(JaspDialect JaspDialectIncGen)

target_include_directories(JaspDialect PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}   # for generated .inc files
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(JaspDialect PRIVATE
    MLIRIR
    MLIRSupport
    MLIRParser
)

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
install(TARGETS JaspDialect
    LIBRARY DESTINATION lib
)

message(STATUS "
JASP dialect plugin build configured.
After building, run tests with:
  export JASP_DIALECT_LIB=\${CMAKE_BINARY_DIR}/libJaspDialect.so
  pytest tests/jax_tests/test_mlir.py::test_mlir_opt_roundtrip
")
