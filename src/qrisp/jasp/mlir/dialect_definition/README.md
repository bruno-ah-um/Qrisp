# JASP MLIR Dialect Definition

This directory contains the TableGen definitions for the JASP (Qrisp) MLIR dialect.

## Files

- **`JaspDialect.td`** - Dialect and quantum types (QuantumState, Qubit, QubitArray)
- **`JaspOps.td`** - Operation definitions (11 operations)
- **`JaspPythonOps.td`** - Python binding specification

## Generating Python Bindings

### Prerequisites

- **MLIR installation** with `mlir-tblgen` in PATH
- Install MLIR: https://mlir.llvm.org/getting_started/

### Generation Command

From this directory, run:

```bash
mlir-tblgen JaspPythonOps.td \
    -gen-python-op-bindings \
    -bind-dialect=jasp \
    -I /path/to/llvm-project/mlir/include \
    -I . \
    -o ../dialect_implementation/_jasp_ops_gen.py
```

Replace `/path/to/llvm-project/mlir/include` with your MLIR include directory.

## Building the Dialect Plugin (libJaspDialect.so)

Building the shared library enables full dialect-aware validation with `mlir-opt`,
going beyond syntax-only checks.

### Prerequisites

- **MLIR and LLVM dev packages** (both are required — MLIR's CMake config depends on LLVM's)
- `cmake` and `ninja` (or `make`)

On Debian/Ubuntu, install them with:

```bash
sudo apt install llvm-19-dev libmlir-19-dev mlir-19-tools
```

Adjust the version number to match your distribution.

### Build

From the repository root:

```bash
cmake -S src/qrisp/jasp/mlir/dialect_definition \
      -B build/jasp_dialect \
      -DMLIR_DIR=/usr/lib/llvm-19/lib/cmake/mlir \
      -DLLVM_DIR=/usr/lib/llvm-19/lib/cmake/llvm
cmake --build build/jasp_dialect
```

Replace `llvm-19` with the version you installed (e.g. `llvm-18`, `llvm-20`).

### Running the full-validation test

Once built, point the test at the library via the `JASP_DIALECT_LIB`
environment variable:

```bash
export JASP_DIALECT_LIB=$(pwd)/build/jasp_dialect/libJaspDialect.so
pytest tests/jax_tests/test_mlir.py::test_mlir_opt_roundtrip -v
```

Without `JASP_DIALECT_LIB` set, the test falls back to syntax-only validation
using `--allow-unregistered-dialect` and still passes.