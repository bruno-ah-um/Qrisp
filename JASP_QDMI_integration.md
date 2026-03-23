# JASP–QDMI Integration: Compiling Quantum Programs to Native C

## Overview

This document describes an experimental prototype that explores compiling JASP
programs to standalone C source code, with quantum kernel execution routed
through the Quantum Device Management Interface (QDMI). The prototype lowers a
high-level Python description through MLIR and the EmitC dialect, producing a
single C file that can be built with Clang and linked against a QDMI-compatible
backend. The goal is to open a discussion on how Qrisp's Python-level quantum
programming model could connect to ahead-of-time compiled execution on real
hardware, and to identify the design trade-offs involved.

## Compilation Pipeline

The prototype follows a multi-stage pipeline: JASP → MLIR → EmitC → C. Three
dedicated xDSL passes prepare the MLIR module for C emission. First,
`strip_quantum_state_from_main` removes the `!jasp.QuantumState` threading
artifact from the `@main` function, turning it into a purely classical entry
point. Second, `lower_jasp_call_to_qdmi` serialises each quantum kernel as an
MLIR string constant and replaces the `jasp.call` invocations with
`emitc.call_opaque "run_jasp_kernel"` operations that pass the serialised
kernel text to the runtime. Third, `lower_classical_to_emitc` rewrites all
remaining classical operations (StableHLO constants, arithmetic, casts) into
their EmitC equivalents. The resulting module is then translated to C++ via
`mlir-translate --mlir-to-cpp` and post-processed to inject the runtime
lifecycle calls.

## C Runtime and QDMI Backend

A minimal, pure-C runtime (`runtime.h` / `runtime.c`) serves as the stable ABI
boundary between the compiler-generated code and the quantum hardware. The
runtime initialises a QDMI session, discovers the first available device, and
exposes a `run_jasp_kernel` function that creates a QDMI job, submits the
serialised MLIR kernel, waits for completion, and returns the measurement result
as an integer. All QDMI types and handles are confined to the runtime's
translation unit — nothing leaks into the generated code. Initialization and
cleanup are idempotent, and configuration (shot count, authentication token,
device selection) can be adjusted through an optional `RuntimeConfig` struct.
The entire runtime is inlined into the emitted C file, so the output is a
self-contained compilation unit that only requires the QDMI headers and driver
library at build time.

## User-Facing API

In this prototype, the pipeline is exposed through a single method:
`Jaspr.to_cpp()`. Given any traced `Jaspr` object, calling `to_cpp()` runs the
full pipeline and returns the C source as a string. This mirrors the existing
`Jaspr.to_mlir()` method and fits naturally into the Qrisp workflow — users can
inspect, save, or compile the output with a standard `clang` invocation. The
design is intentionally minimal and extensible: as QDMI gains support for native
MLIR or QIR program formats the runtime could be updated without changing the
compiler passes or the public API.

## Open Questions

1. **In-tree or external tool?** Should this MLIR-to-C pipeline live inside
   Qrisp (implemented in Python via xDSL), or would it be better maintained as
   a standalone tool outside the Qrisp repository? Keeping it in-tree makes it
   easy to evolve alongside the JASP dialect, but an external tool could have
   its own release cadence and avoid adding QDMI/EmitC dependencies to the core
   framework.

2. **Relationship with the new Qrisp backend interface.** Qrisp is developing a
   new backend interface for hardware execution. How should this QDMI-based
   compilation path relate to that effort? Have both features different use cases?
   
