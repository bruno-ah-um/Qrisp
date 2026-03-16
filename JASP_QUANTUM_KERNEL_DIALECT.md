# JASP Dialect: `jasp.quantum_kernel`

Design notes and implementation reference for the `jasp.quantum_kernel` op.

---

## Motivation

Before this change, quantum functions were identified by sentinel ops:

```mlir
func.func @main(...) {
  %qst = jasp.create_quantum_kernel -> !jasp.QuantumState
  %res, %qst_out = func.call @my_func(%arg, %qst) : ...
  %_ = jasp.consume_quantum_kernel %qst_out : ...
}
```

To know whether a `func.func` is quantum, a tool had to scan the body for these
sentinels. `jasp.quantum_kernel` makes quantum functions unambiguously
identifiable from the op type alone — analogous to how `gpu.func` distinguishes
GPU kernels from regular functions.

---

## IR Shape

After the full pipeline (`lift_quantum_kernels` + `hoist_classical_ops`):

```mlir
// Classical host code — lives directly in builtin.module.
// Post-measurement arithmetic has been hoisted here by hoist_classical_ops.
func.func @main(%x: tensor<i64>, %qst: !jasp.QuantumState) -> (tensor<f64>, !jasp.QuantumState) {
  %raw, %scale = jasp.call @my_kernel(%x) : (tensor<i64>) -> (tensor<i64>, tensor<f64>)
  %result = "stablehlo.convert"(%raw) : (tensor<i64>) -> tensor<f64>
  %scaled = "stablehlo.multiply"(%result, %scale) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %scaled, %qst : tensor<f64>, !jasp.QuantumState
}

// QPU code — all jasp.quantum_kernel ops live inside jasp.module.
// Body contains only QPU-safe ops (jasp.* and stablehlo.constant).
jasp.module @qpu_module {
  jasp.quantum_kernel private @my_kernel(%x: tensor<i64>, %qst: !jasp.QuantumState) -> (tensor<i64>, tensor<f64>) {
  ^bb0(%x: tensor<i64>, %qst: !jasp.QuantumState):
    ...
    %meas, %qst_out = jasp.measure %qubits, %qst_n : ...
    %scale = "stablehlo.constant"() <{value = dense<1.0> : tensor<f64>}> : () -> tensor<f64>
    jasp.return %meas, %scale, %qst_out : tensor<i64>, tensor<f64>, !jasp.QuantumState
  }
}
```

### Convention

| Location | QuantumState |
|---|---|
| `jasp.quantum_kernel` declared `function_type` | absent (classical-only) |
| Entry block arguments | present as the **last** block arg |
| `jasp.return` operands | present as the **last** operand |
| `jasp.call` operands / results | absent (classical-only) |

This matches the JAX lowering convention (QuantumState at the end).

---

## New Ops

### `jasp.module`

Container for all `jasp.quantum_kernel` ops — analogous to `gpu.module`.

- **Properties:** `sym_name`
- **Region:** one region; single block containing `jasp.quantum_kernel` ops
- **Traits:** `IsolatedFromAbove`, `SymbolOpInterface`
- **Purpose:** structurally separates QPU code from classical host code; a downstream pass can walk `jasp.module` to find all quantum kernels without scanning the whole module.  `IsolatedFromAbove` causes the xDSL verifier to enforce that nothing inside `jasp.module` captures SSA values from the surrounding classical scope.

### `jasp.quantum_kernel`

Analogous to `func.func`. Declares a self-contained quantum function.

- **Properties:** `sym_name`, `function_type` (classical-only), `sym_visibility`
- **Region:** one region; entry block has `(<classical args..., !jasp.QuantumState>)` — QuantumState is the **last** block argument
- **Traits:** `IsolatedFromAbove`, `SymbolOpInterface`
- **Verify:**
  - Parent op must be a `JaspModuleOp` (`HasParent` constraint)
  - Entry block last arg must be `!jasp.QuantumState`
  - Classical block args must match `function_type.inputs`
  - `jasp.return` classical operands must match `function_type.outputs`

### `jasp.return`

Terminator for `jasp.quantum_kernel` bodies.

- **Operands:** `(<classical results..., !jasp.QuantumState>)`
- **Traits:** `IsTerminator`
- **Assembly:** `jasp.return %res, %qst : tensor<f64>, !jasp.QuantumState`

### `jasp.call`

Calls a `jasp.quantum_kernel` by symbol. Purely classical from the caller's view.

- **Properties:** `callee` (flat symbol ref)
- **Operands/Results:** classical types only — QuantumState lifecycle managed by the kernel boundary
- **Assembly:** `jasp.call @name(%args) : (input_types) -> (result_types)`

---

## Implementation

| File | Role |
|---|---|
| `src/qrisp/jasp/mlir/xdsl_dialect.py` | xDSL op definitions (`JaspModuleOp`, `QuantumKernelOp`, `JaspReturnOp`, `JaspCallOp`) and verifiers |
| `src/qrisp/jasp/mlir/lift_quantum_kernels.py` | xDSL pass: rewrites sentinel pattern into new ops, then collects kernels into `jasp.module` |
| `src/qrisp/jasp/mlir/hoist_classical_ops.py` | xDSL pass: moves non-QPU-safe ops out of `jasp.quantum_kernel` into `@main` |
| `src/qrisp/jasp/mlir/drop_dead_wrappers.py` | xDSL pass: erases uncalled `private` `func.func` shadow wrappers emitted by JAX |
| `src/qrisp/jasp/mlir/mlir_emission.py` | Pipeline: `fix_quantum_control_flow` → `lift_quantum_kernels` → `hoist_classical_ops` → `drop_dead_wrappers` |
| `src/qrisp/jasp/mlir/dialect_definition/JaspOps.td` | TableGen definitions for C++ MLIR / `mlir-opt` validation |
| `tests/jax_tests/test_mlir.py` | `test_mlir_quantum_kernel_lifting` verifies the promotion |

### Pass: `lift_quantum_kernels`

Runs after `fix_quantum_control_flow`. Two phases:

**Phase 1 — sentinel promotion.** For each `func.func` in the module, look for the triplet:

1. `%qst = jasp.create_quantum_kernel`
2. `func.call @callee(..., %qst)` — QuantumState last operand/result
3. `jasp.consume_quantum_kernel %qst_out`

Then:
- Converts `@callee` (`func.func`) → `jasp.quantum_kernel`: strips QuantumState
  from the declared `function_type`, rewrites `func.return` → `jasp.return`
- Replaces the triplet with `jasp.call @callee(<classical args>)`

**Phase 2 — module collection.** After all promotions, detaches every
`jasp.quantum_kernel` from `builtin.module` and moves it into a new
`jasp.module @qpu_module` appended at the end of `builtin.module`.

JAX-level tracing (`create_quantum_kernel_p`, `consume_quantum_kernel_p` primitives
and the `quantum_kernel` Python decorator) is unchanged — the transformation is
purely at the xDSL post-processing stage.

### Pass: `hoist_classical_ops`

Runs after `lift_quantum_kernels`. Moves classical (non-QPU-safe) ops out of
every `jasp.quantum_kernel` body and into `@main`, where they execute on the CPU
host after the quantum kernel returns.

**Why it is needed.** After `lift_quantum_kernels` the kernel bodies may still
contain classical `stablehlo` arithmetic that JAX inlined from post-measurement
post-processing (type conversions, scaling, etc.). A real QPU backend (QIR,
OpenQASM) cannot execute these ops, so they must live in the classical host.

**QPU-safe allowlist.** An op may remain inside `jasp.quantum_kernel` if its
name starts with `"jasp."` or it is `stablehlo.constant` (used for qubit counts
and gate angles). Everything else is hoisted.

> **xDSL note.** Unregistered ops (all `stablehlo.*`) have `op.name ==
> "builtin.unregistered"`. Their actual op name is in `op.op_name.data` with
> surrounding quotes — e.g. `'"stablehlo.constant"'`.

**Algorithm** (per kernel):

1. Collect `to_hoist`: entry-block ops not in the QPU-safe allowlist.
2. Collect `extra_kernel_deps`: values produced by allowlisted kernel ops (or
   block args) that `to_hoist` ops consume — these must cross the kernel
   boundary as extra return values.
3. Rebuild `jasp.return`: replace hoisted classical results with
   `extra_kernel_deps`, update `function_type.outputs`.
4. Insert a new `jasp.call` with updated result types.
5. Detach `to_hoist` ops from the kernel, fix their operands using a
   substitution map (`extra_kernel_deps` → new call results; classical block
   args → call operands), and insert them in `@main` after the new call.
6. Redirect all uses of the old call results and erase the old call.

**Before/after example:**

```mlir
// Before
jasp.quantum_kernel @k(%arg: tensor<i64>) -> tensor<f64> {
^bb0(%arg: tensor<i64>, %qst: !jasp.QuantumState):
  ...
  %8, %9 = jasp.measure %qubits, %qst5 : ...
  %10 = "stablehlo.constant"() <{value = dense<1.0> : tensor<f64>}> : () -> tensor<f64>
  %11 = "stablehlo.convert"(%8) : (tensor<i64>) -> tensor<f64>
  %12 = "stablehlo.multiply"(%11, %10) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  jasp.return %12, %9 : tensor<f64>, !jasp.QuantumState
}

// After
func.func @main(...) {
  %raw, %scale = jasp.call @k(...) : (...) -> (tensor<i64>, tensor<f64>)
  %11 = "stablehlo.convert"(%raw) : (tensor<i64>) -> tensor<f64>
  %12 = "stablehlo.multiply"(%11, %scale) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  ...
}
jasp.quantum_kernel @k(%arg: tensor<i64>) -> (tensor<i64>, tensor<f64>) {
^bb0(%arg: tensor<i64>, %qst: !jasp.QuantumState):
  ...
  %8, %9 = jasp.measure %qubits, %qst5 : ...
  %10 = "stablehlo.constant"() <{value = dense<1.0> : tensor<f64>}> : () -> tensor<f64>
  jasp.return %8, %10, %9 : tensor<i64>, tensor<f64>, !jasp.QuantumState
}
```

The `stablehlo.constant` `%10` stays in the kernel (it is in the allowlist) and
is threaded back to `@main` as an extra return value.

**Known limitations / future work:**

- **Single call-site assumption.** `_find_jasp_call` returns the first
  `jasp.call` for a given kernel name. If the same kernel is called from
  multiple locations (or from inside a loop body), only the first call site is
  patched. Multiple call-site support requires iterating all uses.

- **`stablehlo.constant` threading.** A constant that is used exclusively by
  hoisted ops is currently kept in the kernel and threaded back through the
  return signature as an extra value. It would be cheaper to clone the constant
  into `@main` instead, eliminating the extra kernel return.

- **SCF ops with quantum regions.** The pass only inspects the flat entry-block
  op list. `scf.if` / `scf.while` ops that contain `jasp.*` ops inside their
  regions are not in the QPU-safe allowlist, so the pass would attempt to hoist
  them — silently producing invalid IR. In practice the IR produced by the
  current JAX tracing never places `scf.*` directly inside a
  `jasp.quantum_kernel` entry block (quantum control flow is handled before
  `lift_quantum_kernels`), but this is not enforced and should be guarded
  explicitly in a future revision.

### Pass: `drop_dead_wrappers`

Runs last in the pipeline. Erases all `private` `func.func` ops in
`builtin.module` that have no callers.

**Why they exist.** JAX's lowering emits a thin `func.func` wrapper for every
JASP primitive it encounters (e.g. `@jasp.create_qubits`, `@jasp.measure`,
`@jasp.quantum_gate`). These wrappers are the original call targets before the
lowering rules inline the JASP ops directly. After inlining, the wrappers are
never called — they are dead code that bloats the IR.

**What is erased.** Any `func.func` that is:
1. marked `private`, and
2. has no `func.call` or `jasp.call` pointing to it anywhere in the module.

`@main` is always `public` so it is never touched.

---

## Design Decisions

**No nesting.** A `jasp.quantum_kernel` cannot contain another
`jasp.quantum_kernel`. Enforced by convention; no verifier check for this yet.

**QuantumState last, not first.** JAX's lowering puts QuantumState as the last
argument/result throughout. The new ops follow the same convention so the block
arg layout of a promoted callee is identical to the original `func.func`.

**`jasp.call` is classical-only.** External consumers (QIR backends, schedulers)
see a clean classical interface. The quantum state lifecycle is an internal detail
of the kernel, inferred from the op type (`jasp.quantum_kernel`) rather than
explicit operands.

**Shadow wrapper functions are removed.** JAX emits thin `func.func` wrappers
(e.g. `@jasp.create_qubits`, `@jasp.measure`) for every primitive it encounters.
Our lowering rules emit JASP ops inline rather than via `func.call`, so these
wrappers are never called. The `drop_dead_wrappers` pass (run last in the
pipeline) erases all `private` `func.func` ops with no callers, leaving only
`@main` in `builtin.module`.

---

## Reference: GPU Dialect Analogy

| `gpu` dialect | `jasp` dialect |
|---|---|
| `gpu.module` | `jasp.module` |
| `gpu.func` | `jasp.quantum_kernel` |
| `gpu.return` | `jasp.return` |
| `gpu.launch_func` | `jasp.call` |
| implicit thread/block index args in body | implicit `!jasp.QuantumState` arg in body |

**Note:** in the GPU dialect, `gpu.func` has a `HasParent<GPUModuleOp>` constraint
and cannot appear outside `gpu.module`. The same structural rule applies here:
`jasp.quantum_kernel` ops are always collected into `jasp.module` by the
`lift_quantum_kernels` pass, and the `HasParent` constraint is enforced by
`QuantumKernelOp.verify_()` in xDSL.

MLIR `gpu.func` source: `mlir/lib/Dialect/GPU/IR/GPUDialect.cpp`
MLIR GPU dialect docs: https://mlir.llvm.org/docs/Dialects/GPU/
