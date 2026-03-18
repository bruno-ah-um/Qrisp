# JASP Dialect: `jasp.call`

Design notes and implementation reference for the `jasp.call` op and the
quantum kernel convention.

---

## Motivation

JAX lowering emits sentinel ops to bracket quantum computation:

```mlir
func.func @main(...) {
  %qst = jasp.create_quantum_kernel -> !jasp.QuantumState
  %res, %qst_out = func.call @my_func(%arg, %qst) : ...
  %_ = jasp.consume_quantum_kernel %qst_out : ...
}
```

The `lift_quantum_kernels` pass replaces this triplet with a single
`jasp.call`, which presents a purely classical interface to the caller.
The callee remains a regular `func.func` — it is identified as a quantum
kernel by having `!jasp.QuantumState` as its last input and output.

This follows the principle of reusing existing MLIR infrastructure
(`func.func`, `func.return`) wherever possible, and only introducing new
ops where semantically necessary.

---

## IR Shape

After the full pipeline (`lift_quantum_kernels` + `hoist_classical_ops`):

```mlir
// Classical host code — lives directly in builtin.module.
// Post-measurement arithmetic has been hoisted here by hoist_classical_ops.
func.func @main(%x: tensor<i64>) -> tensor<f64> {
  %raw, %scale = jasp.call @my_kernel(%x) : (tensor<i64>) -> (tensor<i64>, tensor<f64>)
  %result = "stablehlo.convert"(%raw) : (tensor<i64>) -> tensor<f64>
  %scaled = "stablehlo.multiply"(%result, %scale) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %scaled : tensor<f64>
}

// Quantum kernel — a func.func identified by QuantumState in its signature.
// Body contains only QPU-safe ops (jasp.* and stablehlo.constant).
func.func private @my_kernel(%x: tensor<i64>, %qst: !jasp.QuantumState)
    -> (tensor<i64>, tensor<f64>, !jasp.QuantumState) {
  ...
  %meas, %qst_out = jasp.measure %qubits, %qst_n : ...
  %scale = "stablehlo.constant"() <{value = dense<1.0> : tensor<f64>}> : () -> tensor<f64>
  func.return %meas, %scale, %qst_out : tensor<i64>, tensor<f64>, !jasp.QuantumState
}
```

### Convention

| Location | QuantumState |
|---|---|
| Quantum kernel `func.func` `function_type` | present as the **last** input and **last** output |
| Quantum kernel `func.return` operands | present as the **last** operand |
| `jasp.call` operands / results | absent (classical-only) |

A quantum kernel is identified by checking whether the last input type of a
`func.func` is `!jasp.QuantumState`.  This matches the JAX lowering convention.

---

## New Op

### `jasp.call`

Calls a quantum kernel (`func.func` with QuantumState in its signature) by
symbol.  Purely classical from the caller's view.

- **Properties:** `callee` (flat symbol ref)
- **Operands/Results:** classical types only — QuantumState lifecycle managed at the call boundary
- **Assembly:** `jasp.call @name(%args) : (input_types) -> (result_types)`

---

## Implementation

| File | Role |
|---|---|
| `src/qrisp/jasp/mlir/xdsl_dialect.py` | xDSL op definition (`JaspCallOp`) and JASP dialect types |
| `src/qrisp/jasp/mlir/lift_quantum_kernels.py` | xDSL pass: replaces sentinel triplet with `jasp.call`; callee `func.func` is left unchanged |
| `src/qrisp/jasp/mlir/hoist_classical_ops.py` | xDSL pass: moves non-QPU-safe ops out of quantum kernel `func.func` into `@main` |
| `src/qrisp/jasp/mlir/drop_dead_wrappers.py` | xDSL pass: erases uncalled `private` `func.func` wrappers emitted by JAX |
| `src/qrisp/jasp/mlir/mlir_emission.py` | Pipeline: `fix_quantum_control_flow` → `lift_quantum_kernels` → `hoist_classical_ops` → `drop_dead_wrappers` |
| `src/qrisp/jasp/mlir/dialect_definition/JaspOps.td` | TableGen definitions for C++ MLIR / `mlir-opt` validation |
| `tests/jax_tests/test_mlir.py` | `test_mlir_quantum_kernel_lifting` verifies the transformation |

### Pass: `lift_quantum_kernels`

Runs after `fix_quantum_control_flow`.  For each `func.func` in the module,
looks for the sentinel triplet:

1. `%qst = jasp.create_quantum_kernel`
2. `func.call @callee(..., %qst)` — QuantumState last operand/result
3. `jasp.consume_quantum_kernel %qst_out`

Then replaces the triplet with `jasp.call @callee(<classical args>)`.  The
callee `func.func` is **not** modified — it keeps QuantumState in its
signature and uses `func.return`.

For programs without an explicit `@quantum_kernel` decorator (i.e. `@main`
itself contains quantum ops), the pass extracts `@main`'s body into a new
`func.func private @main_kernel` and inserts a `jasp.call` wrapper.

JAX-level tracing (`create_quantum_kernel_p`, `consume_quantum_kernel_p`
primitives and the `quantum_kernel` Python decorator) is unchanged — the
transformation is purely at the xDSL post-processing stage.

### Pass: `hoist_classical_ops`

Runs after `lift_quantum_kernels`. Moves classical (non-QPU-safe) ops out of
every quantum kernel `func.func` body and into `@main`, where they execute
on the CPU host after the quantum kernel returns.

**Why it is needed.** After `lift_quantum_kernels` the kernel bodies may still
contain classical `stablehlo` arithmetic that JAX inlined from post-measurement
post-processing (type conversions, scaling, etc.). A real QPU backend (QIR,
OpenQASM) cannot execute these ops, so they must live in the classical host.

**QPU-safe allowlist.** An op may remain inside a quantum kernel if its
name starts with `"jasp."`, it is `stablehlo.constant` (used for qubit counts
and gate angles), or it is `func.return`. Everything else is hoisted.

> **xDSL note.** Unregistered ops (all `stablehlo.*`) have `op.name ==
> "builtin.unregistered"`. Their actual op name is in `op.op_name.data` with
> surrounding quotes — e.g. `'"stablehlo.constant"'`.

**Algorithm** (per kernel):

1. Collect `to_hoist`: entry-block ops not in the QPU-safe allowlist.
2. Collect `extra_kernel_deps`: values produced by allowlisted kernel ops (or
   block args) that `to_hoist` ops consume — these must cross the kernel
   boundary as extra return values.
3. Rebuild `func.return`: replace hoisted classical results with
   `extra_kernel_deps`, update `function_type.outputs`.
4. Insert a new `jasp.call` with updated result types.
5. Detach `to_hoist` ops from the kernel, fix their operands using a
   substitution map (`extra_kernel_deps` → new call results; classical block
   args → call operands), and insert them in `@main` after the new call.
6. Redirect all uses of the old call results and erase the old call.

**Before/after example:**

```mlir
// Before
func.func private @k(%arg: tensor<i64>, %qst: !jasp.QuantumState)
    -> (tensor<f64>, !jasp.QuantumState) {
  ...
  %8, %9 = jasp.measure %qubits, %qst5 : ...
  %10 = "stablehlo.constant"() <{value = dense<1.0> : tensor<f64>}> : () -> tensor<f64>
  %11 = "stablehlo.convert"(%8) : (tensor<i64>) -> tensor<f64>
  %12 = "stablehlo.multiply"(%11, %10) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %12, %9 : tensor<f64>, !jasp.QuantumState
}

// After
func.func @main(...) {
  %raw, %scale = jasp.call @k(...) : (...) -> (tensor<i64>, tensor<f64>)
  %11 = "stablehlo.convert"(%raw) : (tensor<i64>) -> tensor<f64>
  %12 = "stablehlo.multiply"(%11, %scale) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  ...
}
func.func private @k(%arg: tensor<i64>, %qst: !jasp.QuantumState)
    -> (tensor<i64>, tensor<f64>, !jasp.QuantumState) {
  ...
  %8, %9 = jasp.measure %qubits, %qst5 : ...
  %10 = "stablehlo.constant"() <{value = dense<1.0> : tensor<f64>}> : () -> tensor<f64>
  func.return %8, %10, %9 : tensor<i64>, tensor<f64>, !jasp.QuantumState
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
  current JAX tracing never places `scf.*` directly inside a quantum kernel
  entry block (quantum control flow is handled before `lift_quantum_kernels`),
  but this is not enforced and should be guarded explicitly in a future revision.

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

**Reuse `func.func` and `func.return`.** Rather than introducing custom
`jasp.quantum_kernel` and `jasp.return` ops, quantum kernels are plain
`func.func` ops. A quantum kernel is identified by checking whether its
signature contains `!jasp.QuantumState`. This follows the MLIR principle of
reusing existing dialect infrastructure and avoids the maintenance cost of
additional ops that every pass would need to handle.

**Only `jasp.call` is new.** `jasp.call` is the one genuinely new op: it
presents a classical-only call interface, hiding the QuantumState lifecycle
from the caller. `func.call` cannot serve this role because its operands/results
must match the callee's `function_type`, which includes QuantumState.

**QuantumState last, not first.** JAX's lowering puts QuantumState as the last
argument/result throughout. The convention is preserved so that the block arg
layout of a quantum kernel is identical to the `func.func` emitted by JAX.

**`jasp.call` is classical-only.** External consumers (QIR backends, schedulers)
see a clean classical interface. The quantum state lifecycle is an internal detail
of the kernel, inferred from the callee's signature rather than explicit operands.

**Shadow wrapper functions are removed.** JAX emits thin `func.func` wrappers
(e.g. `@jasp.create_qubits`, `@jasp.measure`) for every primitive it encounters.
Our lowering rules emit JASP ops inline rather than via `func.call`, so these
wrappers are never called. The `drop_dead_wrappers` pass (run last in the
pipeline) erases all `private` `func.func` ops with no callers, leaving only
`@main` and quantum kernel functions in `builtin.module`.
