"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

"""
xDSL pass: lift_quantum_kernels
================================
Replaces the create/consume quantum kernel sentinel pattern with a
``jasp.call`` at the call site.  The callee ``func.func`` is left unchanged
— it is identified as a quantum kernel by having ``!jasp.QuantumState`` as
its last input and output.

Before (emitted by JAX lowering)::

    func.func @main(%arg: tensor<i64>, %qst_outer: !jasp.QuantumState) -> (...) {
        %qst = jasp.create_quantum_kernel -> !jasp.QuantumState
        %result, %qst_out = func.call @my_kernel(%arg, %qst) : ...
        %_ = jasp.consume_quantum_kernel %qst_out : ...
        func.return %result, %qst_outer : ...
    }
    func.func private @my_kernel(%arg: tensor<i64>, %qst: !jasp.QuantumState)
            -> (tensor<f64>, !jasp.QuantumState) { ... }

After::

    func.func @main(%arg: tensor<i64>, %qst_outer: !jasp.QuantumState) -> (...) {
        %result = jasp.call @my_kernel(%arg) : (tensor<i64>) -> tensor<f64>
        func.return %result, %qst_outer : ...
    }
    func.func private @my_kernel(%arg: tensor<i64>, %qst: !jasp.QuantumState)
            -> (tensor<f64>, !jasp.QuantumState) { ... }

Convention: QuantumState is the **last** argument and **last** result in the
callee's function type — this matches the JAX lowering convention.
"""

from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region
from xdsl.rewriter import Rewriter

from qrisp.jasp.mlir.xdsl_dialect import (
    ConsumeQuantumKernelOp,
    CreateQuantumKernelOp,
    CreateQubitsOp,
    DeleteQubitsOp,
    FuseOp,
    GetQubitOp,
    GetSizeOp,
    JaspCallOp,
    MeasureOp,
    QuantumGateOp,
    QuantumStateType,
    ResetOp,
    SliceOp,
)

# Ops that constitute actual QPU work (not bookkeeping sentinels).
_QUANTUM_OPS = (
    CreateQubitsOp,
    GetQubitOp,
    GetSizeOp,
    SliceOp,
    FuseOp,
    ResetOp,
    MeasureOp,
    DeleteQubitsOp,
    QuantumGateOp,
)


def _is_quantum_state(value) -> bool:
    return isinstance(value.type, QuantumStateType)


def _find_kernel_pattern(func_op: FuncOp):
    """
    Scan the entry block of *func_op* for the triplet:

        %qst   = jasp.create_quantum_kernel
        %r, %q = func.call @callee(..., %qst)   # QuantumState last operand/result
        %_     = jasp.consume_quantum_kernel %q

    Returns ``(create_op, call_op, consume_op)`` or ``None`` if not found.
    """
    if not func_op.regions or func_op.body.blocks.first is None:
        return None

    for block in func_op.body.blocks:
        for op in block.ops:
            if not isinstance(op, CreateQuantumKernelOp):
                continue
            qst_val = op.result

            # Find the func.call that uses this QuantumState as its last operand.
            call_op = None
            for use in qst_val.uses:
                user = use.operation
                if (
                    isinstance(user, CallOp)
                    and user.arguments
                    and _is_quantum_state(user.arguments[-1])
                    and user.arguments[-1] == qst_val
                ):
                    call_op = user
                    break

            if call_op is None:
                continue

            # Find the QuantumState result of the call.
            if not call_op.res or not _is_quantum_state(call_op.res[-1]):
                continue
            qst_out = call_op.res[-1]

            # Find the consume_quantum_kernel that uses qst_out.
            consume_op = None
            for use in qst_out.uses:
                user = use.operation
                if isinstance(user, ConsumeQuantumKernelOp):
                    consume_op = user
                    break

            if consume_op is None:
                continue

            return (op, call_op, consume_op)

    return None


def _has_quantum_ops(op) -> bool:
    """Recursively walk all nested regions to find any JASP quantum op."""
    for region in op.regions:
        for block in region.blocks:
            for nested_op in block.ops:
                if isinstance(nested_op, _QUANTUM_OPS):
                    return True
                if _has_quantum_ops(nested_op):
                    return True
    return False


def _lift_implicit_main_kernel(module: ModuleOp) -> None:
    """Extract the quantum body of ``func.func @main`` into a separate
    ``func.func @main_kernel`` when ``@main`` contains quantum ops directly
    (i.e. no explicit ``@quantum_kernel`` decorator was used).

    Before::

        func.func @main(%qst: !jasp.QuantumState) -> (tensor<i64>, !jasp.QuantumState) {
            %qa, %qst1 = jasp.create_qubits ...
            ...
            func.return %result, %qst_n : ...
        }

    After::

        func.func @main() -> tensor<i64> {
            %result = jasp.call @main_kernel() : () -> tensor<i64>
            func.return %result : tensor<i64>
        }
        func.func private @main_kernel(%qst: !jasp.QuantumState)
                -> (tensor<i64>, !jasp.QuantumState) {
            %qa, %qst1 = jasp.create_qubits ...
            ...
            func.return %result, %qst_n : ...
        }
    """
    module_block = module.body.blocks.first
    if module_block is None:
        return

    # Find func.func @main.
    main_op = next(
        (op for op in module_block.ops
         if isinstance(op, FuncOp) and op.sym_name.data == "main"),
        None,
    )
    if main_op is None:
        return

    # Must have QuantumState as last argument — otherwise it's purely classical.
    entry = main_op.body.blocks.first
    if entry is None or not entry.args:
        return
    if not isinstance(entry.args[-1].type, QuantumStateType):
        return

    # Only lift if @main actually contains quantum ops (not just a jasp.call
    # wrapper produced by the explicit-kernel pass).
    if not _has_quantum_ops(main_op):
        return

    # --- Derive classical types ---
    old_ftype = main_op.function_type
    classical_inputs = list(old_ftype.inputs)[:-1]   # drop trailing QuantumState
    classical_outputs = list(old_ftype.outputs)[:-1]  # drop trailing QuantumState

    # --- Move @main body into func.func private @main_kernel ---
    # The body (including its func.return terminators) is moved as-is.
    region = main_op.detach_region(main_op.body)
    kernel_func = FuncOp("main_kernel", old_ftype, region)
    kernel_func.properties["sym_visibility"] = StringAttr("private")

    # --- Build new classical func.func @main wrapper ---
    new_block = Block(arg_types=classical_inputs)
    jasp_call = JaspCallOp(
        callee="main_kernel",
        arguments=list(new_block.args),
        return_types=classical_outputs,
    )
    new_block.add_op(jasp_call)
    new_block.add_op(ReturnOp(*list(jasp_call.res)))

    new_main = FuncOp(
        "main",
        FunctionType.from_lists(classical_inputs, classical_outputs),
        Region([new_block]),
    )

    # Replace old @main with the new classical wrapper; append the kernel.
    Rewriter.replace_op(main_op, new_main, new_results=[])
    module_block.add_op(kernel_func)


def lift_quantum_kernels(module: ModuleOp) -> None:
    """
    Walk all ``func.func`` ops in *module* and replace any
    create/call/consume sentinel triplets with ``jasp.call``.

    The callee ``func.func`` is left unchanged — a quantum kernel is simply
    a ``func.func`` whose last input/output is ``!jasp.QuantumState``.

    Mutates *module* in-place.
    """
    # Collect candidates first — we mutate the module body during the loop.
    func_ops = [op for op in module.body.ops if isinstance(op, FuncOp)]

    # Track which callees we've already seen so we don't process the same
    # callee more than once.
    processed: set[str] = set()

    for func_op in func_ops:
        pattern = _find_kernel_pattern(func_op)
        if pattern is None:
            continue

        create_op, call_op, consume_op = pattern
        callee_name = call_op.callee.root_reference.data
        processed.add(callee_name)

        # Rewrite the call site: replace create + call + consume with jasp.call.
        _rewrite_call_site(create_op, call_op, consume_op)

    # Handle the implicit case: @main itself is a quantum kernel (no decorator).
    _lift_implicit_main_kernel(module)


def _rewrite_call_site(
    create_op: CreateQuantumKernelOp,
    call_op: CallOp,
    consume_op: ConsumeQuantumKernelOp,
) -> None:
    """
    Replace the (create, func.call, consume) triplet with a single ``jasp.call``.

    The classical arguments and results are preserved; QuantumState values are
    dropped from the call site.
    """
    callee_name = call_op.callee.root_reference.data

    # Classical args = all call operands except the last (QuantumState).
    classical_args = list(call_op.arguments)[:-1]

    # Classical result types = all result types except the last (QuantumState).
    classical_result_types = [r.type for r in call_op.res][:-1]

    jasp_call = JaspCallOp(
        callee=callee_name,
        arguments=classical_args,
        return_types=classical_result_types,
    )

    # Replace func.call with jasp.call.
    # Map: classical results → new jasp.call results; QuantumState result → None (erased).
    new_results = list(jasp_call.res) + [None]
    Rewriter.replace_op(call_op, jasp_call, new_results=new_results, safe_erase=False)

    # Erase create_quantum_kernel (its only user was the func.call, now gone).
    Rewriter.erase_op(create_op, safe_erase=False)

    # Erase consume_quantum_kernel (its QuantumState input is gone; result unused).
    Rewriter.erase_op(consume_op, safe_erase=False)
