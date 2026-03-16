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
Transforms the create/consume quantum kernel sentinel pattern into a proper
``jasp.quantum_kernel`` function-defining op with a ``jasp.call`` call site.

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
    jasp.quantum_kernel @my_kernel(%arg: tensor<i64>) -> tensor<f64> {
    ^bb0(%arg: tensor<i64>, %qst: !jasp.QuantumState):
        ...
        jasp.return %result, %qst_out : tensor<f64>, !jasp.QuantumState
    }

Convention: QuantumState is the **last** argument and **last** result in both
the callee's function type and its entry block — this matches the JAX lowering
convention.
"""

from xdsl.dialects.builtin import FunctionType, ModuleOp, SymbolRefAttr
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
    JaspModuleOp,
    JaspReturnOp,
    MeasureOp,
    QuantumGateOp,
    QuantumKernelOp,
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


def _convert_callee(func_op: FuncOp) -> QuantumKernelOp:
    """
    Convert a ``func.func`` whose last input/output is ``!jasp.QuantumState``
    into a ``jasp.quantum_kernel`` with an equivalent body.

    The body is moved as-is (block args and SSA values remain intact).
    Each ``func.return`` in the body is replaced with ``jasp.return``.
    """
    old_ftype = func_op.function_type

    # Strip QuantumState from the declared type → classical-only external type.
    classical_inputs = list(old_ftype.inputs)[:-1]
    classical_outputs = list(old_ftype.outputs)[:-1]
    new_ftype = FunctionType.from_lists(classical_inputs, classical_outputs)

    # Rewrite func.return → jasp.return inside the body.
    # We must collect ops first to avoid mutating while iterating.
    returns_to_rewrite = []
    for block in func_op.body.blocks:
        for op in block.ops:
            if isinstance(op, ReturnOp):
                returns_to_rewrite.append(op)

    for ret_op in returns_to_rewrite:
        jasp_ret = JaspReturnOp(list(ret_op.arguments))
        Rewriter.replace_op(ret_op, jasp_ret, new_results=[])

    # Detach the region from the old func.func and give it to the new op.
    region = func_op.detach_region(func_op.body)

    kernel_op = QuantumKernelOp(
        kernel_name=func_op.sym_name.data,
        function_type=new_ftype,
        region=region,
        visibility=func_op.sym_visibility,
    )
    return kernel_op


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
    """Promote ``func.func @main`` to a ``jasp.quantum_kernel`` when it
    contains quantum ops directly (i.e. no explicit ``@quantum_kernel``
    decorator was used).

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
        jasp.quantum_kernel @main_kernel() -> tensor<i64> {
        ^bb0(%qst: !jasp.QuantumState):
            %qa, %qst1 = jasp.create_qubits ...
            ...
            jasp.return %result, %qst_n : ...
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
    new_ftype = FunctionType.from_lists(classical_inputs, classical_outputs)

    # --- Rewrite func.return → jasp.return inside the body ---
    returns_to_rewrite = [
        op
        for block in main_op.body.blocks
        for op in block.ops
        if isinstance(op, ReturnOp)
    ]
    for ret_op in returns_to_rewrite:
        jasp_ret = JaspReturnOp(list(ret_op.arguments))
        Rewriter.replace_op(ret_op, jasp_ret, new_results=[])

    # --- Promote @main body → jasp.quantum_kernel @main_kernel ---
    region = main_op.detach_region(main_op.body)
    kernel_op = QuantumKernelOp(
        kernel_name="main_kernel",
        function_type=new_ftype,
        region=region,
    )

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

    # Replace old @main with the new classical wrapper; append the kernel
    # (it will be moved into jasp.module by _collect_into_jasp_module).
    Rewriter.replace_op(main_op, new_main, new_results=[])
    module_block.add_op(kernel_op)


def lift_quantum_kernels(module: ModuleOp) -> None:
    """
    Walk all ``func.func`` ops in *module* and lift any that contain the
    create/call/consume triplet into proper ``jasp.quantum_kernel`` ops.

    Mutates *module* in-place.
    """
    # Collect candidates first — we mutate the module body during the loop.
    func_ops = [op for op in module.body.ops if isinstance(op, FuncOp)]

    # Build a symbol → FuncOp map for callee lookup.
    symbol_table: dict[str, FuncOp] = {
        op.sym_name.data: op for op in func_ops
    }

    # Track which FuncOps have been promoted to QuantumKernelOps already so we
    # don't process the same callee more than once.
    promoted: set[str] = set()

    for func_op in func_ops:
        pattern = _find_kernel_pattern(func_op)
        if pattern is None:
            continue

        create_op, call_op, consume_op = pattern

        callee_name = call_op.callee.root_reference.data
        if callee_name in promoted:
            # Already lifted — just rewrite the call site.
            _rewrite_call_site(create_op, call_op, consume_op)
            continue

        callee_func = symbol_table.get(callee_name)
        if callee_func is None:
            # Callee not found in this module — skip.
            continue

        # Promote the callee func.func → jasp.quantum_kernel.
        kernel_op = _convert_callee(callee_func)

        # Insert the new quantum_kernel op at the same position in the module.
        Rewriter.replace_op(callee_func, kernel_op, new_results=[])
        promoted.add(callee_name)

        # Rewrite the call site: replace create + call + consume with jasp.call.
        _rewrite_call_site(create_op, call_op, consume_op)

    # Handle the implicit case: @main itself is a quantum kernel (no decorator).
    _lift_implicit_main_kernel(module)

    # After all promotions, move every jasp.quantum_kernel into a jasp.module.
    _collect_into_jasp_module(module)


def _collect_into_jasp_module(module: ModuleOp) -> None:
    """Move all jasp.quantum_kernel ops into a single jasp.module container.

    Before::

        builtin.module @jasp_module {
          func.func @main(...) { ... jasp.call @my_kernel ... }
          jasp.quantum_kernel @my_kernel(...) { ... }
          ...
        }

    After::

        builtin.module @jasp_module {
          func.func @main(...) { ... jasp.call @my_kernel ... }
          ...
          jasp.module @qpu_module {
            jasp.quantum_kernel @my_kernel(...) { ... }
          }
        }
    """
    module_block = module.body.blocks.first
    if module_block is None:
        return

    kernel_ops = [op for op in module_block.ops if isinstance(op, QuantumKernelOp)]
    if not kernel_ops:
        return

    # Detach kernels from builtin.module and collect them.
    for op in kernel_ops:
        module_block.detach_op(op)

    # Build a new jasp.module containing all kernels.
    qpu_block = Block()
    for op in kernel_ops:
        qpu_block.add_op(op)
    jasp_mod = JaspModuleOp("qpu_module", Region([qpu_block]))
    module_block.add_op(jasp_mod)


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
