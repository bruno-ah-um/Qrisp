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
xDSL pass: hoist_classical_ops
================================
Moves classical (non-QPU-safe) ops out of quantum kernel ``func.func`` bodies
and into the classical host function (``@main``) that calls them via
``jasp.call``.

A quantum kernel is any ``func.func`` whose last input/output is
``!jasp.QuantumState``.

After ``lift_quantum_kernels`` the kernel bodies may contain classical
``stablehlo`` arithmetic that was inlined from post-measurement processing, e.g.::

    func.func private @sampling_kernel(..., %qst: !jasp.QuantumState)
            -> (..., !jasp.QuantumState) {
      ...
      %8, %9 = jasp.measure %1, %7 : ...
      %12 = "stablehlo.convert"(%8) ...       ← classical
      %13 = "stablehlo.multiply"(%12, %11) ...  ← classical
      func.return %13, %9 ...
    }

After hoisting::

    func.func @main(...) {
      %raw = jasp.call @sampling_kernel(...) : (...) -> tensor<i1>
      %12 = "stablehlo.convert"(%raw) ...
      %13 = "stablehlo.multiply"(%12, ...) ...
      func.return %13, ...
    }
    func.func private @sampling_kernel(..., %qst: !jasp.QuantumState)
            -> (tensor<i1>, !jasp.QuantumState) {
      ...
      %8, %9 = jasp.measure %1, %7 : ...
      func.return %8, %9 ...
    }

QPU-safe allowlist
------------------
- All ``jasp.*`` ops
- ``stablehlo.constant``  (qubit counts, gate angles)

Everything else is hoisted.
"""

from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.rewriter import InsertPoint, Rewriter

from qrisp.jasp.mlir.xdsl_dialect import (
    JaspCallOp,
    QuantumStateType,
)


def _actual_op_name(op) -> str:
    """Return the real op name, unwrapping xDSL's UnregisteredOp wrapper."""
    if op.name == "builtin.unregistered":
        # UnregisteredOp stores the original name with surrounding quotes.
        # op_name is a StringAttr whose .data holds the name with quotes.
        return op.op_name.data.strip('"')
    return op.name


def _is_qpu_safe(op) -> bool:
    """Return True if *op* is allowed to stay inside a quantum kernel."""
    name = _actual_op_name(op)
    return name.startswith("jasp.") or name == "stablehlo.constant" or name == "func.return"


def _find_jasp_call(module: ModuleOp, kernel_name: str) -> JaspCallOp | None:
    """Return the ``jasp.call @kernel_name`` op inside ``@main``, or None."""
    for top_op in module.body.ops:
        if not isinstance(top_op, FuncOp):
            continue
        for block in top_op.body.blocks:
            for op in block.ops:
                if (
                    isinstance(op, JaspCallOp)
                    and op.callee.root_reference.data == kernel_name
                ):
                    return op
    return None


def _hoist_from_kernel(
    module: ModuleOp,
    kernel_func: FuncOp,
) -> None:
    """Hoist non-QPU-safe ops from *kernel_func* to the caller in *module*."""
    kernel_name = kernel_func.sym_name.data

    call_op = _find_jasp_call(module, kernel_name)
    if call_op is None:
        return

    entry = kernel_func.body.blocks.first
    if entry is None:
        return

    # Collect ops to hoist in block order (preserves topological order).
    to_hoist = [op for op in list(entry.ops) if not _is_qpu_safe(op)]
    if not to_hoist:
        return

    to_hoist_result_set: set = {res for op in to_hoist for res in op.results}

    # Classical block args of the kernel (all except the trailing QuantumState).
    classical_block_args = list(entry.args)[:-1]

    # Values produced by allowlisted kernel ops that are consumed by to_hoist
    # ops — these must become extra kernel return values so they can cross the
    # kernel boundary.
    extra_kernel_deps: list = []
    extra_kernel_deps_id_set: set = set()

    for op in to_hoist:
        for operand in op.operands:
            vid = id(operand)
            if vid in extra_kernel_deps_id_set:
                continue
            if operand in to_hoist_result_set:
                continue  # internal dependency — stays inside to_hoist
            if operand in classical_block_args:
                continue  # available in @main via jasp.call operands
            if isinstance(operand.type, QuantumStateType):
                continue  # quantum state — never crosses the boundary
            # Must be a result of an allowlisted kernel op → extra return.
            extra_kernel_deps.append(operand)
            extra_kernel_deps_id_set.add(vid)

    # ------------------------------------------------------------------
    # Locate the func.return terminator of the kernel entry block.
    # ------------------------------------------------------------------
    func_return = next(
        (op for op in entry.ops if isinstance(op, ReturnOp)), None
    )
    if func_return is None:
        return

    old_classical_returns = list(func_return.arguments)[:-1]
    qst_return = func_return.arguments[-1]

    # Classical returns that are NOT produced by to_hoist ops stay in the kernel.
    kept_returns = [v for v in old_classical_returns if v not in to_hoist_result_set]

    # New kernel return list = kept + extra deps + QuantumState.
    new_classical_returns = kept_returns + extra_kernel_deps
    new_result_types = [v.type for v in new_classical_returns]

    # ------------------------------------------------------------------
    # Update func.return and kernel function_type.
    # ------------------------------------------------------------------
    new_ret = ReturnOp(*list(new_classical_returns + [qst_return]))
    Rewriter.replace_op(func_return, new_ret, new_results=[])

    kernel_func.properties["function_type"] = FunctionType.from_lists(
        list(kernel_func.function_type.inputs),
        new_result_types + [QuantumStateType()],
    )

    # ------------------------------------------------------------------
    # Build new jasp.call with updated result types and insert after old one.
    # ------------------------------------------------------------------
    new_call = JaspCallOp(
        callee=kernel_name,
        arguments=list(call_op.arguments),
        return_types=new_result_types,
    )
    Rewriter.insert_op(new_call, InsertPoint.after(call_op))

    # ------------------------------------------------------------------
    # Replace old jasp.call results with new ones for *kept* returns.
    # Old call result[i] corresponds to old_classical_returns[i].
    # ------------------------------------------------------------------
    for i, old_ret_val in enumerate(old_classical_returns):
        if old_ret_val not in to_hoist_result_set:
            j = kept_returns.index(old_ret_val)
            call_op.res[i].replace_all_uses_with(new_call.res[j])

    # ------------------------------------------------------------------
    # Build value_map: kernel-internal value → equivalent value in @main.
    # ------------------------------------------------------------------
    value_map: dict = {}

    # Kernel classical block args → jasp.call operands (same position).
    for ba, ca in zip(classical_block_args, call_op.arguments):
        value_map[ba] = ca

    # Extra kernel deps → new call results at the end.
    n_kept = len(kept_returns)
    for k, dep in enumerate(extra_kernel_deps):
        value_map[dep] = new_call.res[n_kept + k]

    # ------------------------------------------------------------------
    # Detach to_hoist ops from kernel, fix their operands, insert in @main.
    # ------------------------------------------------------------------
    last_anchor = new_call
    for op in to_hoist:
        # Fix operands that reference kernel-internal values.
        for i, operand in enumerate(list(op.operands)):
            if operand in value_map:
                op.operands[i] = value_map[operand]
        # Move op to @main.
        entry.detach_op(op)
        Rewriter.insert_op(op, InsertPoint.after(last_anchor))
        last_anchor = op

    # ------------------------------------------------------------------
    # For old call results that corresponded to to_hoist results:
    # the hoisted op is now in @main and its result is directly available.
    # ------------------------------------------------------------------
    for i, old_ret_val in enumerate(old_classical_returns):
        if old_ret_val in to_hoist_result_set:
            call_op.res[i].replace_all_uses_with(old_ret_val)

    # Erase the old jasp.call (all its result uses have been redirected).
    Rewriter.erase_op(call_op, safe_erase=False)


def _is_quantum_kernel(func_op: FuncOp) -> bool:
    """Return True if *func_op* is a quantum kernel.

    A quantum kernel is a ``func.func`` whose last input type is
    ``!jasp.QuantumState``.
    """
    inputs = list(func_op.function_type.inputs)
    return bool(inputs) and isinstance(inputs[-1], QuantumStateType)


def hoist_classical_ops(module: ModuleOp) -> None:
    """Hoist non-QPU-safe ops from all quantum kernel ``func.func`` bodies.

    Walks ``func.func`` ops in *module* whose signature contains
    ``!jasp.QuantumState``, and moves any non-allowlisted ops (classical
    arithmetic, type conversions, etc.) to the caller in ``@main``.

    Mutates *module* in-place.
    """
    module_block = module.body.blocks.first
    if module_block is None:
        return

    for func_op in list(module_block.ops):
        if not isinstance(func_op, FuncOp):
            continue
        if not _is_quantum_kernel(func_op):
            continue
        _hoist_from_kernel(module, func_op)
