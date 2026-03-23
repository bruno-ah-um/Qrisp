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

xDSL passes: lower JASP MLIR to EmitC dialect for C++ emission.

Three passes prepare the module for ``mlir-translate --mlir-to-cpp``:

1. ``strip_quantum_state_from_main`` — removes ``!jasp.QuantumState`` from
   ``@main``'s signature so it becomes a purely classical function.

2. ``lower_jasp_call_to_qdmi`` — serialises each quantum kernel ``func.func``
   as an MLIR string constant and replaces the ``jasp.call`` with an
   ``emitc.call_opaque "run_jasp_kernel"(...)`` targeting the QDMI runtime.

3. ``lower_classical_to_emitc`` — rewrites classical ``stablehlo.*`` ops and
   ``func.func`` / ``func.return`` to their EmitC equivalents so the module
   consists entirely of EmitC dialect ops.
"""

from __future__ import annotations

from io import StringIO

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    Float64Type,
    FunctionType,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.emitc import (
    EmitC_CallOpaqueOp,
    EmitC_OpaqueAttr,
)
from xdsl.dialects.builtin import UnregisteredOp
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.printer import Printer
from xdsl.rewriter import InsertPoint, Rewriter

from qrisp.jasp.mlir.xdsl_dialect import JaspCallOp, QuantumStateType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detensorize_type(ty):
    """Map ``tensor<T>`` to the bare scalar type ``T``.

    EmitC works with scalar types (``i64``, ``f64``) rather than rank-0
    tensors.  Types that are not rank-0 tensors are returned unchanged.
    """
    if isinstance(ty, TensorType) and ty.get_shape() == ():
        return ty.get_element_type()
    return ty


def _actual_op_name(op: Operation) -> str:
    """Return the real op name, unwrapping xDSL's UnregisteredOp wrapper."""
    if op.name == "builtin.unregistered":
        return op.op_name.data.strip('"')
    return op.name


def _is_quantum_kernel(func_op: FuncOp) -> bool:
    """A quantum kernel has ``!jasp.QuantumState`` as its last input."""
    inputs = list(func_op.function_type.inputs)
    return bool(inputs) and isinstance(inputs[-1], QuantumStateType)


def _serialize_func(func_op: FuncOp) -> str:
    """Print a ``func.func`` op as MLIR text."""
    buf = StringIO()
    printer = Printer(stream=buf)
    printer.print(func_op)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pass 1: strip_quantum_state_from_main
# ---------------------------------------------------------------------------

def strip_quantum_state_from_main(module: ModuleOp) -> None:
    """Remove ``!jasp.QuantumState`` from ``@main``'s signature and return.

    After ``lift_quantum_kernels`` + ``hoist_classical_ops``, ``@main`` may
    still carry a ``!jasp.QuantumState`` argument/result that is simply
    threaded through (an artefact of the JAX tracing convention).  This pass
    removes it so ``@main`` becomes a purely classical function.

    Mutates *module* in-place.
    """
    for op in module.body.ops:
        if not isinstance(op, FuncOp):
            continue
        if op.sym_name.data != "main":
            continue

        inputs = list(op.function_type.inputs)
        outputs = list(op.function_type.outputs)

        # Check if there is a QuantumState to strip.
        has_qst_in = inputs and isinstance(inputs[-1], QuantumStateType)
        has_qst_out = outputs and isinstance(outputs[-1], QuantumStateType)
        if not has_qst_in and not has_qst_out:
            return  # nothing to do

        # Strip QuantumState from function_type.
        new_inputs = [t for t in inputs if not isinstance(t, QuantumStateType)]
        new_outputs = [t for t in outputs if not isinstance(t, QuantumStateType)]
        op.properties["function_type"] = FunctionType.from_lists(
            new_inputs, new_outputs,
        )

        # Keep res_attrs in sync.
        if "res_attrs" in op.properties:
            op.properties["res_attrs"] = ArrayAttr(
                [DictionaryAttr({})] * len(new_outputs)
            )

        entry = op.body.blocks.first
        if entry is None:
            return

        # Strip QuantumState from func.return operands first (so the
        # block arg loses its last use and can be safely erased).
        ret = next((o for o in entry.ops if isinstance(o, ReturnOp)), None)
        if ret is not None:
            new_ret_operands = [
                v for v in ret.arguments
                if not isinstance(v.type, QuantumStateType)
            ]
            new_ret = ReturnOp(*new_ret_operands)
            Rewriter.replace_op(ret, new_ret, new_results=[])

        # Remove QuantumState block arguments.
        qst_args = [a for a in entry.args if isinstance(a.type, QuantumStateType)]
        for qst_arg in reversed(qst_args):
            entry.erase_arg(qst_arg)
        return


# ---------------------------------------------------------------------------
# Pass 2: lower_jasp_call_to_qdmi
# ---------------------------------------------------------------------------

def lower_jasp_call_to_qdmi(module: ModuleOp) -> None:
    """Replace ``jasp.call`` ops with ``emitc.call_opaque "run_jasp_kernel"``.

    For each quantum kernel ``func.func``:
    1. Serialise it as MLIR text.
    2. Remove it from the module.
    3. Replace each ``jasp.call @kernel(...)`` with
       ``emitc.call_opaque "run_jasp_kernel"(kernel_mlir, args...)``
       whose results are detensorised scalars.

    The serialised MLIR string is stored as an ``emitc.opaque`` attribute
    passed through the ``args`` property of ``emitc.call_opaque``.

    Mutates *module* in-place.
    """
    module_block = module.body.blocks.first
    if module_block is None:
        return

    # Collect quantum kernels and their serialised MLIR.
    kernels: dict[str, str] = {}
    kernel_ops: list[FuncOp] = []
    for op in list(module_block.ops):
        if isinstance(op, FuncOp) and _is_quantum_kernel(op):
            name = op.sym_name.data
            kernels[name] = _serialize_func(op)
            kernel_ops.append(op)

    if not kernels:
        return

    # Replace jasp.call ops.
    for op in list(module_block.ops):
        if not isinstance(op, FuncOp):
            continue
        for block in op.body.blocks:
            for call_op in list(block.ops):
                if not isinstance(call_op, JaspCallOp):
                    continue
                callee = call_op.callee.root_reference.data
                if callee not in kernels:
                    continue

                kernel_mlir = kernels[callee]

                # Build the emitc.call_opaque replacement.
                # Result types: detensorise the jasp.call results.
                new_result_types = [_detensorize_type(r.type) for r in call_op.res]

                # Wrap the kernel MLIR as a C string literal so
                # mlir-translate emits it as a const char*.
                # Escape backslashes, quotes, and newlines for C.
                escaped = kernel_mlir.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                c_string = '"' + escaped + '"'
                mlir_attr = EmitC_OpaqueAttr(StringAttr(c_string))

                from xdsl.dialects.builtin import IntegerAttr as BuiltinIntegerAttr, IndexType
                args_list = [mlir_attr]
                for i in range(len(call_op.arguments)):
                    args_list.append(BuiltinIntegerAttr(i, IndexType()))

                new_call = EmitC_CallOpaqueOp(
                    callee="run_jasp_kernel",
                    call_args=list(call_op.arguments),
                    result_types=new_result_types,
                    args=ArrayAttr(args_list),
                )

                Rewriter.replace_op(call_op, new_call)

    # Remove quantum kernel func.func ops from the module.
    for kop in kernel_ops:
        module_block.detach_op(kop)


# ---------------------------------------------------------------------------
# Pass 3: lower_classical_to_emitc
# ---------------------------------------------------------------------------

def _get_stablehlo_constant_value(op: Operation):
    """Extract the DenseIntOrFPElementsAttr value from a stablehlo.constant."""
    # The value is in op.attributes["value"] for unregistered ops.
    return op.attributes.get("value")


def lower_classical_to_emitc(module: ModuleOp) -> None:
    """Rewrite classical ops and func infrastructure to EmitC dialect.

    Handles:
    - ``stablehlo.constant`` → ``emitc.constant`` (via unregistered op)
    - ``stablehlo.multiply`` → ``emitc.mul`` (via unregistered op)
    - ``stablehlo.add`` → ``emitc.add`` (via xDSL EmitC_AddOp)
    - ``stablehlo.subtract`` → ``emitc.sub`` (via unregistered op)
    - ``stablehlo.convert`` → ``emitc.cast`` (via unregistered op)
    - ``func.func`` → ``emitc.func`` (via unregistered op)
    - ``func.return`` → ``emitc.return`` (via unregistered op)

    Since xDSL's EmitC dialect only defines a subset of ops, we emit most
    EmitC ops as unregistered ops in generic MLIR form — ``mlir-translate``
    parses them fine.

    All ``tensor<T>`` types are detensorised to bare scalar ``T``.

    Mutates *module* in-place.
    """
    module_block = module.body.blocks.first
    if module_block is None:
        return

    for func_op in list(module_block.ops):
        if not isinstance(func_op, FuncOp):
            continue

        entry = func_op.body.blocks.first
        if entry is None:
            continue

        _rewrite_ops_in_block(entry)

    # Rewrite func.func → emitc.func by rebuilding the module textually.
    # Since xDSL doesn't have EmitC_FuncOp, we handle this at the
    # serialisation stage in cpp_emission.py instead.


def _rewrite_ops_in_block(block: Block) -> None:
    """Rewrite stablehlo ops to emitc equivalents within a block."""
    for op in list(block.ops):
        name = _actual_op_name(op)

        if name == "stablehlo.constant":
            _rewrite_constant(op)
        elif name == "stablehlo.multiply":
            _rewrite_binary(op, "emitc.mul")
        elif name == "stablehlo.add":
            _rewrite_binary(op, "emitc.add")
        elif name == "stablehlo.subtract":
            _rewrite_binary(op, "emitc.sub")
        elif name == "stablehlo.convert":
            _rewrite_cast(op)


def _rewrite_constant(op: Operation) -> None:
    """Rewrite ``stablehlo.constant`` → ``emitc.constant``."""
    if not op.results:
        return
    old_result = op.results[0]
    new_type = _detensorize_type(old_result.type)

    # Extract the dense value attribute and convert to scalar.
    value_attr = op.attributes.get("value")
    if value_attr is None:
        return

    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    # Extract scalar value from dense attribute.
    scalar_attr = None
    if isinstance(value_attr, DenseIntOrFPElementsAttr):
        data = list(value_attr.data)
        if len(data) == 1:
            scalar_attr = data[0]

    if scalar_attr is None:
        scalar_attr = value_attr

    EmitCConstant = UnregisteredOp.with_name("emitc.constant")
    new_op = EmitCConstant.create(
        result_types=[new_type],
        attributes={"value": scalar_attr},
    )
    Rewriter.replace_op(op, new_op)


def _rewrite_binary(op: Operation, emitc_name: str) -> None:
    """Rewrite a binary stablehlo op to its emitc equivalent."""
    if len(op.operands) != 2 or not op.results:
        return
    new_type = _detensorize_type(op.results[0].type)

    EmitCBinOp = UnregisteredOp.with_name(emitc_name)
    new_op = EmitCBinOp.create(
        operands=list(op.operands),
        result_types=[new_type],
    )
    Rewriter.replace_op(op, new_op)


def _rewrite_cast(op: Operation) -> None:
    """Rewrite ``stablehlo.convert`` → ``emitc.cast``."""
    if not op.operands or not op.results:
        return
    new_type = _detensorize_type(op.results[0].type)

    EmitCCast = UnregisteredOp.with_name("emitc.cast")
    new_op = EmitCCast.create(
        operands=list(op.operands),
        result_types=[new_type],
    )
    Rewriter.replace_op(op, new_op)
