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
xDSL dialect definition for the JASP (Qrisp) quantum dialect.

This module registers the JASP dialect with xDSL so that:
- JASP types (QuantumState, Qubit, QubitArray) parse and print without quotes
- JASP ops are recognised as typed IRDLOperation subclasses instead of
  UnregisteredOp, enabling typed pattern matching in xDSL passes
- Non-variadic ops print in their custom assembly format (no quoted op names)
"""

from xdsl.dialects.builtin import (
    ArrayAttr,
    FlatSymbolRefAttr,
    FunctionType,
    IntegerAttr,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    f64,
    i1,
    i64,
)
from xdsl.dialects.func import FlatSymbolRefAttrConstr, parse_func_op_like, print_func_op_like
from xdsl.ir import Dialect, ParametrizedAttribute, Region, TypeAttribute
from xdsl.utils.exceptions import VerifyException
from xdsl.irdl import (
    AnyAttr,
    EqAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    attr_def,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import IsolatedFromAbove, IsTerminator, SymbolOpInterface


def _scalar_tensor(elem_type):
    """Return an xDSL constraint matching a rank-0 tensor of *elem_type*.

    This is the xDSL equivalent of MLIR TableGen's ``0DTensorOf<[T]>``.
    TensorType parameters are ordered as (shape, element_type, encoding).
    """
    return ParamAttrConstraint(
        TensorType,
        [
            EqAttrConstraint(ArrayAttr([])),  # rank-0 shape
            EqAttrConstraint(elem_type),
            AnyAttr(),  # encoding — not constrained
        ],
    )


# Shorthand constraints for builtin tensor operands/results.
_TensorI64 = _scalar_tensor(i64)  # 0DTensorOf<[I64]>
_TensorI1 = _scalar_tensor(i1)  # 0DTensorOf<[I1]>
# xDSL's AnyOf rejects two ParamAttrConstraints with the same base type unless
# they are pure equality constraints, so the union tensor<i1>|tensor<i64> falls
# back to base(TensorType).  Exact element-type checking is handled by the
# surrounding MeasureOp logic at runtime.
_TensorInt = base(TensorType)  # AnyTypeOf<[0DTensorOf<[I1]>, 0DTensorOf<[I64]>]>


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@irdl_attr_definition
class QuantumStateType(ParametrizedAttribute, TypeAttribute):
    """An opaque type describing the quantum state of the machine.

    This object is passed around the program to capture the computations.
    """

    name = "jasp.QuantumState"


@irdl_attr_definition
class QubitType(ParametrizedAttribute, TypeAttribute):
    """A type describing an individual qubit.

    Qubit objects are semantically identical to integers as they simply index
    the QuantumState. This especially implies that it is semantically
    well-defined to copy a qubit.
    """

    name = "jasp.Qubit"


@irdl_attr_definition
class QubitArrayType(ParametrizedAttribute, TypeAttribute):
    """A type describing a dynamically-sized collection of Qubits.

    QubitArrays enable expression of dynamically-sized programs. They are
    semantically equivalent to immutable arrays of integers.
    """

    name = "jasp.QubitArray"


# Shorthand constraint for operands that accept either a Qubit or a QubitArray.
_QubitOrArray = irdl_to_attr_constraint(QubitType | QubitArrayType)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


@irdl_op_definition
class CreateQubitsOp(IRDLOperation):
    """Allocates a QubitArray containing n qubits.

    N can be dynamically sized.
    """

    name = "jasp.create_qubits"

    amount = operand_def(_TensorI64)
    qst_in = operand_def(QuantumStateType)

    result = result_def(QubitArrayType)
    qst_out = result_def(QuantumStateType)

    assembly_format = (
        "$amount attr-dict `,` $qst_in"
        " `:` type($qst_in) `,` type($amount)"
        " `->` type($result) `,` type($qst_out)"
    )


@irdl_op_definition
class GetQubitOp(IRDLOperation):
    """Retrieves a single qubit from a QubitArray at a given position."""

    name = "jasp.get_qubit"

    qb_array = operand_def(QubitArrayType)
    position = operand_def(_TensorI64)

    result = result_def(QubitType)

    assembly_format = (
        "$qb_array `,` $position attr-dict"
        " `:` type($qb_array) `,` type($position)"
        " `->` type($result)"
    )


@irdl_op_definition
class GetSizeOp(IRDLOperation):
    """Returns the number of qubits in a given QubitArray."""

    name = "jasp.get_size"

    qb_array = operand_def(QubitArrayType)

    size = result_def(_TensorI64)

    assembly_format = "$qb_array attr-dict `:` type($qb_array) `->` type($size)"


@irdl_op_definition
class SliceOp(IRDLOperation):
    """Returns a subset of qubits from a QubitArray using start and end indices."""

    name = "jasp.slice"

    qb_array = operand_def(QubitArrayType)
    start = operand_def(_TensorI64)
    end = operand_def(_TensorI64)

    result = result_def(QubitArrayType)

    assembly_format = (
        "$qb_array `,` $start `,` $end attr-dict"
        " `:` type($qb_array) `,` type($start) `,` type($end)"
        " `->` type($result)"
    )


@irdl_op_definition
class FuseOp(IRDLOperation):
    """Concatenates two qubits or qubit arrays.

    Fuses two QubitArrays, Qubits, or combinations thereof to create a larger
    QubitArray.
    """

    name = "jasp.fuse"

    operand1 = operand_def(_QubitOrArray)
    operand2 = operand_def(_QubitOrArray)

    result = result_def(QubitArrayType)

    assembly_format = (
        "$operand1 `,` $operand2 attr-dict"
        " `:` type($operand1) `,` type($operand2)"
        " `->` type($result)"
    )


@irdl_op_definition
class ResetOp(IRDLOperation):
    """Resets qubits to the |0> state.

    Performs a reset operation on a single qubit or qubit array, returning
    them to the |0> state.
    """

    name = "jasp.reset"

    qubits = operand_def(_QubitOrArray)
    in_qst = operand_def(QuantumStateType)

    out_qst = result_def(QuantumStateType)

    assembly_format = (
        "$qubits `,` $in_qst attr-dict"
        " `:` type($qubits) `,` type($in_qst)"
        " `->` type($out_qst)"
    )


@irdl_op_definition
class MeasureOp(IRDLOperation):
    """The measurement operation.

    Performs a measurement of a given quantum state on a given qubit or qubit
    array.
    """

    name = "jasp.measure"

    meas_q = operand_def(_QubitOrArray)
    in_qst = operand_def(QuantumStateType)

    meas_res = result_def(_TensorInt)
    out_qst = result_def(QuantumStateType)

    assembly_format = (
        "$meas_q `,` $in_qst attr-dict"
        " `:` type($meas_q) `,` type($in_qst)"
        " `->` type($meas_res) `,` type($out_qst)"
    )

    def verify_(self) -> None:
        expected = (
            TensorType(i64, [])
            if isinstance(self.meas_q.type, QubitArrayType)
            else TensorType(i1, [])
        )
        if self.meas_res.type != expected:
            raise VerifyException(
                f"jasp.measure: result type must be '{expected}' when "
                f"measuring '{self.meas_q.type}', got '{self.meas_res.type}'"
            )


@irdl_op_definition
class DeleteQubitsOp(IRDLOperation):
    """Deallocates qubits from a QubitArray.

    Indicates to the execution environment that the corresponding qubits can
    be reused.
    """

    name = "jasp.delete_qubits"

    qubits = operand_def(QubitArrayType)
    in_qst = operand_def(QuantumStateType)

    out_qst = result_def(QuantumStateType)

    assembly_format = (
        "$qubits `,` $in_qst attr-dict"
        " `:` type($qubits) `,` type($in_qst)"
        " `->` type($out_qst)"
    )


@irdl_op_definition
class CreateQuantumKernelOp(IRDLOperation):
    """Creates a quantum state from nothing.

    Indicates to the execution environment that a quantum computation will
    start.
    """

    name = "jasp.create_quantum_kernel"

    result = result_def(QuantumStateType)

    assembly_format = "attr-dict `->` type($result)"


@irdl_op_definition
class ConsumeQuantumKernelOp(IRDLOperation):
    """Destroys the quantum state.

    Indicates to the execution environment that the quantum computation has
    concluded.
    """

    name = "jasp.consume_quantum_kernel"

    qst = operand_def(QuantumStateType)
    success = result_def(_TensorI1)

    assembly_format = "$qst attr-dict `:` type($qst) `->` type($success)"


@irdl_op_definition
class QuantumGateOp(IRDLOperation):
    """The Quantum Gate operation.

    This operation enables quantum processing of quantum states with
    (parametric) gates.
    """

    name = "jasp.quantum_gate"

    gate_type = attr_def(StringAttr)
    gate_params = var_operand_def()
    in_qst = operand_def(QuantumStateType)
    out_qst = result_def(QuantumStateType)

    def print(self, printer: Printer) -> None:
        # Mirrors TableGen assemblyFormat:
        # $gate_type `(` $gate_params `)` `,` $in_qst attr-dict
        #     `:` `(` type($gate_params) `)` `,` type($in_qst) `->` type($out_qst)
        printer.print_string(" ")
        printer.print_attribute(self.gate_type)
        printer.print_string(" (")
        printer.print_list(self.gate_params, printer.print_ssa_value)
        printer.print_string(") , ")
        printer.print_ssa_value(self.in_qst)
        printer.print_string(" : (")
        printer.print_list(self.gate_params, lambda v: printer.print_attribute(v.type))
        printer.print_string(") , ")
        printer.print_attribute(self.in_qst.type)
        printer.print_string(" -> ")
        printer.print_attribute(self.out_qst.type)

    def verify_(self) -> None:
        _valid_param_types = (QubitType, TensorType)
        for i, param in enumerate(self.gate_params):
            if not isinstance(param.type, _valid_param_types):
                raise VerifyException(
                    f"jasp.quantum_gate: gate_params[{i}] must be '!jasp.Qubit' or 'tensor<f64>', got '{param.type}'"
                )
            if isinstance(param.type, TensorType) and param.type != TensorType(f64, []):
                raise VerifyException(
                    f"jasp.quantum_gate: tensor gate_params[{i}] must be 'tensor<f64>', got '{param.type}'"
                )


@irdl_op_definition
class ParityOp(IRDLOperation):
    """Computes parity of measurement results.

    Computes the parity (XOR sum) of a set of measurement results. Supports
    expectation and observable attributes for error correction contexts.
    """

    name = "jasp.parity"

    expectation = attr_def(IntegerAttr)
    observable = attr_def(IntegerAttr)
    measurements = var_operand_def(_TensorI1)
    result = result_def(_TensorI1)

    def print(self, printer: Printer) -> None:
        # Mirrors TableGen assemblyFormat:
        # $measurements attr-dict `:` type($measurements) `->` type($result)
        printer.print_string(" ")
        printer.print_list(self.measurements, printer.print_ssa_value)
        printer.print_string(" {expectation = ")
        printer.print_attribute(self.expectation)
        printer.print_string(", observable = ")
        printer.print_attribute(self.observable)
        printer.print_string("} : ")
        printer.print_list(self.measurements, lambda v: printer.print_attribute(v.type))
        printer.print_string(" -> ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class JaspModuleOp(IRDLOperation):
    """Container for jasp.quantum_kernel ops — analogous to gpu.module.

    All quantum kernels live inside a jasp.module, separating QPU code from
    classical host code at the IR level.  A downstream pass can identify all
    QPU work by iterating the body of the jasp.module rather than scanning
    the whole builtin.module.
    """

    name = "jasp.module"

    body = region_def()
    sym_name = prop_def(StringAttr)

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())

    def __init__(self, module_name: str, region: Region):
        super().__init__(
            properties={"sym_name": StringAttr(module_name)},
            regions=[region],
        )

    @classmethod
    def parse(cls, parser: Parser) -> "JaspModuleOp":
        sym_name = parser.parse_symbol_name()  # returns StringAttr
        region = parser.parse_region()
        return cls(module_name=sym_name.data, region=region)

    def print(self, printer: Printer) -> None:
        printer.print_string(f" @{self.sym_name.data}")
        printer.print_region(self.body)


@irdl_op_definition
class QuantumKernelOp(IRDLOperation):
    """A self-contained quantum function, analogous to func.func.

    The *declared* function type is classical-only — QuantumState does not
    appear in the external signature.  Inside the body region the entry block
    receives the classical parameters followed by a trailing
    ``!jasp.QuantumState`` argument, and ``jasp.return`` yields the classical
    results followed by a trailing ``!jasp.QuantumState`` operand.

    This makes quantum functions unambiguously identifiable in the IR without
    scanning the function body for sentinel ops.
    """

    name = "jasp.quantum_kernel"

    body = region_def()
    sym_name = prop_def(StringAttr)
    function_type = prop_def(FunctionType)  # classical types only — no QuantumState
    sym_visibility = opt_prop_def(StringAttr)

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface())

    def __init__(
        self,
        kernel_name: str,
        function_type: FunctionType,
        region: Region,
        visibility: str | StringAttr | None = None,
    ):
        if isinstance(visibility, str):
            visibility = StringAttr(visibility)
        properties = {
            "sym_name": StringAttr(kernel_name),
            "function_type": function_type,
            "sym_visibility": visibility,
        }
        super().__init__(properties=properties, regions=[region])

    def verify_(self) -> None:
        if not isinstance(self.parent_op(), JaspModuleOp):
            raise VerifyException(
                "jasp.quantum_kernel must be inside a jasp.module"
            )
        if len(self.body.blocks) == 0:
            return
        entry = self.body.blocks.first
        assert entry is not None
        arg_types = list(entry.arg_types)
        if not arg_types or not isinstance(arg_types[-1], QuantumStateType):
            raise VerifyException(
                "jasp.quantum_kernel: entry block must have !jasp.QuantumState "
                "as its last argument"
            )
        classical_args = tuple(arg_types[:-1])
        if classical_args != tuple(self.function_type.inputs):
            raise VerifyException(
                "jasp.quantum_kernel: entry block classical arguments must match "
                "the declared function_type inputs"
            )
        for op in entry.ops:
            if isinstance(op, JaspReturnOp):
                classical_results = tuple(v.type for v in op.values[:-1])
                if classical_results != tuple(self.function_type.outputs):
                    raise VerifyException(
                        "jasp.quantum_kernel: jasp.return classical types must match "
                        "declared function_type outputs"
                    )

    @classmethod
    def parse(cls, parser: Parser) -> "QuantumKernelOp":
        visibility = parser.parse_optional_visibility_keyword()
        (name, input_types, return_types, region, extra_attrs, _, _) = (
            parse_func_op_like(
                parser,
                reserved_attr_names=("sym_name", "function_type", "sym_visibility"),
            )
        )
        op = cls(
            kernel_name=name,
            function_type=FunctionType.from_lists(input_types, return_types),
            region=region,
            visibility=visibility,
        )
        if extra_attrs is not None:
            op.attributes |= extra_attrs.data
        return op

    def print(self, printer: Printer) -> None:
        if self.sym_visibility:
            printer.print_string(f" {self.sym_visibility.data}")
        # Collect non-reserved attributes to pass through.
        reserved = {"sym_name", "function_type", "sym_visibility"}
        extra_attrs = {k: v for k, v in self.attributes.items() if k not in reserved}
        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            extra_attrs,
            reserved_attr_names=list(reserved),
        )


@irdl_op_definition
class JaspReturnOp(IRDLOperation):
    """Terminator for jasp.quantum_kernel bodies.

    Operands: (!jasp.QuantumState, <classical results...>)

    The leading QuantumState is the final state of the quantum computation;
    it is consumed by the kernel boundary and not visible to callers.
    """

    name = "jasp.return"

    values = var_operand_def()

    traits = traits_def(IsTerminator())

    def __init__(self, values):
        super().__init__(operands=[values])

    assembly_format = "($values^ `:` type($values))? attr-dict"

    def verify_(self) -> None:
        if not self.values or not isinstance(self.values[-1].type, QuantumStateType):
            raise VerifyException(
                "jasp.return: last operand must be !jasp.QuantumState"
            )


@irdl_op_definition
class JaspCallOp(IRDLOperation):
    """Calls a jasp.quantum_kernel by symbol.

    From the caller's perspective the operation is purely classical — no
    QuantumState appears in operands or results.  The callee's internal
    QuantumState lifecycle is an implementation detail managed by the kernel
    boundary.
    """

    name = "jasp.call"

    callee = prop_def(FlatSymbolRefAttrConstr)
    arguments = var_operand_def()
    res = var_result_def()

    assembly_format = (
        "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $res)"
    )

    def __init__(
        self,
        callee: str | SymbolRefAttr | FlatSymbolRefAttr,
        arguments,
        return_types,
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(
            operands=[arguments],
            result_types=[return_types],
            properties={"callee": callee},
        )


# ---------------------------------------------------------------------------
# Dialect
# ---------------------------------------------------------------------------


class JaspDialect(Dialect):
    """A dialect for real-time hybrid quantum computation.

    This dialect provides simple data structures that enable expression and
    manipulation of quantum computations.
    """

    name = "jasp"
    operations = [
        CreateQubitsOp,
        GetQubitOp,
        GetSizeOp,
        SliceOp,
        FuseOp,
        ResetOp,
        MeasureOp,
        DeleteQubitsOp,
        CreateQuantumKernelOp,
        ConsumeQuantumKernelOp,
        QuantumGateOp,
        ParityOp,
        JaspModuleOp,
        QuantumKernelOp,
        JaspReturnOp,
        JaspCallOp,
    ]
    attributes = [QuantumStateType, QubitType, QubitArrayType]
