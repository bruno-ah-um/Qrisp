# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Defines DepthMetric, a profiling metric that computes the circuit depth of a Jaspr."""

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal
from jax.typing import ArrayLike

from qrisp._cache_config import qrisp_lru_compilation_cache
from qrisp.circuit.instruction import Instruction
from qrisp.jasp.interpreter_tools.interpreters.profiling_interpreter import (
    BaseMetric,
    build_metric_profiler,
    normalize_slice_bounds,
)
from qrisp.jasp.interpreter_tools.interpreters.utilities import (
    is_abstract,
)
from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.primitives import (
    AbstractQubitArray,
)
from qrisp.jasp.primitives.operation_primitive import greek_letters


def _apply_duration_on_qubits(
    depth_array: jnp.ndarray,
    current_depth: jnp.ndarray,
    qubit_ids: list,
    duration: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a gate of given duration on the given qubits.

    The gate starts once all its qubits are free, and ends ``duration`` later.

    Parameters
    ----------
    depth_array : jnp.ndarray
        The time at which each qubit becomes free.
    current_depth : jnp.ndarray
        The depth of the circuit so far.
    qubit_ids : list
        The global ids of the qubits the gate acts on.
    duration : float
        The duration of the gate, cast to the dtype of ``depth_array``.

    Returns
    -------
    depth_array : jnp.ndarray
        The updated time at which each qubit becomes free.
    current_depth : jnp.ndarray
        The updated depth of the circuit.

    """
    qubit_ids_arr = jnp.asarray(qubit_ids, dtype=jnp.int64)
    touched = jnp.take(depth_array, qubit_ids_arr)
    start = jnp.max(touched)
    end = start + jnp.asarray(duration, dtype=depth_array.dtype)

    depth_array = depth_array.at[qubit_ids_arr].set(end)
    current_depth = jnp.maximum(current_depth, end)
    return depth_array, current_depth


def _iter_definition_ops(definition) -> Iterator[tuple[Instruction, list[Any]]]:
    """Iterate over operations in a quantum circuit definition.

    Parameters
    ----------
    definition : QuantumCircuit
        The definition of a gate.

    Yields
    ------
    tuple[Instruction, list[Any]]
        Each instruction of the definition with the qubits it acts on.

    Raises
    ------
    TypeError
        If an entry of the definition has no qubits.

    """
    for instruction in definition.data:
        qubits = getattr(instruction, "qubits", None)
        if qubits is None:
            raise TypeError(f"Unsupported definition.data entry type {type(instruction)}: {instruction}")
        yield instruction, list(qubits)


# This function creates a lookup table (mapping table) with length `table_size`
# (this must be a concrete integer because of JAX and <= MAX_QUBITS)
# that maps logical indices to global qubit ids in the depth array.
def _create_lookup_table(idx_start: jax.Array | int, table_size: ArrayLike) -> jax.Array:
    """Create a lookup table for qubit IDs starting from idx_start.

    Parameters
    ----------
    idx_start : jax.Array or int
        The global id of the first qubit.
    table_size : ArrayLike
        The number of entries, which must be concrete.

    Returns
    -------
    jax.Array
        The global ids ``idx_start, idx_start + 1, ...``.

    """
    return idx_start + jnp.arange(table_size, dtype=jnp.int64)


def _as_qubit_array_handle(x) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a QubitArray handle or a single qubit id to a ``(table, size)`` handle.

    Parameters
    ----------
    x : tuple or ArrayLike
        A QubitArray handle ``(table, size)``, or the global id of a single qubit.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        The lookup table and the size of the qubit array.

    """
    if isinstance(x, tuple):
        table, size = x
        return jnp.asarray(table, dtype=jnp.int64), jnp.asarray(size, dtype=jnp.int64)

    return jnp.asarray([x], dtype=jnp.int64), jnp.asarray(jnp.int64(1))


def _fuse_qubit_tables(handle_a: tuple, handle_b: tuple, max_qubits: int) -> tuple[jax.Array, jax.Array]:
    """Concatenate the lookup tables of two qubit arrays.

    Parameters
    ----------
    handle_a : tuple
        The ``(table, size)`` handle of the first qubit array.
    handle_b : tuple
        The ``(table, size)`` handle of the second qubit array.
    max_qubits : int
        The maximum number of qubits supported, which bounds the fused size.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The ``(table, size)`` handle of the fused qubit array.

    """
    arr_a, size_a = handle_a
    arr_b, size_b = handle_b

    size_out = jnp.minimum(size_a + size_b, jnp.int64(max_qubits))

    indices_size = size_out if not is_abstract(size_out) else max_qubits
    idxs = jnp.arange(indices_size, dtype=jnp.int64)

    # Entries before size_a come from A, the following ones from B
    a_vals = arr_a[jnp.clip(idxs, 0, jnp.maximum(size_a - 1, 0))]
    b_vals = arr_b[jnp.clip(idxs - size_a, 0, jnp.maximum(size_b - 1, 0))]

    return jnp.where(idxs < size_a, a_vals, b_vals), size_out


def _gate_layers(op, qubits: list) -> Iterator[list]:
    """Yield the qubit ids of each operation that a gate occupies in time.

    A gate without a definition occupies its qubits once. A gate with a definition
    occupies them as the operations of its transpiled definition do.

    Parameters
    ----------
    op : Operation
        The applied gate.
    qubits : list
        The global ids of the qubits the gate acts on.

    Yields
    ------
    list
        The global ids of the qubits of each operation.

    """
    if not getattr(op, "definition", None):
        yield op, qubits
        return

    transpiled_definition = op.definition.transpile()

    # The definition refers to its own Qubit objects: map them to the global
    # ids (integer tracers/array scalars) of the qubits the gate acts on.
    qubit_map = dict(zip(transpiled_definition.qubits, qubits, strict=True))

    for instruction, inner_def_qubits in _iter_definition_ops(transpiled_definition):
        if inner_def_qubits:
            yield instruction.op, [qubit_map[qb] for qb in inner_def_qubits]


_UNKNOWN = object()

# Primitives calling a sub-jaxpr with their arguments, whose outputs are the outputs of the sub-jaxpr
_CALL_PRIMITIVES = {"jit", "pjit", "closed_call"}


def _as_static_scalar(value):
    """Return ``value`` as a hashable scalar, or ``_UNKNOWN`` if it is not a static scalar."""
    if value is _UNKNOWN or isinstance(value, jax.core.Tracer) or np.ndim(value) != 0:
        return _UNKNOWN
    return np.asarray(value)[()]


def _fold_eqn(eqn: JaxprEqn, invals: list) -> list:
    """Evaluate a classical equation whose inputs are all static, or return unknown outputs."""
    unknown = [_UNKNOWN] * len(eqn.outvars)
    # Effectful equations (e.g. debug prints and callbacks) must not run during the analysis
    if eqn.primitive.name.startswith("jasp.") or eqn.effects or any(val is _UNKNOWN for val in invals):
        return unknown
    try:
        with jax.ensure_compile_time_eval():
            outvals = eqn.primitive.bind(*invals, **eqn.params)
    except Exception:  # noqa: BLE001
        return unknown
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
    return [_as_static_scalar(val) for val in outvals]


def _record_gate_params(eqn: JaxprEqn, invals: list, gate_params: dict) -> None:
    """Record the values of the parameters of a gate, or ``_UNKNOWN`` if they differ from a previous call."""
    op = eqn.params["gate"]
    # Invars are the qubits, then the parameters, then the QuantumState
    for var, val in zip(eqn.invars[op.num_qubits : -1], invals[op.num_qubits : -1], strict=True):
        if not isinstance(var, Literal):
            previous = gate_params.get(var, val)
            gate_params[var] = val if previous is not _UNKNOWN and previous == val else _UNKNOWN


def _fold_unknown_sub_jaxprs(eqn: JaxprEqn, gate_params: dict, memo: dict) -> bool:
    """Record the gate parameters of the sub-jaxprs of an equation, assuming unknown inputs.

    Returns
    -------
    bool
        Whether the equation has sub-jaxprs.

    """
    sub_jaxprs = [
        sub
        for param in eqn.params.values()
        for sub in (param if isinstance(param, (tuple, list)) else (param,))
        if isinstance(sub, (Jaxpr, ClosedJaxpr))
    ]
    for sub in sub_jaxprs:
        num_invars = len((sub.jaxpr if isinstance(sub, ClosedJaxpr) else sub).invars)
        _fold_jaxpr(sub, [_UNKNOWN] * num_invars, gate_params, memo)
    return bool(sub_jaxprs)


def _fold_jaxpr(jaxpr, invals: list, gate_params: dict, memo: dict) -> list:
    """Propagate static scalars through a jaxpr, recording the values of gate parameters.

    Equations whose inputs are all static are evaluated, and static values are passed on
    to the sub-jaxprs of calls, the branches of conditionals and the constants of loop
    bodies. Since a sub-jaxpr can be called from several places, a gate parameter is
    recorded as static only if it has the same value at every call site.

    Returns
    -------
    list
        The static values of the outputs of the jaxpr, or ``_UNKNOWN``.

    """
    consts = []
    if isinstance(jaxpr, ClosedJaxpr):
        consts = [_as_static_scalar(const) for const in jaxpr.consts]
        jaxpr = jaxpr.jaxpr

    key = (id(jaxpr), tuple(invals))
    if key in memo:
        return memo[key]

    env = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, invals, strict=True))

    def read(var):
        return _as_static_scalar(var.val) if isinstance(var, Literal) else env.get(var, _UNKNOWN)

    for eqn in jaxpr.eqns:
        eqn_invals = [read(var) for var in eqn.invars]
        name = eqn.primitive.name
        outvals = [_UNKNOWN] * len(eqn.outvars)

        if name == "jasp.quantum_gate":
            _record_gate_params(eqn, eqn_invals, gate_params)
        elif name in _CALL_PRIMITIVES:
            outvals = _fold_jaxpr(eqn.params["jaxpr"], eqn_invals, gate_params, memo)
        elif name == "cond":
            # The first operand selects the branch, the others are passed to the branch
            for branch in eqn.params["branches"]:
                _fold_jaxpr(branch, eqn_invals[1:], gate_params, memo)
        elif name == "while":
            # Only the constants of the body are the same in every iteration
            cond_nconsts, body_nconsts = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
            body_consts = eqn_invals[cond_nconsts : cond_nconsts + body_nconsts]
            num_carries = len(eqn_invals) - cond_nconsts - body_nconsts
            _fold_jaxpr(eqn.params["body_jaxpr"], body_consts + [_UNKNOWN] * num_carries, gate_params, memo)
        elif not _fold_unknown_sub_jaxprs(eqn, gate_params, memo):
            outvals = _fold_eqn(eqn, eqn_invals)

        env.update(zip(eqn.outvars, outvals, strict=True))

    memo[key] = [read(var) for var in jaxpr.outvars]
    return memo[key]


@qrisp_lru_compilation_cache
def static_gate_params(jaspr: Jaspr) -> frozenset:
    """Return the gate parameters of a Jaspr whose values are known at compile time.

    Besides literals, these are parameters computed from constants during tracing,
    such as ``p(jnp.pi / 4, qv)``, also across calls and conditionals. Parameters that
    depend on the arguments of the Jaspr or on a loop iteration are not static.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr whose gate parameters are inspected.

    Returns
    -------
    frozenset[tuple[Var, float]]
        The static (non-literal) parameter variables of the gate equations with their values.

    """
    gate_params = {}
    _fold_jaxpr(jaspr, [_UNKNOWN] * len(jaspr.jaxpr.invars), gate_params, {})
    return frozenset((var, float(val)) for var, val in gate_params.items() if val is not _UNKNOWN)


def _bind_static_params(eqn: JaxprEqn, static_params: dict):
    """Return the gate of a ``quantum_gate`` equation with its parameters bound, if they are static.

    Parameters that are literals of the Jaspr (e.g. ``rx(0.3, qv)``), or that are
    computed from constants (see :func:`static_gate_params`), are known at compile
    time, so the gate can be bound to their values, as the concrete evaluation of the
    primitive does. If any parameter is traced, the abstract gate is returned.

    Parameters
    ----------
    eqn : JaxprEqn
        A ``jasp.quantum_gate`` equation.
    static_params : dict
        The values of the static non-literal gate parameters, by variable.

    Returns
    -------
    tuple[Operation, bool]
        The (possibly bound) gate, and whether all its parameters are static.

    """
    op = eqn.params["gate"]
    # Invars are the qubits, then the parameters, then the QuantumState
    param_vars = eqn.invars[op.num_qubits : -1]

    values = []
    for var in param_vars:
        if isinstance(var, Literal):
            values.append(float(var.val))
        elif var in static_params:
            values.append(static_params[var])
        else:
            return op, False

    if not values:
        return op, True
    return op.bind_parameters({greek_letters[i]: value for i, value in enumerate(values)}), True


# The computation will fail anyway if this is triggered,
# but this way we can print an informative message before the failure happens.
def _warn_overflow(idx_end, max_qubits):
    """Print an error message when the depth metric computation overflows.

    Parameters
    ----------
    idx_end : ArrayLike
        The first free global id after the allocation that overflowed.
    max_qubits : int
        The maximum number of qubits supported.

    """
    jax.debug.print(
        (
            "ERROR: Depth metric computation overflowed: tried to create qubits with "
            "global indices up to {idx_end}, but the maximum supported is {max_qubits}. "
            "Consider increasing the `max_qubits` parameter for depth profiling."
        ),
        idx_end=idx_end,
        max_qubits=max_qubits,
    )


class DepthMetric(BaseMetric):
    """A metric implementation that computes the circuit depth of a Jaspr.

    Parameters
    ----------
    meas_behavior : Callable
        The measurement behavior function.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation. Default is 1024.

    depth_indicator : Callable, optional
        A function mapping a gate to its depth contribution, e.g. ``t_depth_indicator``.
        Non-integer values are accumulated as floats and the final depth is truncated,
        as in ``QuantumCircuit.depth``. ``None`` (default) assigns every gate a depth of 1.

    static_params : frozenset, optional
        The values of the gate parameters known at compile time, as returned by
        :func:`static_gate_params`. Gates are bound to them before calling
        ``depth_indicator``. Literal parameters are always bound.

    """

    def __init__(
        self,
        meas_behavior: Callable,
        max_qubits: int = 1024,
        depth_indicator: Callable | None = None,
        static_params: frozenset = frozenset(),
    ):
        """Initialize the DepthMetric."""
        super().__init__(meas_behavior=meas_behavior)

        # Define a maximum number of qubits to track depth for
        # This can be adjusted as needed (trade-off between memory and flexibility).
        # Unfortunately, JAX does not support dynamic arrays (yet).
        # Ideally, we would implement dynamic resizing in the future
        self._max_qubits: int = max_qubits
        self._depth_indicator = depth_indicator
        self._static_params = static_params
        self._static_param_values = dict(static_params)

    @property
    def max_qubits(self) -> int:
        """Return the maximum number of qubits supported."""
        return self._max_qubits

    @property
    def depth_indicator(self) -> Callable | None:
        """Return the depth indicator, or None for unit depth."""
        return self._depth_indicator

    @classmethod
    def from_cache_key(cls, cache_key) -> "DepthMetric":
        """Rebuild the metric from the key returned by :meth:`cache_key`."""
        return cls(*cache_key)

    def cache_key(self) -> tuple[Callable, int, Callable | None, frozenset]:
        """Return the measurement behavior, maximum number of qubits, indicator and static parameters."""
        return (self.meas_behavior, self.max_qubits, self.depth_indicator, self._static_params)

    @property
    def depth_dtype(self) -> type:
        """Return the dtype of the depths: float for indicators with fractional durations, int otherwise."""
        return jnp.int64 if self.depth_indicator is None else jnp.float64

    def _gate_duration(self, op) -> float:
        """Return the depth contribution of a single (inner) gate."""
        if self.depth_indicator is None:
            return 1
        return float(self.depth_indicator(op))

    def initial_metric(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return a circuit of depth zero with no qubits allocated."""
        # To speed up compilation time, it will be probably necessary to
        # use the same incrementation constants trick used for gate counting.

        # Here is the explanation of the metric data structure:
        #
        # - depth_array: jnp.ndarray of shape (max_qubits,) that keeps track
        #   of the depth of each qubit. Its dtype is `depth_dtype`: float when a
        #   depth indicator may return fractional durations, int otherwise.
        #
        # - current_depth: scalar of dtype `depth_dtype` that keeps track of the
        #   current maximum depth of the circuit.
        #
        # - previous_size: integer scalar that keeps track of the
        #   next available index in the depth_array for newly created qubits.
        #   When we create new qubits, we assign them global indices starting from this value,
        #   and then we update it by the number of qubits created.
        #
        # - invalid: boolean scalar that indicates whether the depth computation
        #   has overflowed the maximum number of qubits supported.
        #
        # - meas_count: integer scalar with the number of qubits measured so far,
        #   used to number the measurements (see BaseMetric._sample_measurement).
        depth_array = jnp.zeros(self.max_qubits, dtype=self.depth_dtype)
        current_depth = jnp.zeros((), dtype=self.depth_dtype)
        previous_size = jnp.int64(0)
        invalid = jnp.bool_(False)
        meas_count = jnp.int64(0)

        return depth_array, current_depth, previous_size, invalid, meas_count

    ##############################################################
    ### Quantum primitive handlers
    ##############################################################

    def _reserve_qubits(self, previous_size: jax.Array, size: ArrayLike, invalid: jax.Array) -> tuple:
        """Reserve global ids for ``size`` new qubits and flag an overflow.

        Parameters
        ----------
        previous_size : jax.Array
            The first free global id.
        size : ArrayLike
            The number of new qubits.
        invalid : jax.Array
            Whether the computation has already overflowed.

        Returns
        -------
        idx_start : jax.Array
            The global id of the first new qubit.
        idx_end : jax.Array
            The first free global id after the new qubits.
        invalid : jax.Array
            Whether the computation has overflowed ``max_qubits``.

        """
        idx_start = previous_size
        idx_end = idx_start + size

        invalid = jnp.logical_or(invalid, idx_end > jnp.int64(self.max_qubits))
        jax.lax.cond(invalid, _warn_overflow, lambda *_: None, idx_end, self.max_qubits)

        return idx_start, idx_end, invalid

    def handle_create_qubits(self, invalues, eqn, context_dic):
        """Assign global ids to the new qubits, flagging an overflow of ``max_qubits``."""
        size, metric_data = invalues
        depth_array, global_depth, previous_size, invalid, meas_count = metric_data

        idx_start, previous_size, invalid = self._reserve_qubits(previous_size, size, invalid)

        table_size = size if not is_abstract(size) else self.max_qubits
        qubit_table_handle = (_create_lookup_table(idx_start, table_size), size)
        metric_data = (depth_array, global_depth, previous_size, invalid, meas_count)

        # Associate the following in context_dic:
        # QubitArray -> qubit_table_handle (qubit_ids_table, size)
        # QuantumState -> metric_data
        return qubit_table_handle, metric_data

    def handle_get_qubit(self, invalues, eqn, context_dic):
        """Look up the global id of the qubit, counting negative indices from the end."""
        qubit_table_handle, qubit_index = invalues

        qubit_ids_table, size = qubit_table_handle

        # Negative indices count from the end of the qubit array. This has to be
        # done here: for a dynamic size the lookup table has max_qubits entries,
        # so indexing it with a negative index would pick the wrong qubit.
        qubit_index = qubit_index + (qubit_index < 0) * size

        # Associate the following in context_dic:
        # Qubit -> qubit_id (integer tracer/array scalar)
        return qubit_ids_table[qubit_index]

    def handle_get_size(self, invalues, eqn, context_dic):
        """Return the size stored in the QubitArray handle."""
        (qubit_table_handle,) = invalues

        _, size = qubit_table_handle

        # Associate the following in context_dic:
        # size -> size_value (integer tracer/scalar)
        return size

    def handle_quantum_gate(self, invalues, eqn, context_dic):
        """Advance the depth of the qubits the gate acts on, gate by gate of its definition."""
        depth_array, current_depth, previous_size, invalid, meas_count = invalues[-1]

        # Indicators may depend on the gate parameters: bind them if they are static
        op = (
            _bind_static_params(eqn, self._static_param_values)[0]
            if self.depth_indicator is not None
            else eqn.params["gate"]
        )
        # Qubits have been already converted to their integer (possibly traced)
        # indices by the `jasp.get_qubit` primitive handler
        qubits = list(invalues[: op.num_qubits])

        for inner_op, qubit_ids in _gate_layers(op, qubits):
            duration = self._gate_duration(inner_op)
            depth_array, current_depth = _apply_duration_on_qubits(
                depth_array=depth_array,
                current_depth=current_depth,
                qubit_ids=qubit_ids,
                duration=duration,
            )

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return depth_array, current_depth, previous_size, invalid, meas_count

    def handle_measure(self, invalues, eqn, context_dic):
        """Sample the measurement outcome and count the measured qubits."""
        target, metric_data = invalues
        *depth_data, meas_count = metric_data

        num_measured = target[1] if isinstance(eqn.invars[0].aval, AbstractQubitArray) else 1
        meas_res = self._sample_measurement(eqn, num_measured, meas_count)

        # Associate the following in context_dic:
        # meas_result -> meas_res
        # QuantumState -> metric_data
        return meas_res, (*depth_data, meas_count + num_measured)

    def handle_fuse(self, invalues, eqn, context_dic):
        """Concatenate the lookup tables of the operands, each a QubitArray or a single qubit."""
        handle_a, handle_b = (_as_qubit_array_handle(value) for value in invalues)

        # Associate the following in context_dic:
        # QubitArray -> (qubit_array_out, size_out)
        return _fuse_qubit_tables(handle_a, handle_b, self.max_qubits)

    def handle_slice(self, invalues, eqn, context_dic):
        """Select the entries of the lookup table in the slice, following Python semantics."""
        (qubit_array, size), start, stop = invalues

        start, stop = normalize_slice_bounds(size, start, stop)
        size_out = stop - start

        table_size = size_out if not is_abstract(size_out) else self.max_qubits

        array_idx_mapping = _create_lookup_table(start, table_size)
        qubit_array_out = qubit_array[array_idx_mapping]

        # Associate the following in context_dic:
        # QubitArray -> (qubit_array_out, size_out)
        return qubit_array_out, size_out

    def handle_reset(self, invalues, eqn, context_dic):
        """Leave the depth unchanged: resets are currently ignored."""
        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return metric_data

    def handle_delete_qubits(self, invalues, eqn, context_dic):
        """Leave the depth unchanged: deleted qubits keep their ids and still count toward ``max_qubits``."""
        _, metric_data = invalues

        # Associate the following in context_dic:
        # QuantumState -> metric_data
        return metric_data


def extract_depth(res: tuple, jaspr: Jaspr, _) -> int:
    """Extract depth from the profiling result.

    Parameters
    ----------
    res : tuple
        The output of the profiler: the Jaspr's return values followed by the metric data.
    jaspr : Jaspr
        The profiled Jaspr.
    _ : None
        The auxiliary data of the profiler, unused by this metric.

    Returns
    -------
    int
        The depth of the circuit.

    Raises
    ------
    ValueError
        If the computation needed more than ``max_qubits`` qubits.

    """
    metric = res[-1] if len(jaspr.outvars) > 1 else res
    _, depth, _, overflowed, _ = metric

    if overflowed:
        raise ValueError("The depth metric computation overflowed the maximum number of qubits supported.")

    # Truncate fractional depths, as QuantumCircuit.depth does
    return int(depth)


# LRU cache controlled by QRISP_COMPILATION_CACHE_SIZE env var
@qrisp_lru_compilation_cache
def get_depth_profiler(
    jaspr: Jaspr,
    meas_behavior: Callable,
    max_qubits: int = 1024,
    callback_threshold: int | None = None,
    depth_indicator: Callable | None = None,
) -> tuple[Callable, None]:
    """Build a depth profiling computer for a given Jaspr.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr expression to profile.

    meas_behavior : Callable
        The measurement behavior function.

    max_qubits : int, optional
        The maximum number of qubits supported for depth computation. Default is 1024.

    callback_threshold : int | None, optional
        Minimum value of ``call_count * inlined_eqn_count`` required to
        trigger ``jax.pure_callback`` wrapping.  ``None`` (default)
        disables callbacks entirely (fastest execution).  ``0`` wraps
        every reused sub-jaxpr (fastest compilation).

    depth_indicator : Callable, optional
        See :class:`DepthMetric`. ``None`` (default) gives the plain circuit depth.

    Returns
    -------
    tuple[Callable, None]
        A depth profiler function and None as auxiliary data.

    """
    # Only indicators look at the gate parameters
    static_params = static_gate_params(jaspr) if depth_indicator is not None else frozenset()
    depth_metric = DepthMetric(meas_behavior, max_qubits, depth_indicator, static_params)

    return build_metric_profiler(jaspr, depth_metric, callback_threshold), None


def simulate_depth(jaspr: Jaspr, *_, **__) -> int:
    """Simulate depth metric via actual simulation.

    Parameters
    ----------
    jaspr : Jaspr
        The Jaspr to simulate.
    *_ : Any
        The arguments of the Jaspr (unused).
    **__ : Any
        Keyword arguments of the simulation (unused).

    Raises
    ------
    NotImplementedError
        Always, since simulation-based depth computation is not implemented yet.

    """
    raise NotImplementedError("Depth metric via simulation is not implemented yet.")
