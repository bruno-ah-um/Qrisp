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

"""Tests the T-depth profiling decorator for Jasp programs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qrisp import (
    QFT,
    QuantumCircuit,
    QuantumFloat,
    control,
    cp,
    cx,
    cz,
    gphase,
    h,
    invert,
    mcp,
    mcx,
    measure,
    p,
    reset,
    rx,
    ry,
    rz,
    s,
    s_dg,
    sx,
    sx_dg,
    t,
    t_depth,
    t_dg,
    u3,
    x,
    y,
    z,
)
from qrisp.jasp import jrange, make_jaspr, qache
from qrisp.jasp.evaluation_tools import profile_jaspr
from qrisp.jasp.interpreter_tools.interpreters.depth_metric import static_gate_params
from qrisp.misc import t_depth_indicator

# T-depth assigned to rotations that are neither Clifford nor T gates
ROTATION_COST = 15


def rotation_indicator(op):
    """Return the T-depth of ``op``, assigning ``ROTATION_COST`` to rotations that are neither Clifford nor T."""
    try:
        return t_depth_indicator(op)
    except ValueError:
        return ROTATION_COST


class TestTDepthGates:
    """Test the T-depth of individual gates and simple gate sequences."""

    def test_t_gate(self):
        """A T gate has a T-depth of 1."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            t(qf[0])
            return measure(qf[0])

        assert main() == 1

    def test_t_dg_gate(self):
        """A T-dagger gate has a T-depth of 1."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            t_dg(qf[0])
            return measure(qf[0])

        assert main() == 1

    def test_clifford_gates_are_free(self):
        """Clifford gates do not contribute to the T-depth."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(2)
            h(qf[0])
            x(qf[1])
            cx(qf[0], qf[1])
            return measure(qf[0])

        assert main() == 0

    @pytest.mark.parametrize(
        "qubit_indices,expected_depth",
        [
            ([0, 1], 1),  # parallel
            ([0, 0], 2),  # sequential
        ],
    )
    def test_two_t_gates(self, qubit_indices, expected_depth):
        """T gates on different qubits run in parallel, on the same qubit in sequence."""

        @t_depth(meas_behavior="0")
        def main(qubit_indices):
            qf = QuantumFloat(2)
            t(qf[qubit_indices[0]])
            t(qf[qubit_indices[1]])
            return measure(qf[0])

        assert main(qubit_indices) == expected_depth

    def test_cnot_propagates_dependencies(self):
        """A Clifford gate is free but still synchronizes the qubits it acts on."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(2)
            t(qf[0])
            cx(qf[0], qf[1])
            t(qf[1])
            return measure(qf[1])

        assert main() == 2

    @pytest.mark.parametrize("gate", [rx, ry, rz, p], ids=["rx", "ry", "rz", "p"])
    @pytest.mark.parametrize(
        "angle, expected_depth",
        [
            (0, 0),
            (np.pi / 2, 0),
            (np.pi, 0),
            (-np.pi / 2, 0),
            (np.pi / 4, 1),
            (3 * np.pi / 4, 1),
            (-np.pi / 4, 1),
            (7 * np.pi / 4, 1),
            (9 * np.pi / 4, 1),
            (2 * np.pi * 3 / 16, ROTATION_COST),
            (0.3, ROTATION_COST),
        ],
    )
    def test_static_rotation_angle(self, gate, angle, expected_depth):
        """Static angles are visible to the indicator: Clifford and T angles are not costed as other rotations."""

        @t_depth(meas_behavior="0", depth_indicator=rotation_indicator)
        def main():
            qf = QuantumFloat(1)
            gate(angle, qf[0])
            return measure(qf[0])

        qc = QuantumCircuit(1)
        getattr(qc, gate.__name__)(angle, 0)

        assert main() == qc.t_depth(depth_indicator=rotation_indicator) == expected_depth

    @pytest.mark.parametrize(
        "angles",
        [
            (np.pi / 4, 0, 0),
            (0, np.pi / 4, 0),
            (0, 0, np.pi / 4),
            (np.pi / 2, np.pi, np.pi / 4),
            (0.3, 0.2, 0.1),
        ],
    )
    def test_u3_matches_circuit(self, angles):
        """The cost of a U3 gate agrees with ``QuantumCircuit.t_depth``."""

        @t_depth(meas_behavior="0", depth_indicator=rotation_indicator)
        def main():
            qf = QuantumFloat(1)
            u3(*angles, qf[0])
            return measure(qf[0])

        qc = QuantumCircuit(1)
        qc.u3(*angles, 0)

        assert main() == qc.t_depth(depth_indicator=rotation_indicator)

    @pytest.mark.parametrize("gate", [s, s_dg, sx, sx_dg, y, z], ids=["s", "s_dg", "sx", "sx_dg", "y", "z"])
    def test_single_qubit_clifford_gates(self, gate):
        """Single-qubit Clifford gates do not contribute to the T-depth."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            gate(qf[0])
            return measure(qf[0])

        qc = QuantumCircuit(1)
        getattr(qc, gate.__name__)(0)

        assert main() == qc.t_depth() == 0

    def test_cz_reset_and_gphase_are_free(self):
        """CZ, reset and global phases do not contribute to the T-depth."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(2)
            cz(qf[0], qf[1])
            reset(qf[0])
            gphase(0.3, qf[0])
            return measure(qf[0])

        assert main() == 0

    def test_runtime_angle(self):
        """Runtime angles are never classified as Clifford or T angles."""

        @t_depth(meas_behavior="0", depth_indicator=rotation_indicator)
        def main(phi):
            qf = QuantumFloat(1)
            rx(phi, qf[0])
            return measure(qf[0])

        assert main(np.pi / 4) == ROTATION_COST

    @pytest.mark.parametrize("rotation_cost", [2.5, 3 * np.log2(1e3)])
    @pytest.mark.parametrize("num_rotations", [1, 10, 100])
    def test_fractional_cost_matches_circuit(self, rotation_cost, num_rotations):
        """Fractional costs are summed and truncated once, as in ``QuantumCircuit.t_depth``."""

        def indicator(op):
            try:
                return t_depth_indicator(op)
            except ValueError:
                return rotation_cost

        @t_depth(meas_behavior="0", depth_indicator=indicator)
        def main():
            qf = QuantumFloat(1)
            for _ in range(num_rotations):
                rx(2 * np.pi * 3 / 16, qf[0])
            return measure(qf[0])

        qc = QuantumCircuit(1)
        for _ in range(num_rotations):
            qc.rx(2 * np.pi * 3 / 16, 0)

        assert main() == qc.t_depth(depth_indicator=indicator) == int(num_rotations * rotation_cost)


class TestTDepthControlFlow:
    """Test the T-depth with measurements, classical control and loops."""

    @pytest.mark.parametrize("meas_behavior,expected_depth", [("0", 0), ("1", 2)])
    def test_measurement_branches(self, meas_behavior, expected_depth):
        """The branch that is taken depends on the measurement behavior."""

        @t_depth(meas_behavior=meas_behavior)
        def main():
            qf = QuantumFloat(2)
            m = measure(qf[0])

            with control(m == 0):
                h(qf[0])
                x(qf[1])

            with control(m == 1):
                t(qf[1])
                t(qf[1])

        assert main() == expected_depth

    def test_jrange(self):
        """T gates inside a dynamic loop accumulate."""

        @t_depth(meas_behavior="0")
        def main(n):
            qf = QuantumFloat(1)
            for _ in jrange(n):
                t(qf[0])
            return measure(qf[0])

        assert main(5) == 5

    def test_qache(self):
        """T gates inside a cached subroutine are counted for every call."""

        @qache
        def subroutine(qf):
            t(qf[0])

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(2)
            subroutine(qf)
            subroutine(qf)
            return measure(qf[0])

        assert main() == 2


class TestTDepthConsistency:
    """Test that Jasp T-depth agrees with ``QuantumCircuit.t_depth``."""

    def test_docstring_circuit(self):
        """T gate, CNOT and another rotation: 1 + 15 = 16."""

        @t_depth(meas_behavior="0", depth_indicator=rotation_indicator)
        def main():
            qf = QuantumFloat(2)
            t(qf[0])
            cx(qf[0], qf[1])
            rx(2 * np.pi * 3 / 2**4, qf[1])
            return measure(qf[1])

        qc = QuantumCircuit(2)
        qc.t(0)
        qc.cx(0, 1)
        qc.rx(2 * np.pi * 3 / 2**4, 1)

        assert main() == qc.t_depth(depth_indicator=rotation_indicator) == 16

    def test_parallel_and_sequential(self):
        """A circuit mixing parallel and sequential T gates."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(3)
            t(qf[0])
            t(qf[1])
            cx(qf[0], qf[2])
            t_dg(qf[2])
            cx(qf[1], qf[2])
            t(qf[0])
            return measure(qf[0])

        qc = QuantumCircuit(3)
        qc.t(0)
        qc.t(1)
        qc.cx(0, 2)
        qc.t_dg(2)
        qc.cx(1, 2)
        qc.t(0)

        assert main() == qc.t_depth()

    def test_macro_gate(self):
        """Gates with a definition are transpiled, and the T-cost is taken per inner gate."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(3)
            mcx([qf[0], qf[1]], qf[2])
            return measure(qf[2])

        qc = QuantumCircuit(3)
        qc.mcx([0, 1], 2)

        assert main() == qc.t_depth()
        assert main() > 0


def _controlled_phase():
    qf = QuantumFloat(2)
    cp(np.pi / 8, qf[0], qf[1])
    return measure(qf[0])


def _controlled_t_phase():
    qf = QuantumFloat(2)
    cp(np.pi / 2, qf[0], qf[1])
    return measure(qf[0])


def _quantum_controlled_rotation():
    qf = QuantumFloat(2)
    with control(qf[0]):
        rx(np.pi / 8, qf[1])
    return measure(qf[0])


def _inverted_block():
    qf = QuantumFloat(2)
    with invert():
        t(qf[0])
        rz(np.pi / 8, qf[1])
        cx(qf[0], qf[1])
        t(qf[1])
    return measure(qf[0])


def _mcx(num_controls, method):
    def program():
        qf = QuantumFloat(num_controls + 1)
        mcx([qf[i] for i in range(num_controls)], qf[num_controls], method=method)
        return measure(qf[0])

    return program


def _adder():
    a = QuantumFloat(3)
    b = QuantumFloat(3)
    a[:] = 3
    b += a
    return measure(b)


def _angles_with_different_precisions():
    qf = QuantumFloat(2)
    rx(np.pi / 4, qf[0])
    rx(2 * np.pi / 2**6, qf[1])
    rz(np.pi / 8, qf[0])
    return measure(qf[0])


def _non_dyadic_static_angle():
    qf = QuantumFloat(1)
    rx(0.3, qf[0])
    return measure(qf[0])


def _angle_computed_at_trace_time():
    qf = QuantumFloat(1)
    p(jnp.array(np.pi) / 4, qf[0])
    return measure(qf[0])


def _multi_controlled_phase():
    qf = QuantumFloat(3)
    mcp(np.pi / 8, qf)
    return measure(qf[0])


def _qft():
    qf = QuantumFloat(4)
    h(qf)
    QFT(qf)
    return measure(qf)


CIRCUIT_PROGRAMS = {
    "controlled_phase": _controlled_phase,
    "controlled_t_phase": _controlled_t_phase,
    "quantum_controlled_rotation": _quantum_controlled_rotation,
    "inverted_block": _inverted_block,
    **{f"mcx_{n}_{method}": _mcx(n, method) for n in [2, 3, 4] for method in ["gray", "auto"]},
    # gray_pt cannot be traced with more than three controls
    **{f"mcx_{n}_gray_pt": _mcx(n, "gray_pt") for n in [2, 3]},
    "adder": _adder,
    "angles_with_different_precisions": _angles_with_different_precisions,
    "non_dyadic_static_angle": _non_dyadic_static_angle,
    "angle_computed_at_trace_time": _angle_computed_at_trace_time,
    # The angles are passed to the branches of the measurement-based uncomputation
    "multi_controlled_phase": _multi_controlled_phase,
}


class TestTDepthMatchesCircuit:
    """Test that the T-depth of a program agrees with the T-depth of its circuit."""

    @staticmethod
    def _t_depth_or_error(compute):
        """Return the T-depth computed by ``compute``, or ``"ValueError"`` if it raises one."""
        try:
            return compute()
        except ValueError:
            return "ValueError"

    def _assert_matches_circuit(self, program, indicator):
        qc = make_jaspr(program)().to_qc()[-1]
        jasp_t_depth = self._t_depth_or_error(lambda: t_depth(meas_behavior="0", depth_indicator=indicator)(program)())
        circuit_t_depth = self._t_depth_or_error(lambda: qc.depth(depth_indicator=indicator or t_depth_indicator))
        assert jasp_t_depth == circuit_t_depth

    @pytest.mark.parametrize("indicator", [None, rotation_indicator], ids=["default_indicator", "custom_indicator"])
    @pytest.mark.parametrize("program", CIRCUIT_PROGRAMS.values(), ids=CIRCUIT_PROGRAMS.keys())
    def test_program(self, program, indicator):
        """Programs without measurement-dependent control flow have the T-depth of their circuit.

        With the default indicator, both raise for rotations that are neither Clifford nor T gates.
        """
        self._assert_matches_circuit(program, indicator)

    @pytest.mark.xfail(strict=True, reason="Angles depending on a loop iteration are costed as runtime angles")
    def test_qft(self):
        """The QFT angles depend on the loop counter, so they are not known at compile time."""
        self._assert_matches_circuit(_qft, rotation_indicator)

    def test_cached_function_called_with_different_angles(self):
        """An angle that differs between calls of a cached function is not classified as a T angle."""

        @qache
        def subroutine(qf, phi):
            p(phi, qf[0])

        @t_depth(meas_behavior="0", depth_indicator=rotation_indicator)
        def main():
            qf = QuantumFloat(1)
            subroutine(qf, np.pi / 4)
            subroutine(qf, 0.3)
            return measure(qf[0])

        assert main() == 2 * ROTATION_COST

    def test_cached_function_called_with_the_same_angle(self):
        """An angle that is the same in every call of a cached function is costed by its value."""

        @qache
        def subroutine(qf, phi):
            p(phi, qf[0])

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            subroutine(qf, np.pi / 4)
            subroutine(qf, np.pi / 4)
            return measure(qf[0])

        assert main() == 2


class TestTDepthIndicator:
    """Test the default T-depth indicator and custom indicators."""

    @pytest.mark.parametrize("angle", [0.3, 2 * np.pi * 3 / 16])
    def test_default_indicator_rejects_other_rotations(self, angle):
        """The default indicator raises for static rotations that are neither Clifford nor T gates."""

        @t_depth(meas_behavior="0")
        def main():
            qf = QuantumFloat(1)
            t(qf[0])
            rx(angle, qf[0])
            return measure(qf[0])

        with pytest.raises(ValueError, match="custom depth indicator"):
            main()

    def test_default_indicator_rejects_runtime_angles(self):
        """Runtime angles are not known to be Clifford or T, so the default indicator raises."""

        @t_depth(meas_behavior="0")
        def main(phi):
            qf = QuantumFloat(1)
            rx(phi, qf[0])
            return measure(qf[0])

        with pytest.raises(ValueError, match="custom depth indicator"):
            main(np.pi / 4)

    def test_docstring_example(self):
        """The custom indicator example of the docstring gives 11."""

        def my_t_depth_indicator(op):
            try:
                return t_depth_indicator(op)
            except ValueError:
                return 10

        @t_depth(meas_behavior="0", depth_indicator=my_t_depth_indicator)
        def circuit(phi):
            qv = QuantumFloat(2)
            t(qv[0])
            cx(qv[0], qv[1])
            rx(phi, qv[1])

        assert circuit(0.3) == 11

    def test_indicator_receives_bound_and_inner_gates(self):
        """The indicator gets static angles bound, and the gates of definitions instead of the macro-gate."""
        seen = []

        def recording_indicator(op):
            seen.append((op.name, tuple(op.params)))
            return 0

        @t_depth(meas_behavior="0", depth_indicator=recording_indicator)
        def main():
            qf = QuantumFloat(3)
            rx(0.3, qf[0])
            mcx([qf[0], qf[1]], qf[2])
            return measure(qf[2])

        assert main() == 0
        assert ("rx", (0.3,)) in seen
        assert "mcx" not in {name for name, _ in seen}
        assert {name for name, _ in seen} >= {"t", "t_dg", "cx", "h"}

    def test_indicator_is_part_of_the_cache_key(self):
        """Different indicators must not share a cached profiler."""

        def circuit():
            qf = QuantumFloat(1)
            t(qf[0])
            return measure(qf[0])

        assert t_depth(meas_behavior="0")(circuit)() == 1
        assert t_depth(meas_behavior="0", depth_indicator=lambda op: 7)(circuit)() == 7

    def test_jaspr_method(self):
        """``Jaspr.t_depth`` accepts a custom indicator."""

        def main(phi):
            qf = QuantumFloat(1)
            t(qf[0])
            rx(phi, qf[0])
            return measure(qf[0])

        jaspr = make_jaspr(main)(0.3)

        assert jaspr.t_depth(0.3, meas_behavior="0", depth_indicator=rotation_indicator) == 1 + ROTATION_COST


class TestTDepthArguments:
    """Test argument handling of the decorator."""

    def test_sim_not_supported(self):
        """Simulation is not available as a fallback for depth-like metrics."""

        @t_depth(meas_behavior="sim")
        def main():
            qf = QuantumFloat(1)
            t(qf[0])

        with pytest.raises(NotImplementedError):
            main()

    def test_max_qubits_overflow(self):
        """Exceeding ``max_qubits`` raises an error."""

        @t_depth(meas_behavior="0", max_qubits=2)
        def main():
            qf = QuantumFloat(4)
            t(qf[3])
            return measure(qf[0])

        with pytest.raises(ValueError, match="overflow"):
            main()

    def test_static_params_do_not_run_effects(self, capsys):
        """Effectful equations are not evaluated when looking for static parameters."""

        def main():
            qf = QuantumFloat(1)
            jax.debug.print("static value {}", jnp.array(1.0) / 4)
            p(jnp.array(np.pi) / 4, qf[0])
            return measure(qf[0])

        static_gate_params(make_jaspr(main)())

        assert "static value" not in capsys.readouterr().out

    @pytest.mark.parametrize("callback_threshold", [None, 0])
    def test_static_params_with_callbacks(self, callback_threshold):
        """Static parameters are kept when the metric is rebuilt for a callback."""

        @qache
        def subroutine(qf):
            p(jnp.array(np.pi) / 4, qf[0])

        def main():
            qf = QuantumFloat(1)
            for _ in range(3):
                subroutine(qf)
            return measure(qf[0])

        jaspr = make_jaspr(main)()
        profiler = profile_jaspr(
            jaspr,
            "depth",
            "0",
            depth_indicator=t_depth_indicator,
            callback_threshold=callback_threshold,
        )

        assert profiler() == 3

    def test_fractional_durations(self):
        """Fractional durations of an indicator are summed, then truncated."""

        def main():
            qf = QuantumFloat(1)
            t(qf[0])
            x(qf[0])
            return measure(qf[0])

        jaspr = make_jaspr(main)()

        # 2.5 + 2.5 = 5.0: fractional durations are summed, then truncated
        assert profile_jaspr(jaspr, "depth", "0", depth_indicator=lambda op: 2.5)() == 5
