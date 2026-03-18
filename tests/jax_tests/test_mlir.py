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

from qrisp import *
from qrisp.operators import a, c

def test_mlir_generation():
    
    
    # Test some features to make sure mlir generation works properly
    def inner(i):
        
        a = QuantumVariable(i)
        b = QuantumFloat(i)
        
        h(a)
        
        meas_res = measure(a)
        
        with control(meas_res == 0):
            for i in jrange(b.size):
                rz(1/i, b[i])
                cx(a[i], b[i])

        return a, b                
    
    def main(i):
        return expectation_value(inner, shots = i*10)(i)

    jaspr = make_jaspr(main)(2)
    xdsl_module = jaspr.to_mlir()

    # Test wheter stablehlo control flow is properly removed    
    
    from xdsl.printer import Printer
    
    def main():
        
        qv = QuantumVariable(2)
        h(qv[0])
        
        c = measure(qv[0])
        
        for i in jrange(qv.size):
            with control(c):
                x(qv[1])
        

    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()

    mlir_str = str(xdsl_module)
    
    assert "stablehlo.case" not in mlir_str
    assert "stablehlo.while" not in mlir_str
    assert "stablehlo.return" not in mlir_str
    
    # Test https://github.com/eclipse-qrisp/Qrisp/pull/296#issuecomment-3468979932
    
    H = a(0)
    orbital_amount = H.find_minimal_qubit_amount()
    U = qache(H.trotterization(forward_evolution=False))

    # Finding the gound state energy of the Water molecule with QPE
    def main():
        qv = QuantumFloat(orbital_amount)
        [x(qv[i]) for i in range(1)] # Prepare Hartree-Fock state, H2O molecule has 10 electrons

        qpe_res = QPE(qv,U,precision=1,kwargs={"steps":1})
        phi = measure(qpe_res)
        return phi

    jaspr = make_jaspr(main)()
    mlir = jaspr.to_mlir()

def test_mlir_basic_dialect_operations():
    """
    Test that basic JASP dialect operations are properly emitted in MLIR.
    This verifies the lowering rules for fundamental quantum operations.
    """
    from qrisp import QuantumVariable, h, cx, measure, x
    from qrisp.jasp import make_jaspr
    
    def main():
        # Test create_qubits
        qv = QuantumVariable(3)
        
        # Test quantum gates
        h(qv[0])
        cx(qv[0], qv[1])
        x(qv[2])
        
        # Test measurement
        result = measure(qv)
        
        return result
    
    # Create jaspr and convert to MLIR
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify that JASP dialect operations appear in the MLIR
    assert "jasp.create_qubits" in mlir_str, "create_qubits operation not found in MLIR"
    assert "jasp.measure" in mlir_str, "measure operation not found in MLIR"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "gate operations not found in MLIR"

def test_mlir_quantum_control_flow_rewriting():
    """
    Test that StableHLO control flow is properly rewritten to SCF for quantum types.
    Verifies the fix_quantum_control_flow function works correctly.
    """
    from qrisp import QuantumVariable, QuantumFloat, h, x, rz, cx, measure, control
    from qrisp.jasp import make_jaspr, jrange
    
    def main():
        qv = QuantumVariable(3)
        qf = QuantumFloat(2)
        
        h(qv[0])
        
        # Create measurement-based control flow
        meas_result = measure(qv[0])
        
        # Use control flow with quantum types
        with control(meas_result == 0):
            for i in jrange(qf.size):
                rz(1.0, qf[i])
                cx(qv[1], qf[i])
        
        # Additional control structure
        for j in jrange(qv.size):
            with control(meas_result):
                x(qv[j])
        
        return qv, qf
    
    # Create jaspr and convert to MLIR
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify that StableHLO control flow has been removed
    assert "stablehlo.case" not in mlir_str, "stablehlo.case should be rewritten to scf.if"
    assert "stablehlo.while" not in mlir_str, "stablehlo.while should be rewritten to scf.while"
    assert "stablehlo.return" not in mlir_str, "stablehlo.return should be rewritten to scf.yield"
    
    # Verify that SCF operations are present
    assert "scf.if" in mlir_str or "scf.while" in mlir_str or "scf.yield" in mlir_str, \
        "SCF control flow operations should be present"

def test_mlir_grovers_algorithm():
    """
    Test MLIR generation and execution for Grover's algorithm.
    Verifies that complex algorithms can be lowered to MLIR correctly.
    """
    from qrisp import QuantumFloat
    from qrisp.grover import tag_state, grovers_alg
    from qrisp.jasp import make_jaspr
    import numpy as np
    
    # Define oracle for Grover's algorithm (matching existing test pattern)
    def test_oracle(qf_list, phase=np.pi):
        tag_dic = {qf_list[0]: 0, qf_list[1]: 0.5}
        tag_state(tag_dic, phase=phase)
    
    def main():
        qf_list = [QuantumFloat(2, -2), QuantumFloat(2, -2)]
        grovers_alg(qf_list, test_oracle)
        return qf_list[0], qf_list[1]
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"

def test_mlir_qae_algorithm():
    """
    Test MLIR generation for Quantum Amplitude Estimation (QAE).
    Verifies that complex estimation algorithms can be lowered to MLIR correctly.
    """
    from qrisp import QuantumFloat, ry, z, QAE
    from qrisp.jasp import make_jaspr, terminal_sampling
    import numpy as np
    
    def state_function(qb):
        ry(np.pi / 4, qb)
    
    def oracle_function(qb):
        z(qb)
    
    def main():
        qb = QuantumFloat(1)
        res = QAE([qb], state_function, oracle_function, precision=3)
        return res
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected JASP dialect operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    assert "jasp.measure" in mlir_str, "Should contain measurement operations"
    
    # Verify algorithm produces correct result using terminal_sampling wrapper
    @terminal_sampling
    def main_sampling():
        qb = QuantumFloat(1)
        res = QAE([qb], state_function, oracle_function, precision=3)
        return res
    
    meas_res = main_sampling()
    assert np.round(meas_res[0.125], 2) == 0.5, f"Expected ~0.5 probability for 0.125"
    assert np.round(meas_res[0.875], 2) == 0.5, f"Expected ~0.5 probability for 0.875"

def test_mlir_iqpe():
    """
    Test MLIR generation for Iterative Quantum Phase Estimation (IQPE).
    Verifies that IQPE can be properly lowered to MLIR.
    """
    from qrisp import QuantumVariable, h, x, rx, IQPE
    from qrisp.jasp import make_jaspr
    import numpy as np
    
    def U(qv):
        x_val = 1/2**3
        y_val = 1/2**2
        rx(x_val * 2 * np.pi, qv[0])
        rx(y_val * 2 * np.pi, qv[1])
    
    def main():
        qv = QuantumVariable(2)
        x(qv)
        h(qv)
        return IQPE(qv, U, precision=4)
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    assert "jasp.measure" in mlir_str, "Should contain measurement operations"

def test_mlir_iqae():
    """
    Test MLIR generation for Iterative Quantum Amplitude Estimation (IQAE).
    Verifies that IQAE can be properly lowered to MLIR.
    """
    from qrisp import QuantumFloat, QuantumBool, control, h, ry, IQAE
    from qrisp.jasp import make_jaspr, jrange
    import numpy as np
    
    # State function for integration example
    def state_function(inp, tar):
        h(inp)  # Distribution
        
        N = 2**inp.size
        for k in jrange(inp.size):
            with control(inp[k]):
                ry(2**(k+1)/N, tar)
    
    def main():
        n = 4  # Smaller precision for faster testing
        inp = QuantumFloat(n, -n)
        tar = QuantumBool()
        input_list = [inp, tar]
        
        eps = 0.05
        alpha = 0.05
        
        return IQAE(input_list, state_function, eps=eps, alpha=alpha)
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    assert "jasp.measure" in mlir_str, "Should contain measurement operations"

def test_mlir_hamiltonian_simulation():
    """
    Test MLIR generation for Hamiltonian simulation.
    Verifies that Hamiltonian trotterization can be lowered to MLIR.
    """
    from qrisp import QuantumFloat, x, qache
    from qrisp.jasp import make_jaspr
    from qrisp.operators import a
    
    # Create a simple Hamiltonian
    H = a(0)
    orbital_amount = H.find_minimal_qubit_amount()
    U = qache(H.trotterization(forward_evolution=False))
    
    def main():
        qv = QuantumFloat(orbital_amount)
        # Prepare initial state
        x(qv[0])
        # Apply Hamiltonian evolution
        U(qv)
        return qv
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"

def test_mlir_qaoa():
    """
    Tests MLIR generation for the complete QAOA workflow as shown in the 
    'How to use QAOA in Jasp' documentation. This test verifies that the 
    entire QAOA optimization loop (including QAOAProblem setup, cost operator,
    mixer, and sample array post-processing) can be compiled to MLIR.
    """
    from qrisp import QuantumVariable, make_jaspr
    from qrisp.qaoa import QAOAProblem, RX_mixer, create_maxcut_cost_operator, create_maxcut_sample_array_post_processor
    import networkx as nx

    def main():
        # Create a random graph for the MaxCut problem
        G = nx.erdos_renyi_graph(6, 0.7, seed=133)

        # Create the sample array post-processor for Jasp (works with integer arrays)
        cl_cost = create_maxcut_sample_array_post_processor(G)

        # Create quantum argument
        qarg = QuantumVariable(G.number_of_nodes())

        # Set up the QAOA problem with cost operator, mixer, and classical cost function
        qaoa_maxcut = QAOAProblem(
            cost_operator=create_maxcut_cost_operator(G),
            mixer=RX_mixer,
            cl_cost_function=cl_cost
        )
        
        # Run QAOA with depth 5, max 50 iterations, and SPSA optimizer
        res_sample = qaoa_maxcut.run(qarg, depth=5, max_iter=50, optimizer="SPSA")

        return res_sample

    # Generate MLIR from the complete QAOA workflow
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)

    # Verify that MLIR contains expected JASP dialect operations
    assert "jasp.create_qubits" in mlir_str, "MLIR should contain jasp.create_qubits operation"
    assert "jasp.quantum_gate" in mlir_str, "MLIR should contain jasp.quantum_gate operations"
    assert "jasp.measure" in mlir_str, "MLIR should contain jasp.measure operation"
    
    # Verify that control flow has been rewritten from StableHLO to SCF for optimization loop
    assert "scf.while" in mlir_str, "MLIR should contain SCF while for QAOA optimization loop"
    assert "scf.yield" in mlir_str, "MLIR should contain SCF yield operations"
    
    # Verify that QAOA-specific operations are present
    assert "jasp.slice" in mlir_str or "jasp.get_qubit" in mlir_str, "MLIR should contain qubit indexing operations"

def test_mlir_array_operations():
    """
    Test MLIR generation for quantum array operations.
    Verifies that array slicing and fusion are properly lowered to MLIR.
    """
    from qrisp import QuantumVariable, h, cx
    from qrisp.jasp import make_jaspr, jrange
    
    def main():
        # Create quantum arrays
        qv1 = QuantumVariable(4)
        qv2 = QuantumVariable(3)
        
        # Test slicing operations
        h(qv1[0:2])
        cx(qv1[1], qv2[0])
        
        # Test iteration with jrange
        for i in jrange(3):
            h(qv2[i])
        
        return qv1, qv2
    
    # Test MLIR generation
    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)
    
    # Verify MLIR contains expected operations
    assert "jasp.create_qubits" in mlir_str, "Should contain qubit creation"
    assert "jasp.slice" in mlir_str or "jasp.get_qubit" in mlir_str, "Should contain array access operations"
    assert "jasp.quantum_gate" in mlir_str or "jasp.gate" in mlir_str, "Should contain quantum gates"
    

def test_mlir_jasp_dialect_registration():
    """
    Test that the JASP dialect is properly registered in xDSL.

    When the dialect is registered, types and op names are printed without
    surrounding quotes.  An unregistered op would appear as::

        "jasp.create_qubits"(...)

    A registered op appears as::

        jasp.create_qubits ...

    Same rule applies to types: ``!jasp.QuantumState`` vs
    ``!"jasp.QuantumState"``.
    """
    from qrisp import QuantumFloat, QuantumVariable, cx, h, measure, x
    from qrisp.jasp import make_jaspr

    def main():
        qv = QuantumVariable(3)
        qf = QuantumFloat(2)
        h(qv[0])
        cx(qv[0], qv[1])
        x(qf[0])
        result = measure(qv[0])
        return result

    jaspr = make_jaspr(main)()
    xdsl_module = jaspr.to_mlir()
    mlir_str = str(xdsl_module)

    # Types must appear without quotes
    assert (
        '!"jasp.QuantumState"' not in mlir_str
    ), "QuantumState type is unregistered (printed with quotes)"
    assert (
        '!"jasp.QubitArray"' not in mlir_str
    ), "QubitArray type is unregistered (printed with quotes)"
    assert (
        '!"jasp.Qubit"' not in mlir_str
    ), "Qubit type is unregistered (printed with quotes)"
    assert (
        "!jasp.QuantumState" in mlir_str
    ), "QuantumState type not found in MLIR output"

    # Op names must appear without quotes
    assert (
        '"jasp.create_qubits"' not in mlir_str
    ), "create_qubits op is unregistered (printed with quotes)"
    assert (
        '"jasp.quantum_gate"' not in mlir_str
    ), "quantum_gate op is unregistered (printed with quotes)"
    assert (
        '"jasp.measure"' not in mlir_str
    ), "measure op is unregistered (printed with quotes)"

    assert "jasp.create_qubits" in mlir_str, "create_qubits op not found in MLIR output"
    assert "jasp.quantum_gate" in mlir_str, "quantum_gate op not found in MLIR output"
    assert "jasp.measure" in mlir_str, "measure op not found in MLIR output"


def test_mlir_quantum_kernel_lifting():
    """
    Test that the quantum_kernel decorator produces a jasp.call at the call
    site, replacing the create/consume sentinel pair.  The callee stays as a
    func.func with !jasp.QuantumState in its signature.
    """
    from qrisp import QuantumFloat, measure
    from qrisp.jasp import make_jaspr, quantum_kernel
    from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir

    @quantum_kernel
    def inner(k):
        qf = QuantumFloat(4)
        return measure(qf)

    def main(k):
        return inner(k)

    jaspr = make_jaspr(main)(1)
    xdsl_module = jaspr_to_mlir(jaspr)
    mlir_str = str(xdsl_module)

    # The call site must be a jasp.call (purely classical).
    assert "jasp.call" in mlir_str, \
        "Expected jasp.call op in MLIR output"

    # The callee stays as a func.func (no jasp.quantum_kernel or jasp.module).
    assert "jasp.quantum_kernel" not in mlir_str, \
        "jasp.quantum_kernel should not appear — quantum kernels are func.func"
    assert "jasp.module" not in mlir_str, \
        "jasp.module should not appear — no container needed"
    assert "jasp.return" not in mlir_str, \
        "jasp.return should not appear — kernel uses func.return"

    # The quantum kernel must be a func.func with QuantumState in its signature.
    from qrisp.jasp.mlir.xdsl_dialect import QuantumStateType
    from xdsl.dialects.func import FuncOp
    top_level_ops = list(xdsl_module.body.blocks[0].ops)
    quantum_kernels = [
        op for op in top_level_ops
        if isinstance(op, FuncOp)
        and list(op.function_type.inputs)
        and isinstance(list(op.function_type.inputs)[-1], QuantumStateType)
    ]
    kernel_names = {op.sym_name.data for op in quantum_kernels}
    assert "inner" in kernel_names, \
        f"Expected 'inner' quantum kernel func.func, found: {kernel_names}"

    # The call site must NOT thread QuantumState explicitly.
    # A jasp.call line should only carry classical types.
    call_lines = [l for l in mlir_str.splitlines() if "jasp.call" in l]
    for line in call_lines:
        assert "QuantumState" not in line, \
            f"jasp.call should not mention QuantumState: {line}"


def test_mlir_opt_roundtrip():
    """
    Test that the JASP generic MLIR output is accepted by C++ mlir-opt.

    This test is skipped automatically when mlir-opt is not on PATH, so it
    can be enabled in CI environments where LLVM/MLIR is built with the JASP
    dialect.

    Two modes are supported:

    1. **Syntax-only** (default): runs with ``--allow-unregistered-dialect``
       to validate that the output is syntactically correct C++ MLIR, without
       requiring the compiled JASP dialect plugin.

    2. **Full dialect validation**: if the environment variable
       ``JASP_DIALECT_LIB`` is set to the path of the compiled
       ``libJaspDialect.so`` (built from
       ``src/qrisp/jasp/mlir/dialect_definition/CMakeLists.txt``), the test
       loads the plugin and runs *without* ``--allow-unregistered-dialect``,
       exercising full type and op verification against the TableGen spec.

    The "generic MLIR" string tested here is the intermediate representation
    produced by jaxlib (C++ MLIR) before xDSL re-parses it.  It uses quoted
    op names (``"jasp.create_qubits"(...)``) and quoted types
    (``!jasp.QuantumState``), which is the format any C++ MLIR tool would
    consume.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from io import StringIO

    import pytest

    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        pytest.skip("mlir-opt not found on PATH")

    from qrisp import QuantumVariable, cx, h, measure
    from qrisp.jasp import make_jaspr
    from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules
    from qrisp.jasp.mlir.jaxpr_lowering import lower_jaxpr_to_MLIR

    def main():
        qv = QuantumVariable(3)
        h(qv[0])
        cx(qv[0], qv[1])
        result = measure(qv[0])
        return result

    jaspr = make_jaspr(main)()
    mlir_module = lower_jaxpr_to_MLIR(jaspr, lowering_rules=jasp_lowering_rules)

    # Capture the generic MLIR string (all ops in quoted generic form)
    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    mlir_module.operation.print(print_generic_op_form=True)
    sys.stdout = old_stdout
    generic_mlir_str = captured.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(generic_mlir_str)
        tmp_path = f.name

    dialect_lib = os.environ.get("JASP_DIALECT_LIB")
    if dialect_lib:
        # Full JASP dialect validation: load the compiled plugin so that JASP
        # types/ops are validated against the TableGen spec.  Other dialects
        # (e.g. stablehlo) are still accepted as unregistered, because mlir-opt
        # does not bundle them.
        cmd = [
            mlir_opt,
            f"--load-dialect-plugin={dialect_lib}",
            "--allow-unregistered-dialect",
            tmp_path,
        ]
    else:
        # Syntax-only: accept unregistered dialects, validates MLIR structure.
        cmd = [mlir_opt, "--allow-unregistered-dialect", tmp_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"mlir-opt failed with exit code {result.returncode}.\n"
        f"command: {' '.join(cmd)}\n"
        f"stderr:\n{result.stderr}"
    )


def test_mlir_opt_roundtrip_reset():
    """
    Test that jasp.reset is syntactically valid C++ MLIR.

    Uses a circuit that resets a qubit after a gate, exercising the
    ResetOp lowering rule.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from io import StringIO

    import pytest

    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        pytest.skip("mlir-opt not found on PATH")

    from qrisp import QuantumVariable, cx, h, measure, reset
    from qrisp.jasp import make_jaspr
    from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules
    from qrisp.jasp.mlir.jaxpr_lowering import lower_jaxpr_to_MLIR

    def main():
        qv = QuantumVariable(2)
        h(qv[0])
        cx(qv[0], qv[1])
        reset(qv[1])
        result = measure(qv[0])
        return result

    jaspr = make_jaspr(main)()
    mlir_module = lower_jaxpr_to_MLIR(jaspr, lowering_rules=jasp_lowering_rules)

    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    mlir_module.operation.print(print_generic_op_form=True)
    sys.stdout = old_stdout
    generic_mlir_str = captured.getvalue()

    assert '"jasp.reset"' in generic_mlir_str, "jasp.reset not found in generic MLIR"

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(generic_mlir_str)
        tmp_path = f.name

    dialect_lib = os.environ.get("JASP_DIALECT_LIB")
    if dialect_lib:
        cmd = [mlir_opt, f"--load-dialect-plugin={dialect_lib}", "--allow-unregistered-dialect", tmp_path]
    else:
        cmd = [mlir_opt, "--allow-unregistered-dialect", tmp_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"mlir-opt failed with exit code {result.returncode}.\n"
        f"command: {' '.join(cmd)}\n"
        f"stderr:\n{result.stderr}"
    )


def test_mlir_opt_roundtrip_slice_fuse_getsize():
    """
    Test that jasp.slice, jasp.fuse, and jasp.get_size are syntactically
    valid C++ MLIR.

    - slice: produced by qv[start:stop] indexing on a QubitArray.
    - get_size: produced by qv[:] (slice with implicit stop).
    - fuse: produced by concatenating two DynamicQubitArrays with +.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from io import StringIO

    import pytest

    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        pytest.skip("mlir-opt not found on PATH")

    from qrisp import QuantumVariable, h, measure
    from qrisp.jasp import make_jaspr
    from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules
    from qrisp.jasp.mlir.jaxpr_lowering import lower_jaxpr_to_MLIR

    def main():
        qv1 = QuantumVariable(4)
        qv2 = QuantumVariable(2)
        h(qv1[0])
        # slice: qv1[1:3] → slice_p
        sub = qv1[1:3]
        h(sub[0])
        # fuse: concatenate the two registers → fuse_p
        fused = qv1.reg + qv2.reg
        # get_size: qv1[:] uses implicit stop → get_size_p internally
        full = qv1[:]
        result = measure(qv1[0])
        return result

    jaspr = make_jaspr(main)()
    mlir_module = lower_jaxpr_to_MLIR(jaspr, lowering_rules=jasp_lowering_rules)

    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    mlir_module.operation.print(print_generic_op_form=True)
    sys.stdout = old_stdout
    generic_mlir_str = captured.getvalue()

    assert '"jasp.slice"' in generic_mlir_str, "jasp.slice not found in generic MLIR"
    assert '"jasp.fuse"' in generic_mlir_str, "jasp.fuse not found in generic MLIR"
    assert '"jasp.get_size"' in generic_mlir_str, "jasp.get_size not found in generic MLIR"

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(generic_mlir_str)
        tmp_path = f.name

    dialect_lib = os.environ.get("JASP_DIALECT_LIB")
    if dialect_lib:
        cmd = [mlir_opt, f"--load-dialect-plugin={dialect_lib}", "--allow-unregistered-dialect", tmp_path]
    else:
        cmd = [mlir_opt, "--allow-unregistered-dialect", tmp_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"mlir-opt failed with exit code {result.returncode}.\n"
        f"command: {' '.join(cmd)}\n"
        f"stderr:\n{result.stderr}"
    )


def test_mlir_opt_roundtrip_parity():
    """
    Test that jasp.parity is syntactically valid C++ MLIR.

    Uses a GHZ-state circuit where parity of all qubit measurements
    is computed, exercising the ParityOp lowering rule.
    """
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from io import StringIO

    import pytest

    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        pytest.skip("mlir-opt not found on PATH")

    from qrisp import QuantumVariable, cx, h, measure
    from qrisp.jasp import make_jaspr, parity
    from qrisp.jasp.mlir.jasp_lowering_rules import jasp_lowering_rules
    from qrisp.jasp.mlir.jaxpr_lowering import lower_jaxpr_to_MLIR

    def main():
        qv = QuantumVariable(4)
        h(qv[0])
        cx(qv[0], qv[1])
        cx(qv[0], qv[2])
        cx(qv[0], qv[3])
        a = measure(qv[0])
        b = measure(qv[1])
        c = measure(qv[2])
        d = measure(qv[3])
        return parity(a, b, c, d)

    jaspr = make_jaspr(main)()
    mlir_module = lower_jaxpr_to_MLIR(jaspr, lowering_rules=jasp_lowering_rules)

    captured = StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    mlir_module.operation.print(print_generic_op_form=True)
    sys.stdout = old_stdout
    generic_mlir_str = captured.getvalue()

    assert '"jasp.parity"' in generic_mlir_str, "jasp.parity not found in generic MLIR"

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(generic_mlir_str)
        tmp_path = f.name

    dialect_lib = os.environ.get("JASP_DIALECT_LIB")
    if dialect_lib:
        cmd = [mlir_opt, f"--load-dialect-plugin={dialect_lib}", "--allow-unregistered-dialect", tmp_path]
    else:
        cmd = [mlir_opt, "--allow-unregistered-dialect", tmp_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"mlir-opt failed with exit code {result.returncode}.\n"
        f"command: {' '.join(cmd)}\n"
        f"stderr:\n{result.stderr}"
    )


def test_mlir_opt_roundtrip_quantum_kernel():
    """
    Test that the xDSL output containing jasp.call and quantum kernel
    func.func ops is syntactically valid C++ MLIR.

    Unlike the other roundtrip tests (which validate the pre-xDSL JAX MLIR),
    this test runs the *full* pipeline — including lift_quantum_kernels and
    drop_dead_wrappers — and then validates the resulting xDSL module.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    from io import StringIO

    import pytest

    mlir_opt = shutil.which("mlir-opt")
    if mlir_opt is None:
        pytest.skip("mlir-opt not found on PATH")

    from xdsl.printer import Printer

    from qrisp import QuantumFloat, cx, h, measure
    from qrisp.jasp import make_jaspr, quantum_kernel
    from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir
    import jax.numpy as jnp

    @quantum_kernel
    def sampling_kernel(k):
        qf = QuantumFloat(4)
        h(qf[k])
        cx(qf[k], qf[0])
        return measure(qf)

    def main(k):
        return sampling_kernel(k)

    jaspr = make_jaspr(main)(jnp.int64(1))
    xdsl_module = jaspr_to_mlir(jaspr)

    # Verify expected ops are present in the xDSL output.
    mlir_str = str(xdsl_module)
    assert "jasp.call" in mlir_str, "Expected jasp.call in xDSL output"
    assert "func.func" in mlir_str, "Expected func.func in xDSL output"

    # Serialise in generic form so mlir-opt can parse it without the custom
    # assembly format implementation (--allow-unregistered-dialect).
    buf = StringIO()
    printer = Printer(stream=buf, print_generic_format=True)
    printer.print_op(xdsl_module)
    generic_mlir_str = buf.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(generic_mlir_str)
        tmp_path = f.name

    dialect_lib = os.environ.get("JASP_DIALECT_LIB")
    if dialect_lib:
        cmd = [
            mlir_opt,
            f"--load-dialect-plugin={dialect_lib}",
            "--allow-unregistered-dialect",
            tmp_path,
        ]
    else:
        cmd = [mlir_opt, "--allow-unregistered-dialect", tmp_path]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"mlir-opt failed with exit code {result.returncode}.\n"
        f"command: {' '.join(cmd)}\n"
        f"stderr:\n{result.stderr}"
    )
