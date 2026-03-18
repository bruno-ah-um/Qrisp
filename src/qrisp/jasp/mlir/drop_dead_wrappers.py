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
xDSL pass: drop_dead_wrappers
==============================
Erases private ``func.func`` ops that have no callers anywhere in the module.

JAX's ``lower_jaxpr_to_fun`` emits a thin ``func.func`` wrapper for every
primitive it encounters (e.g. ``@jasp.create_qubits``, ``@jasp.measure``).
Our custom lowering rules emit the JASP ops *inline* instead of via
``func.call``, so these wrappers are never referenced and become dead code.

This pass removes them, leaving only functions that are actually called.
"""

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import CallOp, FuncOp

from qrisp.jasp.mlir.xdsl_dialect import JaspCallOp


def _collect_called_symbols(module: ModuleOp) -> set[str]:
    """Return the set of symbol names referenced by any call in the module.

    Checks both ``func.call`` and ``jasp.call`` — the latter is used to
    invoke quantum kernel ``func.func`` ops.
    """
    called: set[str] = set()
    for block in module.body.blocks:
        for op in block.walk():
            if isinstance(op, (CallOp, JaspCallOp)):
                called.add(op.callee.root_reference.data)
    return called


def drop_dead_wrappers(module: ModuleOp) -> None:
    """Erase private func.func ops with no callers from *module*.

    Mutates *module* in-place.
    """
    called_symbols = _collect_called_symbols(module)

    module_block = module.body.blocks.first
    if module_block is None:
        return

    dead = [
        op for op in module_block.ops
        if isinstance(op, FuncOp)
        and op.sym_visibility is not None
        and op.sym_visibility.data == "private"
        and op.sym_name.data not in called_symbols
    ]

    for op in dead:
        module_block.detach_op(op)
