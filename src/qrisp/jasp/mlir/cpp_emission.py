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

C++ emission pipeline for JASP programs.

Compiles a Jaspr to C++ source code by:

1. Running the standard JASP MLIR pipeline (``jaspr_to_mlir``).
2. Lowering quantum kernel calls to QDMI via EmitC (``lower_to_emitc`` passes).
3. Serialising the EmitC module as MLIR text.
4. Rewriting ``func.func`` / ``func.return`` to ``emitc.func`` / ``emitc.return``
   in the textual IR (xDSL lacks these ops).
5. Translating the EmitC MLIR to C++ via ``mlir-translate --mlir-to-cpp``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from io import StringIO
from pathlib import Path

from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from qrisp.jasp.jasp_expression import Jaspr
from qrisp.jasp.mlir.mlir_emission import jaspr_to_mlir
from qrisp.jasp.mlir.lower_to_emitc import (
    lower_classical_to_emitc,
    lower_jasp_call_to_qdmi,
    strip_quantum_state_from_main,
)


# ---------------------------------------------------------------------------
# Textual IR fixups
# ---------------------------------------------------------------------------

def _protect_opaque_strings(mlir_text: str):
    """Extract ``#emitc.opaque<"...">`` strings and replace with placeholders.

    Returns (modified_text, list_of_extracted_strings).
    The caller can perform substitutions on the modified text without
    corrupting the opaque attribute contents, then restore them.
    """
    placeholders = []
    def _replace(m):
        placeholders.append(m.group(0))
        return f"__OPAQUE_PLACEHOLDER_{len(placeholders) - 1}__"
    # Match #emitc.opaque<"..."> — the inner string may contain escaped quotes.
    modified = re.sub(r'#emitc\.opaque<"(?:[^"\\]|\\.)*">', _replace, mlir_text)
    return modified, placeholders


def _restore_opaque_strings(mlir_text: str, placeholders: list[str]) -> str:
    """Restore opaque attribute strings from placeholders."""
    for i, original in enumerate(placeholders):
        mlir_text = mlir_text.replace(f"__OPAQUE_PLACEHOLDER_{i}__", original)
    return mlir_text


def _func_to_emitc_func(mlir_text: str) -> str:
    """Rewrite ``func.func`` / ``func.return`` to ``emitc.func`` / ``emitc.return``.

    xDSL does not define ``emitc.func`` or ``emitc.return``, so we perform
    the substitution on the serialised MLIR text before handing it to
    ``mlir-translate``.

    Also strips visibility keywords (``public`` / ``private``) from function
    declarations — EmitC functions do not use MLIR visibility.

    Opaque attribute strings (containing serialised kernel MLIR) are protected
    from substitution.
    """
    mlir_text, placeholders = _protect_opaque_strings(mlir_text)

    # func.func [public|private] @name → emitc.func @name
    mlir_text = re.sub(
        r'\bfunc\.func\s+(?:public\s+|private\s+)?(@\w+)',
        r'emitc.func \1',
        mlir_text,
    )
    # func.return → emitc.return
    mlir_text = mlir_text.replace("func.return", "emitc.return")

    mlir_text = _restore_opaque_strings(mlir_text, placeholders)
    return mlir_text


def _detensorize_text(mlir_text: str) -> str:
    """Replace residual ``tensor<T>`` types with bare scalar ``T`` in MLIR text.

    After the xDSL passes some operand types may still print as ``tensor<i64>``
    (e.g. on emitc.call_opaque arguments whose SSA values have not been
    re-typed).  ``mlir-translate`` requires scalar types for EmitC, so we fix
    them here.

    Opaque attribute strings are protected from substitution.
    """
    mlir_text, placeholders = _protect_opaque_strings(mlir_text)

    mlir_text = re.sub(r'\btensor<i64>', 'i64', mlir_text)
    mlir_text = re.sub(r'\btensor<i32>', 'i32', mlir_text)
    mlir_text = re.sub(r'\btensor<i1>', 'i1', mlir_text)
    mlir_text = re.sub(r'\btensor<f64>', 'f64', mlir_text)
    mlir_text = re.sub(r'\btensor<f32>', 'f32', mlir_text)

    mlir_text = _restore_opaque_strings(mlir_text, placeholders)
    return mlir_text


# ---------------------------------------------------------------------------
# C++ post-processing
# ---------------------------------------------------------------------------

def _load_runtime_source() -> str:
    """Read and inline the runtime source files into a single C block.

    Reads ``runtime.h``, ``runtime_internal.h``, and ``runtime.c`` from the
    ``runtime/`` directory next to this module, strips their mutual
    ``#include`` directives (since they are being concatenated), and returns
    the combined source text.

    The only remaining external ``#include`` directives are for QDMI and
    standard-library headers.
    """
    runtime_dir = Path(__file__).parent / "runtime"

    runtime_h = (runtime_dir / "runtime.h").read_text()
    runtime_internal_h = (runtime_dir / "runtime_internal.h").read_text()
    runtime_c = (runtime_dir / "runtime.c").read_text()

    # Strip local includes — they are concatenated here.
    runtime_internal_h = re.sub(
        r'#include\s+"runtime\.h"\s*\n', '', runtime_internal_h,
    )
    runtime_c = re.sub(
        r'#include\s+"runtime_internal\.h"\s*\n', '', runtime_c,
    )

    return runtime_h + "\n" + runtime_internal_h + "\n" + runtime_c


_COMPILE_COMMAND = (
    "// Compile with (set QDMI_ROOT to your QDMI checkout):\n"
    "//   clang -std=c11 -I$QDMI_ROOT/include -I$QDMI_ROOT/examples/driver \\\n"
    "//     thisfile.c $QDMI_ROOT/build/examples/driver/"
    "libqdmi_example_driver.a \\\n"
    "//     -ldl -lstdc++ -o program\n"
)


def _inject_runtime_calls(cpp_code: str) -> str:
    """Inline the full QDMI runtime and wrap ``main()`` with lifecycle calls.

    Prepends a compile-command comment, the inlined runtime source
    (``runtime.h`` + ``runtime_internal.h`` + ``runtime.c``), and inserts
    ``runtime_init()`` / ``runtime_cleanup()`` into ``main()``.

    The result is a single, self-contained C file that only requires the
    QDMI headers and driver library at compile time.
    """
    runtime_source = _load_runtime_source()
    cpp_code = _COMPILE_COMMAND + "\n#include <stdint.h>\n\n" + runtime_source + "\n" + cpp_code

    # mlir-translate emits ``int64_t main()`` for an i64-returning entry
    # point, but C requires ``main`` to return ``int``.
    cpp_code = re.sub(r'\bint64_t(\s+main\s*\()', r'int\1', cpp_code)

    # Find main() and inject runtime_init / runtime_cleanup.
    # Match the opening brace of main's body.
    main_match = re.search(r'(\w+\s+main\s*\([^)]*\)\s*\{)', cpp_code)
    if main_match is None:
        return cpp_code

    # Insert runtime_init() right after the opening brace.
    insert_pos = main_match.end()
    cpp_code = cpp_code[:insert_pos] + "\n  runtime_init();" + cpp_code[insert_pos:]

    # Find the end of main() by brace matching from the opening brace.
    brace_start = main_match.start() + main_match.group(0).index('{')
    depth = 0
    main_end = len(cpp_code)
    for i in range(brace_start, len(cpp_code)):
        if cpp_code[i] == '{':
            depth += 1
        elif cpp_code[i] == '}':
            depth -= 1
            if depth == 0:
                main_end = i
                break

    # Insert runtime_cleanup() before each return in main's body.
    main_body = cpp_code[brace_start:main_end]
    main_body = re.sub(
        r'(\n(\s*))return\b',
        r'\1runtime_cleanup();\1return',
        main_body,
    )
    cpp_code = cpp_code[:brace_start] + main_body + cpp_code[main_end:]

    return cpp_code


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _find_mlir_translate() -> str:
    """Locate the ``mlir-translate`` binary on the system."""
    for name in ("mlir-translate", "mlir-translate-19", "mlir-translate-18"):
        path = shutil.which(name)
        if path is not None:
            return path
    raise FileNotFoundError(
        "mlir-translate not found on PATH. "
        "Install MLIR tools (e.g. apt install mlir-19-tools)."
    )


def jaspr_to_emitc(jaspr: Jaspr) -> ModuleOp:
    """Run the full JASP → EmitC lowering pipeline, returning the xDSL module.

    This is useful for inspecting the intermediate EmitC IR before C++
    translation.
    """
    module = jaspr_to_mlir(jaspr)
    strip_quantum_state_from_main(module)
    lower_jasp_call_to_qdmi(module)
    lower_classical_to_emitc(module)
    return module


def jaspr_to_cpp(jaspr: Jaspr) -> str:
    """Compile a Jaspr all the way to C++ source code.

    Pipeline:
        jaspr_to_mlir → strip_quantum_state → lower_jasp_call_to_qdmi
        → lower_classical_to_emitc → textual fixups → mlir-translate --mlir-to-cpp

    Returns the C++ source as a string.
    """
    module = jaspr_to_emitc(jaspr)

    # Serialise to MLIR text (custom format so func.func/func.return
    # print in their standard form, which our regex can handle).
    buf = StringIO()
    printer = Printer(stream=buf)
    printer.print(module)
    mlir_text = buf.getvalue()

    # Textual fixups for ops xDSL cannot represent natively.
    mlir_text = _func_to_emitc_func(mlir_text)
    mlir_text = _detensorize_text(mlir_text)

    # Translate to C++.
    translate_bin = _find_mlir_translate()
    result = subprocess.run(
        [translate_bin, "--mlir-to-cpp"],
        input=mlir_text,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mlir-translate failed (exit {result.returncode}):\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- input IR ---\n{mlir_text}"
        )

    cpp_code = result.stdout
    cpp_code = _inject_runtime_calls(cpp_code)
    return cpp_code
