/*
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
*
* Internal runtime API for EmitC-generated JASP programs.
*
* These functions are called by the lowered EmitC code (e.g. via
* emitc.call_opaque "run_jasp_kernel") but are NOT part of the stable
* public contract defined in runtime.h.
*/

#ifndef RUNTIME_INTERNAL_H
#define RUNTIME_INTERNAL_H

#include "runtime.h"

#include <stddef.h>
#include <stdint.h>

/*
 * Execute a JASP quantum kernel via QDMI.
 *
 * Called from EmitC-generated code as:
 *   emitc.call_opaque "run_jasp_kernel"(kernel_mlir)
 *
 * Parameters:
 *   kernel_mlir  - serialized MLIR text of the quantum kernel
 *
 * Returns the measurement result as an integer.
 */
int64_t run_jasp_kernel(const char *kernel_mlir);

/*
 * Configure the number of shots used by run_jasp_kernel.
 * Must be called before runtime_init (or between cleanup/init cycles).
 * Default: 1024.
 */
void runtime_set_shots(size_t shots);

/*
 * Configure the session token used for QDMI authentication.
 * Must be called before runtime_init (or between cleanup/init cycles).
 * The string is NOT copied — caller must keep it alive.
 * Default: "demo".
 */
void runtime_set_token(const char *token);

#endif /* RUNTIME_INTERNAL_H */
