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
* Stable public C API for the JASP runtime.
*
* This header is the contract between EmitC-generated code and the runtime.
* It exposes only plain C types — no QDMI types leak through here.
*
* MLIR contract:
*   func.func private @runtime_init() -> ()
*   func.func private @runtime_cleanup() -> ()
*/

#ifndef RUNTIME_H
#define RUNTIME_H

typedef struct {
    int debug;
    int device_index;
} RuntimeConfig;

void runtime_init(void);
void runtime_init_with_config(RuntimeConfig *config);
void runtime_cleanup(void);

#endif /* RUNTIME_H */
