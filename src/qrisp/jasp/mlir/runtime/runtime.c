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
* Minimal C runtime for JASP programs compiled via MLIR/EmitC.
*
* This file is the stable C boundary between EmitC-generated code and the
* QDMI backend:
*
*   MLIR -> EmitC -> C -> runtime.c -> QDMI
*
* Build (assuming QDMI_ROOT points to the QDMI checkout):
*
*   clang -std=c11 -Wall -Wextra \
*       -I. -I$(QDMI_ROOT)/include -I$(QDMI_ROOT)/examples/driver \
*       output.c runtime.c \
*       $(QDMI_ROOT)/build/examples/driver/libqdmi_example_driver.a \
*       -ldl -lstdc++ -o program
*
* Note: -lstdc++ is a transitive dependency of the QDMI example driver
* (which is implemented in C++).  This runtime itself is pure C.
*/

#include "runtime_internal.h"

#include "qdmi_example_driver.h"
#include "qdmi/client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Internal state — no QDMI types escape this translation unit.
 * ----------------------------------------------------------------------- */

static int runtime_initialized = 0;
static int runtime_debug = 0;
static size_t runtime_shots = 1024;
static const char *runtime_token = "demo";

static QDMI_Session qdmi_session = NULL;
static QDMI_Device qdmi_device = NULL;

/* -----------------------------------------------------------------------
 * Internal configuration (runtime_internal.h)
 * ----------------------------------------------------------------------- */

void runtime_set_shots(size_t shots) {
    if (shots > 0) {
        runtime_shots = shots;
    }
}

void runtime_set_token(const char *token) {
    if (token) {
        runtime_token = token;
    }
}

/* -----------------------------------------------------------------------
 * Initialization (runtime.h)
 * ----------------------------------------------------------------------- */

void runtime_init(void) {
    if (runtime_initialized) return;

    /* 1. Start the QDMI driver (reads qdmi.conf) */
    if (QDMI_driver_init() != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: QDMI driver init failed\n");
        return;
    }

    /* 2. Allocate and configure a session */
    if (QDMI_session_alloc(&qdmi_session) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: QDMI session alloc failed\n");
        QDMI_driver_shutdown();
        return;
    }

    if (QDMI_session_set_parameter(qdmi_session, QDMI_SESSION_PARAMETER_TOKEN,
                                   strlen(runtime_token) + 1,
                                   runtime_token) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: QDMI set token failed\n");
        QDMI_session_free(qdmi_session);
        qdmi_session = NULL;
        QDMI_driver_shutdown();
        return;
    }

    if (QDMI_session_init(qdmi_session) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: QDMI session init failed\n");
        QDMI_session_free(qdmi_session);
        qdmi_session = NULL;
        QDMI_driver_shutdown();
        return;
    }

    /* 3. Get the first available device */
    size_t devices_size = 0;
    if (QDMI_session_query_session_property(
            qdmi_session, QDMI_SESSION_PROPERTY_DEVICES,
            0, NULL, &devices_size) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: QDMI query devices size failed\n");
        QDMI_session_free(qdmi_session);
        qdmi_session = NULL;
        QDMI_driver_shutdown();
        return;
    }

    if (QDMI_session_query_session_property(
            qdmi_session, QDMI_SESSION_PROPERTY_DEVICES,
            devices_size, &qdmi_device, NULL) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: QDMI query devices failed\n");
        QDMI_session_free(qdmi_session);
        qdmi_session = NULL;
        QDMI_driver_shutdown();
        return;
    }

    if (runtime_debug) {
        size_t name_size = 0;
        QDMI_device_query_device_property(qdmi_device,
                                          QDMI_DEVICE_PROPERTY_NAME,
                                          0, NULL, &name_size);
        char name_buf[128];
        if (name_size > 0 && name_size <= sizeof(name_buf)) {
            QDMI_device_query_device_property(qdmi_device,
                                              QDMI_DEVICE_PROPERTY_NAME,
                                              name_size, name_buf, NULL);
            fprintf(stderr, "runtime: device = %s\n", name_buf);
        }
    }

    runtime_initialized = 1;
}

void runtime_init_with_config(RuntimeConfig *config) {
    if (runtime_initialized) return;

    if (config) {
        runtime_debug = config->debug;
        (void)config->device_index; /* future: select Nth device */
    }

    runtime_init();
}

/* -----------------------------------------------------------------------
 * Kernel execution (runtime_internal.h)
 * ----------------------------------------------------------------------- */

int64_t run_jasp_kernel(const char *kernel_mlir) {
    if (!runtime_initialized || qdmi_device == NULL) {
        fprintf(stderr, "runtime: not initialized\n");
        return -1;
    }

    if (runtime_debug) {
        fprintf(stderr, "runtime: submitting kernel (%zu bytes)\n",
                strlen(kernel_mlir));
    }

    /* 1. Create a job on the device */
    QDMI_Job job = NULL;
    if (QDMI_device_create_job(qdmi_device, &job) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: create job failed\n");
        return -1;
    }

    /* 2. Set program format (QASM2 for now; will switch to a native
     *    MLIR/QIR format once QDMI supports it) */
    QDMI_Program_Format fmt = QDMI_PROGRAM_FORMAT_QASM2;
    if (QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                               sizeof(fmt), &fmt) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: set format failed\n");
        QDMI_job_free(job);
        return -1;
    }

    /* 3. Set the kernel program */
    if (QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAM,
                               strlen(kernel_mlir) + 1,
                               kernel_mlir) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: set program failed\n");
        QDMI_job_free(job);
        return -1;
    }

    /* 4. Set number of shots */
    if (QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_SHOTSNUM,
                               sizeof(runtime_shots),
                               &runtime_shots) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: set shots failed\n");
        QDMI_job_free(job);
        return -1;
    }

    /* 5. Submit and wait */
    if (QDMI_job_submit(job) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: submit failed\n");
        QDMI_job_free(job);
        return -1;
    }

    if (QDMI_job_wait(job, 0) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: wait failed\n");
        QDMI_job_free(job);
        return -1;
    }

    /* 6. Retrieve shot results into a local buffer and parse */
    size_t needed = 0;
    if (QDMI_job_get_results(job, QDMI_JOB_RESULT_SHOTS,
                             0, NULL, &needed) != QDMI_SUCCESS) {
        fprintf(stderr, "runtime: query result size failed\n");
        QDMI_job_free(job);
        return -1;
    }

    int64_t result = 0;
    if (needed > 0) {
        char *buf = (char *)malloc(needed + 1);
        if (buf == NULL) {
            fprintf(stderr, "runtime: allocation failed\n");
            QDMI_job_free(job);
            return -1;
        }
        if (QDMI_job_get_results(job, QDMI_JOB_RESULT_SHOTS,
                                 needed, buf, NULL) != QDMI_SUCCESS) {
            fprintf(stderr, "runtime: get results failed\n");
            free(buf);
            QDMI_job_free(job);
            return -1;
        }
        buf[needed] = '\0';
        result = strtoll(buf, NULL, 10);
        free(buf);
    }

    QDMI_job_free(job);
    return result;
}

/* -----------------------------------------------------------------------
 * Cleanup (runtime.h)
 * ----------------------------------------------------------------------- */

void runtime_cleanup(void) {
    if (!runtime_initialized) return;

    qdmi_device = NULL;

    if (qdmi_session != NULL) {
        QDMI_session_free(qdmi_session);
        qdmi_session = NULL;
    }

    QDMI_driver_shutdown();

    runtime_debug = 0;
    runtime_shots = 1024;
    runtime_token = "demo";
    runtime_initialized = 0;
}
